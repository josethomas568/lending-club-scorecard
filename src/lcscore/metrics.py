"""Discrimination, calibration and stability metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def ks_stat(y, pd_hat) -> float:
    # evaluated only at distinct score thresholds, so tied scores are never split
    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(pd_hat))
    return float(np.max(tpr - fpr))


def summary(y, pd_hat) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(pd_hat), 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, p)
    return {"n": int(len(y)), "bads": int(y.sum()), "bad_rate": float(y.mean()),
            "mean_pd": float(p.mean()), "auc": float(auc), "gini": float(2 * auc - 1),
            "ks": ks_stat(y, p), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p))}


def lift_table(y, pd_hat, n_bins=10) -> pd.DataFrame:
    """Decile 1 = riskiest 10%."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat)})
    df["decile"] = pd.qcut(df.p.rank(method="first", ascending=False), n_bins,
                           labels=range(1, n_bins + 1)).astype(int)
    t = df.groupby("decile").agg(loans=("y", "size"), bads=("y", "sum"),
                                 min_pd=("p", "min"), max_pd=("p", "max"),
                                 mean_pd=("p", "mean")).reset_index()
    t["bad_rate"] = t.bads / t.loans
    t["cum_bads_captured"] = t.bads.cumsum() / t.bads.sum()
    t["cum_goods"] = (t.loans - t.bads).cumsum() / (t.loans - t.bads).sum()
    t["ks"] = t.cum_bads_captured - t.cum_goods
    t["lift"] = t.bad_rate / df.y.mean()
    t["cum_lift"] = t.cum_bads_captured / (t.loans.cumsum() / t.loans.sum())
    return t


def calibration_table(y, pd_hat, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=range(1, n_bins + 1)).astype(int)
    t = df.groupby("bin").agg(loans=("y", "size"), mean_pd=("p", "mean"),
                              observed=("y", "mean")).reset_index()
    se = np.sqrt(t.mean_pd * (1 - t.mean_pd) / t.loans)
    t["z"] = (t.observed - t.mean_pd) / se
    return t


def ece(y, pd_hat, n_bins=10) -> float:
    t = calibration_table(y, pd_hat, n_bins)
    return float(np.sum(t.loans * np.abs(t.observed - t.mean_pd)) / t.loans.sum())


def psi(expected_share, actual_share) -> float:
    e = np.clip(np.asarray(expected_share, float), 1e-4, None)
    a = np.clip(np.asarray(actual_share, float), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_from_values(expected, actual, edges) -> float:
    e = np.bincount(np.searchsorted(edges, expected, side="left"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, actual, side="left"), minlength=len(edges) + 1)
    return psi(e / e.sum(), a / a.sum())


def traffic_light(value, amber, red) -> str:
    return "red" if value >= red else "amber" if value >= amber else "green"
