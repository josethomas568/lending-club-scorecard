"""Step 3c — benchmark vs challenger: KS, AUC/Gini, lift tables, calibration,
and reason codes (scorecard points shortfall; LightGBM SHAP).

Reason codes
  Scorecard : for each characteristic, points shortfall = max attainable points
              minus points awarded; the top 4 shortfalls are the reasons
              (the standard adverse-action method for points scorecards).
  LightGBM  : TreeSHAP contributions in log-odds (LightGBM pred_contrib, exact
              for trees); the top 4 characteristics pushing PD up are the reasons.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from lcscore import plots
from lcscore.config import load_config, load_json, save_json
from lcscore.features import FEATURES, LABELS, MODEL_SAMPLES, load_abt
from lcscore.metrics import calibration_table, ece, lift_table, summary
from lcscore.plots import CHALLENGER, REFERENCE, SCORECARD, plt

N_REASONS = 4
MODELS = {"Scorecard (WoE logistic)": SCORECARD, "LightGBM challenger": CHALLENGER}


def main():
    cfg = load_config()
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    R, F = P["results_dir"], P["figures_dir"]
    sc = pd.read_parquet(P["model_dir"] / "scores_scorecard.parquet")
    ch = pd.read_parquet(P["model_dir"] / "scores_challenger.parquet")
    df = sc.merge(ch[["id", "pd"]].rename(columns={"pd": "pd_lgbm"}), on="id")
    df = df.rename(columns={"pd": "pd_scorecard"})
    pcols = {"Scorecard (WoE logistic)": "pd_scorecard", "LightGBM challenger": "pd_lgbm"}

    # ---- discrimination and calibration by sample ---------------------------
    rows = []
    for s in MODEL_SAMPLES:
        d = df[df["sample"] == s]
        for m, c in pcols.items():
            rows.append({"model": m, "sample": s, **summary(d.bad_12m, d[c]),
                         "ece": ece(d.bad_12m, d[c])})
    perf = pd.DataFrame(rows)
    perf.to_csv(R / "performance.csv", index=False)

    oot = df[df["sample"] == "oot"]
    tests = {"in-time test": df[df["sample"] == "test_in_time"], "out-of-time": oot}
    for m, c in pcols.items():
        tag = "scorecard" if c == "pd_scorecard" else "lgbm"
        lift_table(oot.bad_12m, oot[c]).to_csv(R / f"lift_oot_{tag}.csv", index=False)
        for name, d in tests.items():
            calibration_table(d.bad_12m, d[c]).to_csv(
                R / f"calibration_{name.replace(' ', '_').replace('-', '_')}_{tag}.csv", index=False)

    # ROC + KS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for m, c in pcols.items():
        fpr, tpr, _ = roc_curve(oot.bad_12m, oot[c])
        auc = perf.query("model == @m and sample == 'oot'").auc.iloc[0]
        axes[0].plot(fpr, tpr, color=MODELS[m], label=f"{m}  AUC {auc:.3f}")
        lt = lift_table(oot.bad_12m, oot[c], 20)
        x = np.r_[0, lt.loans.cumsum() / lt.loans.sum()]
        axes[1].plot(x, np.r_[0, lt.cum_bads_captured], color=MODELS[m], label=m)
    axes[0].plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--")
    axes[0].set(title="ROC — out-of-time", xlabel="False positive rate (goods declined)",
                ylabel="True positive rate (bads caught)")
    axes[1].plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--", label="Random")
    axes[1].set(title="Cumulative bads captured — out-of-time",
                xlabel="Share of applicants, riskiest first", ylabel="Share of bads captured")
    for ax in axes:
        ax.legend(loc="lower right")
    plots.save(fig, F / "roc_capture_oot.png")

    # Calibration plots: in-time vs out-of-time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, (name, d) in zip(axes, tests.items()):
        top = 0
        for m, c in pcols.items():
            t = calibration_table(d.bad_12m, d[c])
            ax.plot(t.mean_pd, t.observed, marker="o", ms=5, color=MODELS[m], label=m)
            top = max(top, t.mean_pd.max(), t.observed.max())
        ax.plot([0, top], [0, top], color=REFERENCE, lw=1, ls="--", label="Perfect calibration")
        ax.set(title=f"Calibration — {name}", xlabel="Mean predicted PD (decile)",
               ylabel="Observed 12-month bad rate")
        ax.legend(loc="upper left")
    plots.save(fig, F / "calibration.png")

    # Score distribution (scorecard), goods vs bads, OOT
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.arange(oot.score.min() - 2, oot.score.max() + 6, 5)
    ax.hist(oot.score[oot.bad_12m == 0], bins=bins, density=True, color=SCORECARD, histtype="step", lw=2, label="Good")
    ax.hist(oot.score[oot.bad_12m == 1], bins=bins, density=True, color=CHALLENGER, histtype="step", lw=2, label="Bad")
    ks = perf.query("model == 'Scorecard (WoE logistic)' and sample == 'oot'").ks.iloc[0]
    ax.set(title=f"Scorecard score distribution — out-of-time (KS {ks:.3f})",
           xlabel="Score (higher = lower risk)", ylabel="Density")
    ax.legend()
    plots.save(fig, F / "score_distribution_oot.png")

    # ---- reason codes ------------------------------------------------------
    pts_cols = [c for c in sc.columns if c.startswith("pts_")]
    points = pd.read_csv(P["model_dir"] / "scorecard_points.csv")
    max_pts = points.groupby("feature").points.max()
    shortfall = pd.DataFrame({c[4:]: max_pts[c[4:]] - oot[c] for c in pts_cols}, index=oot.index)
    sc_reasons = _top_reasons(shortfall)

    booster = lgb.Booster(model_file=str(P["model_dir"] / "challenger_lgbm.txt"))
    manifest = load_json(P["model_dir"] / "challenger_manifest.json")
    abt = load_abt(P["abt_path"]).set_index("id").loc[oot.id]
    X = abt[FEATURES].copy()
    for c in manifest["categorical"]:
        X[c] = pd.Categorical(X[c], categories=manifest["categorical_levels"][c])
    contrib = booster.predict(X, pred_contrib=True)
    shap = pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=oot.index)
    # SHAP additivity check: contributions + expected value = raw log-odds score
    raw = booster.predict(X, raw_score=True)
    additivity_err = float(np.max(np.abs(contrib.sum(axis=1) - raw)))
    lg_reasons = _top_reasons(shap.where(shap > 0, 0))

    reasons = pd.DataFrame({"id": oot.id.values, "score": oot.score.values,
                            "pd_scorecard": oot.pd_scorecard.values, "pd_lgbm": oot.pd_lgbm.values})
    for i in range(N_REASONS):
        reasons[f"scorecard_reason_{i+1}"] = [LABELS.get(r[i], "") if len(r) > i else ""
                                              for r in sc_reasons]
        reasons[f"lgbm_reason_{i+1}"] = [LABELS.get(r[i], "") if len(r) > i else ""
                                         for r in lg_reasons]
    # agreement on applicants a lender would actually decline: riskiest 20% by scorecard
    risky = (oot.score <= oot.score.quantile(0.2)).to_numpy()
    top1 = np.mean([a[:1] == b[:1] for a, b, k in zip(sc_reasons, lg_reasons, risky) if k])
    overlap = np.mean([len(set(a) & set(b)) / max(len(a), 1)
                       for a, b, k in zip(sc_reasons, lg_reasons, risky) if k])
    reasons[risky].sort_values("score").head(25).to_csv(R / "reason_codes_sample.csv", index=False)

    imp = shap.abs().mean().sort_values()
    imp.rename("mean_abs_shap").to_csv(R / "shap_importance.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh([LABELS[f] for f in imp.index], imp.values, color=CHALLENGER, height=0.6)
    ax.set(title="LightGBM — mean |SHAP| (log-odds), out-of-time", xlabel="Mean |SHAP value|")
    ax.grid(axis="y", visible=False)
    plots.save(fig, F / "shap_importance.png")

    meta = {"reason_code_top1_agreement_decline_zone": float(top1),
            "reason_code_top4_overlap_decline_zone": float(overlap),
            "shap_additivity_max_abs_error": additivity_err}
    save_json(meta, R / "reason_code_meta.json")
    print(perf[["model", "sample", "n", "bad_rate", "mean_pd", "auc", "gini", "ks", "ece"]]
          .round(4).to_string(index=False))
    print(meta)


def _top_reasons(frame: pd.DataFrame) -> list[list[str]]:
    vals, cols = frame.to_numpy(), np.array(frame.columns)
    order = np.argsort(-vals, axis=1, kind="mergesort")[:, :N_REASONS]
    return [[cols[j] for j in row if vals[i, j] > 0] for i, row in enumerate(order)]


if __name__ == "__main__":
    main()
