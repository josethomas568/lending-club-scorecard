"""Step 4 — monitoring by issue quarter, against the development (train) baseline.

* Score PSI: scorecard score and LightGBM PD, 10 bands cut at baseline deciles.
* Characteristic stability: PSI of each scorecard characteristic's bin mix, plus
  the characteristic-analysis points shift  sum_bins (actual% - expected%) x points,
  i.e. how many score points the population mix moved for that characteristic.
* Score drift: mean and P10/P50/P90 of score per quarter.
* Back-test: predicted vs realised 12-month bad rate per quarter (mature quarters only).
* Vintage curves: cumulative bad rate (lifetime bad event, 90+ DPD estimate) by
  months on book for each issue-year vintage, plotted only where observed.

Traffic lights: PSI < 0.10 green, 0.10–0.25 amber, >= 0.25 red.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from lcscore import plots
from lcscore.config import load_config
from lcscore.features import LABELS
from lcscore.metrics import psi, psi_from_values, traffic_light
from lcscore.plots import CHALLENGER, INK2, REFERENCE, SCORECARD, SERIES, STATUS, plt


def main():
    cfg = load_config()
    mc = cfg["monitoring"]
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    R, F = P["results_dir"], P["figures_dir"]
    sc = pd.read_parquet(P["model_dir"] / "scores_scorecard.parquet")
    ch = pd.read_parquet(P["model_dir"] / "scores_challenger.parquet")[["id", "pd"]]
    df = sc.merge(ch.rename(columns={"pd": "pd_lgbm"}), on="id")
    points = pd.read_csv(P["model_dir"] / "scorecard_points.csv")
    chars = list(points.feature.unique())
    base = df[df["sample"] == "train"]
    q = np.linspace(0, 1, mc["score_bins"] + 1)[1:-1]
    score_edges = np.unique(np.quantile(base.score, q))
    lgbm_edges = np.unique(np.quantile(base.pd_lgbm, q))

    rows, csi_rows = [], []
    for quarter, d in df.groupby("issue_quarter"):
        mature = d.bad_12m.notna()
        rows.append({
            "quarter": quarter, "loans": len(d),
            "psi_score": psi_from_values(base.score, d.score, score_edges),
            "psi_lgbm": psi_from_values(base.pd_lgbm, d.pd_lgbm, lgbm_edges),
            "mean_score": d.score.mean(), "p10": d.score.quantile(.1),
            "p50": d.score.median(), "p90": d.score.quantile(.9),
            "mature_share": mature.mean(),
            "predicted_bad_rate_scorecard": d.loc[mature, "pd"].mean() if mature.any() else np.nan,
            "predicted_bad_rate_lgbm": d.loc[mature, "pd_lgbm"].mean() if mature.any() else np.nan,
            "realised_bad_rate": d.loc[mature, "bad_12m"].mean() if mature.mean() > 0.99 else np.nan,
        })
        for f in chars:
            pts = points[points.feature == f].set_index("bin_no").points
            nb = len(pts)
            e = np.bincount(base[f"bin_{f}"], minlength=nb) / len(base)
            a = np.bincount(d[f"bin_{f}"], minlength=nb) / len(d)
            csi_rows.append({"quarter": quarter, "feature": f, "csi": psi(e, a),
                             "points_shift": float(np.sum((a - e) * pts.reindex(range(nb)).to_numpy()))})
    mon = pd.DataFrame(rows)
    mon["status_score"] = [traffic_light(v, mc["psi_amber"], mc["psi_red"]) for v in mon.psi_score]
    mon["status_lgbm"] = [traffic_light(v, mc["psi_amber"], mc["psi_red"]) for v in mon.psi_lgbm]
    csi = pd.DataFrame(csi_rows)
    csi["status"] = [traffic_light(v, mc["psi_amber"], mc["psi_red"]) for v in csi.csi]
    mon.to_csv(R / "monitoring_quarterly.csv", index=False)
    csi.to_csv(R / "characteristic_stability.csv", index=False)

    # vintage curves (lifetime bad event by months on book)
    abt = pd.read_parquet(P["abt_path"], columns=["id", "issue_d", "bad_event", "mob_bad",
                                                  "mob_observed", "exclusion_reason"])
    abt = abt[abt.exclusion_reason.isna()]
    abt["vintage"] = pd.to_datetime(abt.issue_d).dt.year
    vint = []
    for v, d in abt.groupby("vintage"):
        max_mob = int(d.mob_observed.min())  # every loan in the vintage observed this long
        bad_mob = d.mob_bad.where(d.bad_event == 1)
        for m in range(0, min(max_mob, 60) + 1):
            vint.append({"vintage": v, "mob": m, "loans": len(d),
                         "cum_bad_rate": float((bad_mob <= m).sum() / len(d))})
    vint = pd.DataFrame(vint)
    vint.to_csv(R / "vintage_curves.csv", index=False)

    _plot_stability(mon, mc, F)
    _plot_csi(csi, F)
    _plot_vintages(vint, F)
    _plot_backtest(mon, F)

    print(mon[["quarter", "loans", "psi_score", "status_score", "psi_lgbm", "mean_score",
               "predicted_bad_rate_scorecard", "realised_bad_rate"]].round(3).to_string(index=False))
    worst = csi.sort_values("csi", ascending=False).head(5)
    print(worst.round(3).to_string(index=False))


def _plot_stability(mon, mc, F):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(len(mon))
    ax = axes[0]
    ax.axhspan(mc["psi_amber"], mc["psi_red"], color=STATUS["amber"], alpha=0.12, lw=0)
    ax.axhspan(mc["psi_red"], max(mon[["psi_score", "psi_lgbm"]].max().max() * 1.1, mc["psi_red"] * 1.2),
               color=STATUS["red"], alpha=0.10, lw=0)
    ax.plot(x, mon.psi_score, color=SCORECARD, marker="o", ms=4, label="Scorecard score")
    ax.plot(x, mon.psi_lgbm, color=CHALLENGER, marker="o", ms=4, label="LightGBM PD")
    ax.set(title="Population stability index vs development baseline", ylabel="PSI")
    ax.text(x[-1], mc["psi_amber"], " amber 0.10", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.text(x[-1], mc["psi_red"], " red 0.25", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.fill_between(x, mon.p10, mon.p90, color=SCORECARD, alpha=0.15, lw=0, label="P10–P90")
    ax.plot(x, mon.p50, color=SCORECARD, label="Median")
    ax.plot(x, mon.mean_score, color=SCORECARD, ls="--", lw=1.2, label="Mean")
    ax.set(title="Scorecard score drift by issue quarter", ylabel="Score")
    ax.set_xticks(x[::2], mon.quarter[::2], rotation=45)
    ax.legend(loc="lower left", ncol=3)
    plots.save(fig, F / "stability_and_drift.png")


def _plot_csi(csi, F):
    piv = csi.pivot(index="feature", columns="quarter", values="csi")
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(piv) + 1.8))
    im = ax.imshow(piv.values, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.25, np.nanmax(piv.values)))
    ax.set_yticks(range(len(piv)), [LABELS.get(f, f) for f in piv.index])
    ax.set_xticks(range(0, piv.shape[1], 2), piv.columns[::2], rotation=45)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.18 else "#0b0b0b")
    fig.colorbar(im, ax=ax, label="Characteristic PSI", shrink=0.8)
    ax.set_title("Characteristic stability by issue quarter (values shown where >= 0.10)")
    plots.save(fig, F / "characteristic_stability.png")


def _plot_vintages(vint, F):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    vintages = sorted(vint.vintage.unique())
    vintages = [v for v in vintages if v >= 2012]
    for i, v in enumerate(vintages):
        d = vint[vint.vintage == v]
        ax.plot(d.mob, d.cum_bad_rate, color=SERIES[i % len(SERIES)], label=str(v), lw=1.8)
        ax.text(d.mob.iloc[-1] + 0.5, d.cum_bad_rate.iloc[-1], str(v), fontsize=8,
                color=INK2, va="center")
    ax.axvline(12, color=REFERENCE, lw=1, ls="--")
    ax.text(12.3, ax.get_ylim()[1] * 0.95, "12-month window", fontsize=8, color=INK2, va="top")
    ax.set(title="Vintage curves — cumulative bad rate by months on book",
           xlabel="Months on book", ylabel="Cumulative bad rate (count basis)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(title="Issue year", ncol=2, loc="upper left")
    plots.save(fig, F / "vintage_curves.png")


def _plot_backtest(mon, F):
    d = mon.dropna(subset=["realised_bad_rate"])
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, d.realised_bad_rate, color="#0b0b0b", marker="o", ms=4, label="Realised 12-month bad rate")
    ax.plot(x, d.predicted_bad_rate_scorecard, color=SCORECARD, marker="o", ms=4, label="Scorecard mean PD")
    ax.plot(x, d.predicted_bad_rate_lgbm, color=CHALLENGER, marker="o", ms=4, label="LightGBM mean PD")
    ax.set_xticks(x[::2], d.quarter.iloc[::2], rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set(title="Back-test: predicted vs realised bad rate by issue quarter", ylabel="Bad rate")
    ax.legend(loc="upper left")
    plots.save(fig, F / "backtest_by_quarter.png")


if __name__ == "__main__":
    main()
