"""Step 3a — benchmark: WoE/IV logistic scorecard.

1. Fine/coarse classing on the training sample (monotonic WoE, min 5% per bin).
2. Screen on information value: drop IV < 0.02; flag IV > 0.50 as suspected leakage.
3. Drop the weaker of any pair of WoE characteristics correlated above 0.70.
4. Logistic regression on WoE; remove, one at a time, characteristics with a
   wrong-signed coefficient or p-value above the threshold.
5. Scale to points: base 600 at 50:1 good:bad odds, 20 points to double the odds.
   Points per attribute are rounded to integers; the score is their sum.

Outputs: model/scorecard_points.csv (the implementation spec), model/scorecard_spec.json,
results/iv_table.csv, results/selection_log.csv, model/scores_scorecard.parquet
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lcscore.binning import CategoricalBinner, NumericBinner
from lcscore.config import load_config, save_json
from lcscore.features import CATEGORICAL, FEATURES, LABELS, NUMERIC, load_abt


def main():
    cfg = load_config()
    sc = cfg["scorecard"]
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    df = load_abt(P["abt_path"])
    train = df[df["sample"] == "train"]
    y = train.bad_12m.astype(float)
    log = []

    # ---- 1-2. binning and IV screen ------------------------------------------
    binners, iv_rows = {}, []
    for f in FEATURES:
        if f in NUMERIC:
            b = NumericBinner(f, sc["fine_bins"], sc["max_bins"], sc["min_bin_share"]).fit(train[f], y)
        else:
            b = CategoricalBinner(f, sc["max_bins"], sc["min_bin_share"],
                                  sc["rare_category_share"]).fit(train[f], y)
        binners[f] = b
        iv = b.table.iv.sum()
        iv_rows.append({"feature": f, "label": LABELS[f], "iv": iv, "bins": len(b.table)})
    iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False)
    iv_table.to_csv(P["results_dir"] / "iv_table.csv", index=False)
    pd.concat([b.table for b in binners.values()]).to_csv(P["results_dir"] / "binning_detail.csv", index=False)

    selected = []
    for r in iv_table.itertuples():
        if r.iv < sc["min_iv"]:
            log.append((r.feature, "dropped", f"IV {r.iv:.4f} < {sc['min_iv']}"))
        else:
            if r.iv > sc["max_iv"]:
                # Flag, don't drop: a very high IV means "check this is truly
                # known at application", which is a human review, not a rule.
                log.append((r.feature, "flagged", f"IV {r.iv:.3f} > {sc['max_iv']}: "
                            "confirm it is an application-time attribute"))
            selected.append(r.feature)

    W = pd.DataFrame({f: binners[f].transform(train[f]) for f in selected}, index=train.index)

    # ---- 3. correlation screen (keep the higher-IV member) -------------------
    ivs = iv_table.set_index("feature").iv
    corr = W.corr().abs()
    for f in sorted(selected, key=lambda f: -ivs[f]):
        if f not in selected:
            continue
        for g in list(selected):
            if g != f and ivs[g] <= ivs[f] and corr.loc[f, g] > sc["max_corr"]:
                selected.remove(g)
                log.append((g, "dropped", f"|corr| {corr.loc[f, g]:.2f} with {f} (higher IV)"))

    # ---- 4. logistic regression with sign / significance pruning -------------
    while True:
        X = sm.add_constant(W[selected])
        fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        coefs, pvals = fit.params.drop("const"), fit.pvalues.drop("const")
        wrong = coefs[coefs >= 0]
        weak = pvals[pvals > sc["max_pvalue"]]
        if len(wrong):
            f = wrong.idxmax()
            reason = f"coefficient {coefs[f]:+.3f} has wrong sign (expected < 0 on WoE)"
        elif len(weak):
            f = weak.idxmax()
            reason = f"p-value {pvals[f]:.3g} > {sc['max_pvalue']}"
        else:
            break
        selected.remove(f)
        log.append((f, "dropped", reason))
    for f in selected:
        log.append((f, "kept", f"IV {ivs[f]:.3f}, coef {coefs[f]:+.3f}, p {pvals[f]:.2g}"))
    sel_log = pd.DataFrame(log, columns=["feature", "decision", "reason"])
    sel_log.to_csv(P["results_dir"] / "selection_log.csv", index=False)

    # ---- 5. scaling to integer points ---------------------------------------
    factor = sc["pdo"] / np.log(2)
    offset = sc["base_score"] - factor * np.log(sc["base_odds"])
    b0, k = fit.params["const"], len(selected)
    rows = []
    for f in selected:
        t = binners[f].table.copy()
        t["bin_no"] = range(len(t))
        t["coef"] = coefs[f]
        t["points"] = np.round(-factor * coefs[f] * t.woe + (offset - factor * b0) / k).astype(int)
        # category bin that unseen levels fall into at scoring time
        t["is_else"] = (t.type == "category") & (t.bin_no == getattr(binners[f], "else_group", -1))
        rows.append(t)
    points = pd.concat(rows, ignore_index=True)
    points = points[["feature", "bin_no", "type", "lo", "hi", "category", "is_else", "count",
                     "bads", "woe", "iv", "coef", "points"]]
    points.to_csv(P["model_dir"] / "scorecard_points.csv", index=False)

    spec = {"features": selected, "intercept": b0, "coefficients": coefs.to_dict(),
            "factor": factor, "offset": offset, "pdo": sc["pdo"],
            "base_score": sc["base_score"], "base_odds": sc["base_odds"],
            "else_category_bin": {f: int(binners[f].else_group) for f in selected
                                  if f in CATEGORICAL},
            "model_summary": fit.summary2().tables[1].reset_index().to_dict("records"),
            "train_n": int(len(train)), "pseudo_r2": float(fit.prsquared)}
    save_json(spec, P["model_dir"] / "scorecard_spec.json")
    with open(P["model_dir"] / "scorecard_binners.pkl", "wb") as fh:
        pickle.dump({f: binners[f] for f in selected}, fh)

    # ---- development scores for every scored loan --------------------------
    out = df[["id", "sample", "issue_d", "issue_quarter", "bad_12m"]].copy()
    total = np.zeros(len(df), dtype=int)
    for f in selected:
        pts = points[points.feature == f].points.to_numpy()
        out[f"pts_{f}"] = pts[binners[f].bin_index(df[f])]
        out[f"bin_{f}"] = binners[f].bin_index(df[f])
        total += out[f"pts_{f}"].to_numpy()
    out["score"] = total
    out["pd"] = 1 / (1 + np.exp((out.score - offset) / factor))
    out.to_parquet(P["model_dir"] / "scores_scorecard.parquet", index=False)

    print(iv_table.to_string(index=False))
    print(sel_log.to_string(index=False))
    print(f"{k} characteristics, score range "
          f"{points.groupby('feature').points.min().sum()}–{points.groupby('feature').points.max().sum()}")


if __name__ == "__main__":
    main()
