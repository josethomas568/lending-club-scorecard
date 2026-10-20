"""Step 3b — challenger: LightGBM on the raw (un-binned) candidate characteristics.

* Same population, target and samples as the scorecard.
* Early stopping uses a 10% slice of the TRAIN sample (id mod 10 == 0), so the
  in-time test and the out-of-time sample stay untouched for evaluation.
* Monotone constraints on characteristics whose risk direction is not in
  doubt, so reason codes stay defensible (e.g. a higher FICO never raises PD).
* Categorical levels are frozen from training and written to a manifest the
  production scoring script reads; unseen levels score as missing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd

from lcscore.config import load_config, save_json
from lcscore.features import CATEGORICAL, FEATURES, NUMERIC, load_abt

MONOTONE = {"fico": -1, "revol_util": 1, "inq_last_6mths": 1, "delinq_2yrs": 1,
            "pub_rec": 1, "pub_rec_bankruptcies": 1, "dti": 1, "loan_to_income": 1}


def to_matrix(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in NUMERIC:
        X[c] = X[c].astype(float)
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=levels[c])
    return X


def main():
    cfg = load_config()
    ch = cfg["challenger"]
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    df = load_abt(P["abt_path"])
    train_all = df[df["sample"] == "train"]
    stop_mask = (train_all.id % 10 == 0).to_numpy()
    levels = {c: sorted(train_all[c].dropna().astype(str).unique()) for c in CATEGORICAL}

    Xtr, ytr = to_matrix(train_all[~stop_mask], levels), train_all.bad_12m[~stop_mask]
    Xes, yes = to_matrix(train_all[stop_mask], levels), train_all.bad_12m[stop_mask]
    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "learning_rate": ch["learning_rate"], "num_leaves": ch["num_leaves"],
              "min_child_samples": ch["min_child_samples"],
              "feature_fraction": ch["feature_fraction"],
              "bagging_fraction": ch["bagging_fraction"], "bagging_freq": ch["bagging_freq"],
              "lambda_l2": ch["lambda_l2"], "seed": ch["seed"], "deterministic": True,
              "num_threads": 4,
              "monotone_constraints": [MONOTONE.get(f, 0) for f in FEATURES],
              "monotone_constraints_method": "advanced"}
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=CATEGORICAL, free_raw_data=False)
    des = lgb.Dataset(Xes, yes, reference=dtr)
    booster = lgb.train(params, dtr, ch["num_boost_round"], valid_sets=[des],
                        callbacks=[lgb.early_stopping(ch["early_stopping_rounds"], verbose=False)])
    booster.save_model(str(P["model_dir"] / "challenger_lgbm.txt"),
                       num_iteration=booster.best_iteration)

    manifest = {"features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
                "categorical_levels": levels, "monotone_constraints": MONOTONE,
                "best_iteration": booster.best_iteration,
                "early_stopping_auc": booster.best_score["valid_0"]["auc"], "params": params}
    save_json(manifest, P["model_dir"] / "challenger_manifest.json")

    out = df[["id", "sample", "issue_d", "issue_quarter", "bad_12m"]].copy()
    out["pd"] = booster.predict(to_matrix(df, levels), num_iteration=booster.best_iteration)
    out.to_parquet(P["model_dir"] / "scores_challenger.parquet", index=False)
    print(f"best iteration {booster.best_iteration}, early-stopping AUC "
          f"{manifest['early_stopping_auc']:.4f}")


if __name__ == "__main__":
    main()
