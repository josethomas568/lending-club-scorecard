"""Production-style scoring for the LightGBM challenger.

Deliberately standalone: it imports nothing from the development package and
rebuilds every feature from the mart tables itself. Its only model inputs are
the two hand-off artifacts: challenger_lgbm.txt and challenger_manifest.json.

    python production/score_challenger.py --mart data/mart \
        --model outputs/model/challenger_lgbm.txt \
        --manifest outputs/model/challenger_manifest.json \
        --min-issue-date 2012-01-01 --out outputs/model/prod_scores_challenger.parquet
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd


def build_features(mart: str, min_issue_date: str, application_type: str, policy_code: float):
    loans = pd.read_parquet(f"{mart}/loans")
    bureau = pd.read_parquet(f"{mart}/bureau")
    d = loans.merge(bureau, on="id", how="inner", validate="one_to_one")
    d["issue_d"] = pd.to_datetime(d["issue_d"])
    d["earliest_cr_line"] = pd.to_datetime(d["earliest_cr_line"])
    d = d[(d.issue_d >= pd.Timestamp(min_issue_date))
          & (d.application_type == application_type)
          & (d.policy_code == policy_code)
          & d.fico_range_low.notna() & d.fico_range_high.notna() & d.annual_inc.notna()]

    f = pd.DataFrame({"id": d.id.to_numpy()})
    f["fico"] = ((d.fico_range_low + d.fico_range_high) / 2).to_numpy()
    for c in ["revol_util", "inq_last_6mths", "delinq_2yrs", "mths_since_last_delinq",
              "open_acc", "total_acc", "pub_rec", "pub_rec_bankruptcies", "mort_acc", "dti",
              "annual_inc", "loan_amnt", "revol_bal"]:
        f[c] = d[c].astype(float).to_numpy()
    months = ((d.issue_d.dt.year - d.earliest_cr_line.dt.year) * 12
              + (d.issue_d.dt.month - d.earliest_cr_line.dt.month))
    f["credit_hist_months"] = months.astype(float).to_numpy()
    emp = d.emp_length.astype("object")
    yrs = pd.to_numeric(emp.str.extract(r"^(\d+)", expand=False), errors="coerce")
    yrs = yrs.mask(emp == "< 1 year", 0.0).mask(emp == "10+ years", 10.0)
    f["emp_length_yrs"] = yrs.astype(float).to_numpy()
    inc = d.annual_inc.where(d.annual_inc > 0)
    f["loan_to_income"] = (d.loan_amnt / inc).to_numpy()
    f["term"] = d.term_months.astype("Int64").astype(str).to_numpy()
    f["home_ownership"] = d.home_ownership.where(
        d.home_ownership.isin(["MORTGAGE", "RENT", "OWN"]), "OTHER").to_numpy()
    f["purpose"] = d.purpose.to_numpy()
    f["verification_status"] = d.verification_status.to_numpy()
    return f


def score(features: pd.DataFrame, model_path: str, manifest_path: str) -> pd.DataFrame:
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    X = features[manifest["features"]].copy()
    for c in manifest["categorical"]:
        # levels frozen at training; anything unseen becomes NaN (LightGBM's missing branch)
        X[c] = pd.Categorical(X[c], categories=manifest["categorical_levels"][c])
    booster = lgb.Booster(model_file=model_path)
    if booster.feature_name() != manifest["features"]:
        raise ValueError("feature order in model file does not match the manifest")
    return pd.DataFrame({"id": features.id, "pd": booster.predict(X)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mart", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min-issue-date", default="2012-01-01")
    ap.add_argument("--application-type", default="Individual")
    ap.add_argument("--policy-code", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    feats = build_features(a.mart, a.min_issue_date, a.application_type, a.policy_code)
    out = score(feats, a.model, a.manifest)
    out.to_parquet(a.out, index=False)
    print(f"scored {len(out):,} applications -> {a.out}  (mean PD {np.mean(out.pd):.4f})")


if __name__ == "__main__":
    main()
