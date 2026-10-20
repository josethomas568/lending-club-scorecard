"""Write a schema-faithful synthetic stand-in for the Kaggle Lending Club file.

It exists only to smoke-test the pipeline where the real file is unavailable.
It reproduces the real file's quirks that the pipeline must handle — string
dates like "Dec-2015", " 36 months" with a leading space, "10+ years",
loan-status labels, extra columns we ignore, and non-loan footer rows — plus a
known default process with vintage drift so the models and monitors have
something to find. Numbers produced from it are NOT Lending Club results.

    python scripts/make_synthetic_sample.py --scale 0.1
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Approximate accepted-loan counts per issue year in the real file.
YEAR_COUNTS = {2007: 603, 2008: 2393, 2009: 5281, 2010: 12537, 2011: 21721,
               2012: 53367, 2013: 134814, 2014: 235629, 2015: 421095,
               2016: 434407, 2017: 443579, 2018: 495242}
SNAPSHOT = pd.Timestamp("2019-03-01")
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other",
            "major_purchase", "medical", "small_business", "car", "moving",
            "vacation", "house", "wedding", "renewable_energy", "educational"]
PURPOSE_P = np.array([.58, .22, .065, .06, .022, .012, .011, .01, .007, .006,
                      .004, .002, .0007, .0003])
PURPOSE_RISK = {"small_business": .55, "medical": .2, "moving": .2, "other": .15,
                "house": .1, "educational": .3, "renewable_energy": .1,
                "vacation": .05, "debt_consolidation": .05, "major_purchase": -.05,
                "home_improvement": -.05, "wedding": -.1, "car": -.15,
                "credit_card": -.12}


def month_str(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%b-%Y")


def add_months(base: pd.Series, months) -> pd.Series:
    periods = base.dt.to_period("M") + np.asarray(months, dtype=int)
    return periods.dt.to_timestamp()


def generate(scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for year, count in YEAR_COUNTS.items():
        n = max(int(count * scale), 30)
        month = rng.integers(1, 13, n)
        frames.append(pd.DataFrame({"issue": pd.to_datetime(
            {"year": np.full(n, year), "month": month, "day": 1})}))
    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    yr = df.issue.dt.year.to_numpy()
    q = (df.issue.dt.month.to_numpy() - 1) // 3

    # --- applicant attributes, with gradual population drift after 2015 ----
    drift = np.clip((yr - 2015) + q / 4, 0, None) * (yr <= 2017) \
        + np.where(yr == 2018, 1.0, 0)          # 2016-17 loosening, 2018 partial tightening
    fico_low = np.clip(np.round((rng.normal(696 - 6 * drift, 30, n)) / 5) * 5, 660, 845)
    inq = np.clip(rng.poisson(0.65 + 0.12 * drift), 0, 8)
    delinq = np.clip(rng.poisson(0.3, n) * (rng.random(n) < 0.18), 0, 15)
    open_acc = np.clip(rng.poisson(11.5, n), 1, 60)
    total_acc = open_acc + rng.poisson(13, n)
    revol_util = np.clip(rng.beta(2.2, 2.0, n) * 100 + 4 * drift, 0, 120).round(1)
    revol_util = np.where(rng.random(n) < 0.0007, np.nan, revol_util)
    annual_inc = np.round(np.exp(rng.normal(11.1, 0.5, n)), -2)
    dti = np.clip(rng.gamma(4.0, 4.4 + 0.6 * drift, n), 0, 60).round(2)
    pub_rec = (rng.random(n) < 0.17).astype(int) * rng.integers(1, 3, n)
    pub_rec_bk = np.minimum(pub_rec, (rng.random(n) < 0.8).astype(int))
    mort_acc = np.clip(rng.poisson(1.6, n), 0, 20).astype(float)
    mort_acc[(yr < 2013) & (rng.random(n) < 0.6)] = np.nan   # field sparsely populated early
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "ANY", "OTHER", "NONE"], n,
                      p=[.49, .395, .11, .002, .002, .001])
    home = np.where(mort_acc > 0, np.where(rng.random(n) < .75, "MORTGAGE", home), home)
    verif = rng.choice(["Not Verified", "Source Verified", "Verified"], n, p=[.31, .39, .30])
    term_m = np.where(rng.random(n) < 0.24 + 0.03 * drift, 60, 36)
    loan_amnt = np.clip(np.round(rng.gamma(2.2, 6800, n) / 25) * 25, 1000, 40000)
    purpose = rng.choice(PURPOSES, n, p=PURPOSE_P / PURPOSE_P.sum())
    emp_years = rng.choice(np.arange(-1, 11), n,
                           p=[.065, .085, .065, .09, .08, .06, .06, .05, .05, .045, .04, .31])
    emp_length = np.select(
        [emp_years == -1, emp_years == 0, emp_years == 1, emp_years == 10],
        [None, "< 1 year", "1 year", "10+ years"],
        default=pd.Series(emp_years).astype(str).add(" years").to_numpy())
    hist_months = np.clip(rng.gamma(5.0, 38, n).astype(int), 36, 720)
    earliest = add_months(df.issue, -hist_months)
    last_delinq = np.where(delinq > 0, rng.integers(1, 24, n),
                           np.where(rng.random(n) < 0.33, rng.integers(24, 120, n), np.nan))

    # --- lifetime default process --------------------------------------------
    purpose_risk = np.vectorize(PURPOSE_RISK.get)(purpose)
    lti = loan_amnt / annual_inc
    signal = (- 0.022 * (fico_low - 700)
             + 0.17 * inq
             + 0.012 * (np.nan_to_num(revol_util, nan=50.0) - 50)
             + 0.024 * (dti - 18)
             + 0.55 * (term_m == 60)
             + 0.9 * np.clip(lti - 0.25, 0, None)
             + 0.10 * np.minimum(delinq, 3)
             + 0.12 * pub_rec
             - 0.08 * np.nan_to_num(mort_acc, nan=1.0)
             + 0.30 * (home == "RENT")
             - 0.06 * np.clip(emp_years, 0, 10) / 3
             - 0.0012 * (hist_months - 180)
             + 0.10 * (verif == "Verified")
             + 1.6 * purpose_risk)
    logit = (-2.45 + 2.0 * signal
             + 0.30 * drift                       # unexplained vintage deterioration
             + rng.normal(0, 0.3, n))             # idiosyncratic noise
    defaults = rng.random(n) < 1 / (1 + np.exp(-logit))
    # month on book at which 90+ DPD is reached
    mob_bad = np.clip(np.round(rng.gamma(2.6, 6.0 + 2.5 * (term_m == 60), n)) + 3, 4, term_m - 1)
    prepay = np.where(rng.random(n) < 0.42, rng.integers(3, term_m), 10_000)

    df_out = pd.DataFrame(index=range(n))
    observable = ((SNAPSHOT.to_period("M") - df.issue.dt.to_period("M"))
                  .apply(lambda x: x.n).to_numpy())
    status = np.full(n, "Current", dtype=object)
    last_pay_mob = np.minimum(observable, term_m).astype(float)
    bad_before_prepay = defaults & (mob_bad < prepay)
    bad_seen = bad_before_prepay & (mob_bad <= observable)
    co = bad_seen & (mob_bad + 1 <= observable)
    status[bad_seen] = "Late (31-120 days)"
    status[co] = np.where(rng.random(co.sum()) < 0.995, "Charged Off", "Default")
    last_pay_mob[bad_seen] = mob_bad[bad_seen] - 4
    upcoming = bad_before_prepay & ~bad_seen & (mob_bad - observable <= 2)
    status[upcoming] = rng.choice(["In Grace Period", "Late (16-30 days)"], upcoming.sum())
    last_pay_mob[upcoming] = observable[upcoming] - 1
    paid = ~bad_seen & ~upcoming & ((prepay <= observable) | (term_m <= observable))
    status[paid] = "Fully Paid"
    last_pay_mob[paid] = np.minimum(prepay, term_m)[paid]
    last_pay = add_months(df.issue, np.maximum(last_pay_mob, 0).astype(int))
    last_pay = last_pay.where(last_pay_mob >= 1)
    old_policy = (yr <= 2010) & (rng.random(n) < 0.06)
    status = np.where(old_policy & np.isin(status, ["Fully Paid", "Charged Off"]),
                      "Does not meet the credit policy. Status:" + status, status)

    grade_score = -logit + rng.normal(0, 0.4, n)
    grade = pd.qcut(grade_score, 7, labels=list("GFEDCBA")).astype(str)
    df_out = pd.DataFrame({
        "id": (60_000_000 + rng.permutation(n)).astype(str),
        "member_id": "",
        "loan_amnt": loan_amnt, "funded_amnt": loan_amnt, "funded_amnt_inv": loan_amnt,
        "term": np.where(term_m == 60, " 60 months", " 36 months"),
        "int_rate": np.round(6 + (ord("A") - np.array([ord(g) for g in grade])) * -3.4
                             + rng.normal(0, 1, n), 2),
        "installment": np.round(loan_amnt / term_m * 1.15, 2),
        "grade": grade, "sub_grade": grade + rng.integers(1, 6, n).astype(str),
        "emp_title": "", "emp_length": emp_length, "home_ownership": home,
        "annual_inc": annual_inc, "verification_status": verif,
        "issue_d": month_str(df.issue), "loan_status": status, "pymnt_plan": "n",
        "url": "https://lendingclub.com/browse/loanDetail.action?loan_id=",
        "purpose": purpose, "title": "", "zip_code": "100xx", "addr_state": "CA",
        "dti": dti, "delinq_2yrs": delinq.astype(float),
        "earliest_cr_line": month_str(earliest),
        "fico_range_low": fico_low, "fico_range_high": fico_low + 4,
        "inq_last_6mths": inq.astype(float), "mths_since_last_delinq": last_delinq,
        "open_acc": open_acc.astype(float), "pub_rec": pub_rec.astype(float),
        "revol_bal": np.round(revol_util / 100 * rng.gamma(2, 9000, n)),
        "revol_util": revol_util, "total_acc": total_acc.astype(float),
        "last_pymnt_d": month_str(last_pay), "policy_code": 1.0,
        "application_type": np.where((yr >= 2016) & (rng.random(n) < 0.05),
                                     "Joint App", "Individual"),
        "mort_acc": mort_acc, "pub_rec_bankruptcies": pub_rec_bk.astype(float),
        "last_fico_range_high": np.clip(fico_low + rng.normal(0, 40, n), 300, 850).round(),
        "recoveries": 0.0, "total_pymnt": 0.0,
    })
    footer = pd.DataFrame({"id": ["Total amount funded in policy code 1: 0",
                                  "Total amount funded in policy code 2: 0"]})
    return pd.concat([df_out, footer], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(REPO / "data/raw/accepted_2007_to_2018Q4.csv.gz"))
    args = ap.parse_args()
    df = generate(args.scale, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", newline="") as fh:
        df.to_csv(fh, index=False)
    # marker records the synthetic file's path and size; config.py ignores it once
    # the file at that path is replaced by the real download
    (REPO / "data/raw/SYNTHETIC").write_text(f"{out.resolve()}\n{out.stat().st_size}\n")
    print(f"wrote {len(df):,} rows (incl. 2 footer rows) to {out}")


if __name__ == "__main__":
    main()
