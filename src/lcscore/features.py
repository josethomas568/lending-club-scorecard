"""Candidate characteristics. Everything here is known at application time.

Deliberately excluded (see decision log in the MDD):
  grade, sub_grade, int_rate  - Lending Club's own risk pricing; using them would
                                make the scorecard a re-weighting of LC's model
  last_fico_*, total_pymnt, recoveries, last_pymnt_d, out_prncp, ...
                              - observed after origination (target leakage)
  emp_title, title, desc, zip_code, addr_state
                              - free text / geography with fair-lending risk
"""
import pandas as pd

NUMERIC = ["fico", "revol_util", "inq_last_6mths", "delinq_2yrs", "mths_since_last_delinq",
           "open_acc", "total_acc", "pub_rec", "pub_rec_bankruptcies", "mort_acc", "dti",
           "annual_inc", "loan_amnt", "revol_bal", "credit_hist_months", "emp_length_yrs",
           "loan_to_income"]
CATEGORICAL = ["term", "home_ownership", "purpose", "verification_status"]
FEATURES = NUMERIC + CATEGORICAL

LABELS = {
    "fico": "FICO score at application",
    "revol_util": "Revolving utilization",
    "inq_last_6mths": "Credit inquiries in last 6 months",
    "delinq_2yrs": "Delinquencies in last 2 years",
    "mths_since_last_delinq": "Months since last delinquency",
    "open_acc": "Number of open accounts",
    "total_acc": "Total accounts on file",
    "pub_rec": "Public records",
    "pub_rec_bankruptcies": "Public-record bankruptcies",
    "mort_acc": "Mortgage accounts",
    "dti": "Debt-to-income ratio",
    "annual_inc": "Annual income",
    "loan_amnt": "Loan amount requested",
    "revol_bal": "Revolving balance",
    "credit_hist_months": "Length of credit history",
    "emp_length_yrs": "Length of employment",
    "loan_to_income": "Loan amount relative to income",
    "term": "Loan term",
    "home_ownership": "Home ownership",
    "purpose": "Loan purpose",
    "verification_status": "Income verification status",
}

MODEL_SAMPLES = ["train", "test_in_time", "oot"]
SCORED_SAMPLES = MODEL_SAMPLES + ["monitor_only"]


def load_abt(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["sample"].isin(SCORED_SAMPLES)].copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("object")
    return df.reset_index(drop=True)
