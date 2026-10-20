"""Step 1 — load the raw Kaggle file in PySpark, type it, split it into three
mart tables (loans, bureau, outcomes), join them back, and reconcile row counts
at every hop. An independent DuckDB count of the raw file is the first check.

Outputs
  data/mart/{loans,bureau,outcomes}/   typed parquet
  outputs/results/control_totals.json  totals the BigQuery mirror must match
  outputs/results/ingest_reconciliation.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import pandas as pd
from pyspark.sql import functions as F

from lcscore.config import load_config, save_json
from lcscore.spark import spark_session

LOAN_COLS = ["id", "issue_d", "term_months", "loan_amnt", "purpose", "application_type",
             "policy_code", "int_rate", "grade", "sub_grade", "verification_status",
             "home_ownership", "emp_length", "annual_inc", "dti"]
BUREAU_COLS = ["id", "fico_range_low", "fico_range_high", "earliest_cr_line",
               "inq_last_6mths", "delinq_2yrs", "mths_since_last_delinq", "open_acc",
               "total_acc", "pub_rec", "pub_rec_bankruptcies", "revol_bal", "revol_util",
               "mort_acc"]
OUTCOME_COLS = ["id", "loan_status", "last_pymnt_d"]
DOUBLE_COLS = ["loan_amnt", "policy_code", "annual_inc", "dti", "fico_range_low",
               "fico_range_high", "inq_last_6mths", "delinq_2yrs", "mths_since_last_delinq",
               "open_acc", "total_acc", "pub_rec", "pub_rec_bankruptcies", "revol_bal",
               "mort_acc"]


def month_date(col):
    # "Dec-2015" -> 2015-12-01
    return F.to_date(F.concat(F.lit("01-"), F.trim(col)), "dd-MMM-yyyy")


def pct(col):
    return F.regexp_replace(F.trim(col), "%", "").cast("double")


def main():
    cfg = load_config()
    raw = cfg["paths"]["raw_csv"]
    mart = Path(cfg["paths"]["mart_dir"])
    recon = []

    # Independent engine count of the raw file (handles quoted newlines).
    duck_rows = duckdb.sql(
        f"select count(*) from read_csv('{raw}', all_varchar=true, header=true)").fetchone()[0]
    recon.append(("raw rows (DuckDB, independent)", duck_rows))

    spark = spark_session(cfg, "lc-ingest")
    df = (spark.read.option("header", True).option("multiLine", True)
          .option("escape", '"').option("inferSchema", False).csv(raw))
    raw_rows = df.count()
    recon.append(("raw rows (Spark)", raw_rows))

    is_loan = F.col("id").rlike(r"^\d+$") & F.col("loan_amnt").isNotNull()
    df = df.filter(is_loan)
    loan_rows = df.count()
    recon.append(("non-loan rows dropped (footer/summary lines)", raw_rows - loan_rows))

    dupes = loan_rows - df.select("id").distinct().count()
    recon.append(("duplicate loan ids", dupes))
    df = df.dropDuplicates(["id"])

    typed = df.select(
        F.col("id").cast("long").alias("id"),
        month_date(F.col("issue_d")).alias("issue_d"),
        F.regexp_extract("term", r"(\d+)", 1).cast("int").alias("term_months"),
        pct(F.col("int_rate")).alias("int_rate"),
        pct(F.col("revol_util")).alias("revol_util"),
        month_date(F.col("earliest_cr_line")).alias("earliest_cr_line"),
        month_date(F.col("last_pymnt_d")).alias("last_pymnt_d"),
        *[F.col(c).cast("double").alias(c) for c in DOUBLE_COLS],
        *[F.trim(F.col(c)).alias(c) for c in
          ["purpose", "application_type", "grade", "sub_grade", "verification_status",
           "home_ownership", "emp_length", "loan_status"]],
    ).cache()
    clean_rows = typed.count()
    recon.append(("clean loans", clean_rows))
    bad_dates = typed.filter(F.col("issue_d").isNull()).count()
    recon.append(("loans with unparseable issue_d", bad_dates))

    tables = {"loans": LOAN_COLS, "bureau": BUREAU_COLS, "outcomes": OUTCOME_COLS}
    controls = {}
    for name, cols in tables.items():
        t = typed.select(*cols)
        t.write.mode("overwrite").parquet(str(mart / name))
        back = spark.read.parquet(str(mart / name))
        n = back.count()
        recon.append((f"{name} table rows (re-read from parquet)", n))
        aggs = [F.count(F.lit(1)).alias("row_count"),
                F.countDistinct("id").alias("distinct_ids"),
                F.sum(F.col("id") % 1000003).alias("id_checksum")]
        for c in cols:
            if c == "id":
                continue
            dtype = dict(back.dtypes)[c]
            aggs.append(F.sum(F.col(c).isNull().cast("int")).alias(f"nulls__{c}"))
            if dtype in ("double", "int", "bigint"):
                aggs.append(F.round(F.sum(c), 2).alias(f"sum__{c}"))
            if dtype == "date":
                aggs += [F.min(c).cast("string").alias(f"min__{c}"),
                         F.max(c).cast("string").alias(f"max__{c}")]
        controls[name] = back.agg(*aggs).first().asDict()

    loans = spark.read.parquet(str(mart / "loans"))
    bureau = spark.read.parquet(str(mart / "bureau"))
    outcomes = spark.read.parquet(str(mart / "outcomes"))
    joined = loans.join(bureau, "id", "inner").join(outcomes, "id", "inner")
    recon.append(("joined loans ⋈ bureau ⋈ outcomes", joined.count()))
    recon.append(("loans without bureau row", loans.join(bureau, "id", "left_anti").count()))
    recon.append(("loans without outcome row", loans.join(outcomes, "id", "left_anti").count()))
    controls["status_counts"] = {r["loan_status"]: r["count"] for r in
                                 outcomes.groupBy("loan_status").count().collect()}
    spark.stop()

    rec = pd.DataFrame(recon, columns=["check", "rows"])
    expected = {"raw rows (Spark)": duck_rows,
                "clean loans": loan_rows - dupes,
                "joined loans ⋈ bureau ⋈ outcomes": loan_rows - dupes,
                "loans without bureau row": 0, "loans without outcome row": 0,
                "loans with unparseable issue_d": 0}
    for name in tables:
        expected[f"{name} table rows (re-read from parquet)"] = loan_rows - dupes
    rec["expected"] = rec["check"].map(expected)
    rec["status"] = ["PASS" if pd.isna(e) or r == e else "FAIL"
                     for r, e in zip(rec["rows"], rec["expected"])]
    out = Path(cfg["paths"]["results_dir"])
    rec.to_csv(out / "ingest_reconciliation.csv", index=False)
    save_json(controls, out / "control_totals.json")
    print(rec.to_string(index=False))
    if (rec["status"] == "FAIL").any():
        sys.exit("ingest reconciliation FAILED")


if __name__ == "__main__":
    main()
