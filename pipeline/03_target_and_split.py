"""Step 2 — observation/performance windows, bad flag, model features and the
out-of-time split, built in PySpark from the three mart tables.

Observation point : issue date (application scorecard; attributes are as of application).
Performance window: 12 months from issue.
Bad               : charged off / default, or 90+ days past due, with the 90+ DPD
                    date (estimated as last_pymnt_d + 4 months; issue_d + 4 if no
                    payment was ever made) falling within 12 months of issue.
Good              : every other loan with a complete 12-month window, including
                    loans repaid early and loans that went bad after month 12.
Immature          : issue_d + 12 months > snapshot -> no label; used for monitoring only.

Samples
  train / test_in_time : dev window, split 75/25 on a hash of the loan id
  oot                  : out-of-time, mature loans issued after the dev window
  monitor_only         : immature loans (scored for stability monitoring, no label)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyspark.sql import functions as F

from lcscore.config import load_config, save_json
from lcscore.spark import spark_session

BAD_STATUSES = ["Charged Off", "Default",
                "Does not meet the credit policy. Status:Charged Off"]


def main():
    cfg = load_config()
    tcfg, pcfg, scfg = cfg["target"], cfg["population"], cfg["split"]
    mart = Path(cfg["paths"]["mart_dir"])
    spark = spark_session(cfg, "lc-target")

    loans = spark.read.parquet(str(mart / "loans"))
    bureau = spark.read.parquet(str(mart / "bureau"))
    outcomes = spark.read.parquet(str(mart / "outcomes"))
    df = loans.join(bureau, "id").join(outcomes, "id")

    snapshot = (df.agg(F.max("last_pymnt_d")).first()[0] if tcfg["snapshot_date"] == "auto"
                else F.to_date(F.lit(tcfg["snapshot_date"])))
    snapshot_lit = F.lit(str(snapshot)).cast("date")
    lag, perf = tcfg["dpd90_lag_months"], tcfg["performance_months"]

    est_bad = F.add_months(F.coalesce("last_pymnt_d", "issue_d"), lag)
    bad_event = (F.col("loan_status").isin(BAD_STATUSES)
                 | ((F.col("loan_status") == "Late (31-120 days)") & (est_bad <= snapshot_lit)))
    df = (df
          .withColumn("snapshot_date", snapshot_lit)
          .withColumn("bad_event", bad_event.cast("int"))
          .withColumn("bad_date", F.when(bad_event, est_bad))
          .withColumn("mob_bad", F.when(bad_event, F.months_between("bad_date", "issue_d").cast("int")))
          .withColumn("mob_observed", F.months_between(snapshot_lit, "issue_d").cast("int"))
          .withColumn("mature", (F.add_months("issue_d", perf) <= snapshot_lit).cast("int"))
          .withColumn("bad_12m", F.when(F.col("mature") == 1,
                      F.coalesce((F.col("bad_event") == 1) & (F.col("mob_bad") <= perf),
                                 F.lit(False)).cast("int"))))

    # ---- model features (development implementation) -----------------------
    emp = F.col("emp_length")
    df = (df
          .withColumn("fico", (F.col("fico_range_low") + F.col("fico_range_high")) / 2)
          .withColumn("credit_hist_months", F.months_between("issue_d", "earliest_cr_line").cast("int"))
          .withColumn("emp_length_yrs",
                      F.when(emp == "< 1 year", 0)
                       .when(emp == "10+ years", 10)
                       .otherwise(F.regexp_extract(emp, r"^(\d+)", 1).cast("int")).cast("double"))
          .withColumn("loan_to_income", F.when(F.col("annual_inc") > 0,
                                               F.col("loan_amnt") / F.col("annual_inc")))
          .withColumn("term", F.col("term_months").cast("string"))
          .withColumn("home_ownership", F.when(F.col("home_ownership").isin("MORTGAGE", "RENT", "OWN"),
                                               F.col("home_ownership")).otherwise("OTHER"))
          .withColumn("issue_quarter", F.concat(F.year("issue_d").cast("string"), F.lit("Q"),
                                                F.quarter("issue_d").cast("string"))))

    # ---- population and split ---------------------------------------------
    excl = (F.when(F.col("issue_d") < F.lit(pcfg["min_issue_date"]).cast("date"), "issued before min_issue_date")
             .when(F.col("application_type") != pcfg["application_type"], "joint application")
             .when(F.col("policy_code") != pcfg["policy_code"], "policy_code")
             .when(F.col("fico").isNull() | F.col("annual_inc").isNull(), "missing FICO or income"))
    df = df.withColumn("exclusion_reason", excl)

    oot_end = (F.add_months(snapshot_lit, -perf) if scfg["oot_end"] == "auto"
               else F.lit(scfg["oot_end"]).cast("date"))
    in_dev = F.col("issue_d").between(F.lit(scfg["dev_start"]).cast("date"),
                                      F.lit(scfg["dev_end"]).cast("date"))
    in_train = F.pmod(F.xxhash64("id"), F.lit(10000)) < int(scfg["train_share"] * 10000)
    df = df.withColumn("sample",
        F.when(F.col("exclusion_reason").isNotNull(), "excluded")
         .when(F.col("mature") == 0, "monitor_only")
         .when(in_dev & in_train, "train")
         .when(in_dev, "test_in_time")
         .when((F.col("issue_d") >= F.lit(scfg["oot_start"]).cast("date"))
               & (F.col("issue_d") <= oot_end), "oot")
         .otherwise("monitor_only"))

    keep = ["id", "issue_d", "issue_quarter", "snapshot_date", "loan_status", "last_pymnt_d",
            "bad_event", "bad_date", "mob_bad", "mob_observed", "mature", "bad_12m",
            "exclusion_reason", "sample", "grade", "int_rate",
            "fico", "revol_util", "inq_last_6mths", "delinq_2yrs", "mths_since_last_delinq",
            "open_acc", "total_acc", "pub_rec", "pub_rec_bankruptcies", "mort_acc", "dti",
            "annual_inc", "loan_amnt", "revol_bal", "credit_hist_months", "emp_length_yrs",
            "loan_to_income", "term", "home_ownership", "purpose", "verification_status"]
    abt = df.select(*keep)
    out = cfg["paths"]["abt_path"]
    abt.write.mode("overwrite").parquet(out)
    abt = spark.read.parquet(out)

    summary = (abt.groupBy("sample")
               .agg(F.count(F.lit(1)).alias("loans"),
                    F.sum("bad_12m").alias("bads"),
                    F.round(F.avg("bad_12m"), 4).alias("bad_rate"),
                    F.min("issue_d").cast("string").alias("first_issue"),
                    F.max("issue_d").cast("string").alias("last_issue"))
               .orderBy("first_issue").toPandas())
    exclusions = abt.groupBy("exclusion_reason").count().toPandas()
    by_status = (abt.filter("sample in ('train','test_in_time','oot')")
                 .groupBy("loan_status", "bad_12m").count().orderBy("loan_status").toPandas())

    n = abt.count()
    controls = abt.agg(F.count(F.lit(1)).alias("row_count"),
                       F.countDistinct("id").alias("distinct_ids"),
                       F.sum(F.col("id") % 1000003).alias("id_checksum"),
                       F.round(F.sum("bad_12m"), 2).alias("sum__bad_12m"),
                       F.round(F.sum("fico"), 2).alias("sum__fico"),
                       F.round(F.sum("loan_to_income"), 2).alias("sum__loan_to_income"),
                       F.sum(F.col("bad_12m").isNull().cast("int")).alias("nulls__bad_12m"),
                       F.min("issue_d").cast("string").alias("min__issue_d"),
                       F.max("issue_d").cast("string").alias("max__issue_d")).first().asDict()
    spark.stop()

    res = Path(cfg["paths"]["results_dir"])
    summary.to_csv(res / "sample_summary.csv", index=False)
    by_status.to_csv(res / "status_to_target.csv", index=False)
    save_json(controls, res / "abt_control_totals.json")
    save_json({"snapshot_date": str(snapshot), "abt_rows": n,
               "exclusions": dict(zip(exclusions.exclusion_reason.fillna("none"), exclusions["count"]))},
              res / "target_meta.json")
    print(f"snapshot date: {snapshot}")
    print(summary.to_string(index=False))
    print(by_status.to_string(index=False))
    assert n == controls["distinct_ids"], "ABT is not one row per loan"


if __name__ == "__main__":
    main()
