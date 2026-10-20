"""Step 1b — mirror the mart tables (and the ABT, once built) into the BigQuery
sandbox and reconcile them against Spark's control totals.

The sandbox needs no billing account but supports no DML and no streaming, so
tables are written with load jobs only (WRITE_TRUNCATE, then APPEND per parquet
part file, uploaded straight from disk — no Cloud Storage bucket needed).
Sandbox tables expire 60 days after creation; re-run this script to refresh.

    gcloud auth application-default login          # once
    python pipeline/02_bigquery_mirror.py           # load + reconcile in BigQuery
    python pipeline/02_bigquery_mirror.py --reconcile-only
    python pipeline/02_bigquery_mirror.py --backend duckdb   # same SQL, local dry run

Reconciliation compares, per table: row count, distinct ids, an id checksum,
null counts per column, column sums (relative tolerance 1e-9 for floating-point
summation order), and min/max of every date column.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from lcscore.config import load_config, load_json

TABLES = ["loans", "bureau", "outcomes"]


def reconciliation_sql(table_ref: str, controls: dict) -> str:
    exprs = ["COUNT(*) AS row_count",
             "COUNT(DISTINCT id) AS distinct_ids",
             "SUM(MOD(id, 1000003)) AS id_checksum"]
    for key in controls:
        if "__" not in key:
            continue
        kind, col = key.split("__", 1)
        if kind == "nulls":
            exprs.append(f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {key}")
        elif kind == "sum":
            exprs.append(f"ROUND(SUM({col}), 2) AS {key}")
        elif kind in ("min", "max"):
            exprs.append(f"CAST({kind.upper()}({col}) AS STRING) AS {key}")
    return f"SELECT\n  " + ",\n  ".join(exprs) + f"\nFROM {table_ref}"


def compare(table: str, expected: dict, actual: dict) -> list[dict]:
    rows = []
    for key, exp in expected.items():
        act = actual.get(key)
        if isinstance(exp, float) or isinstance(act, float):
            ok = (exp is None and act is None) or (
                exp is not None and act is not None
                and math.isclose(float(exp), float(act), rel_tol=1e-9, abs_tol=0.05))
        else:
            ok = str(exp) == str(act)
        rows.append({"table": table, "metric": key, "spark": exp,
                     "warehouse": act, "status": "PASS" if ok else "FAIL"})
    return rows


class BigQueryBackend:
    def __init__(self, cfg):
        from google.cloud import bigquery
        self.bq = bigquery
        self.client = bigquery.Client(project=cfg["bigquery"]["project"])
        self.dataset = f'{cfg["bigquery"]["project"]}.{cfg["bigquery"]["dataset"]}'
        ds = bigquery.Dataset(self.dataset)
        ds.location = cfg["bigquery"]["location"]
        self.client.create_dataset(ds, exists_ok=True)

    def ref(self, table):
        return f"`{self.dataset}.{table}`"

    def load(self, table, parquet_dir: Path):
        parts = sorted(parquet_dir.glob("*.parquet"))
        for i, part in enumerate(parts):
            job_config = self.bq.LoadJobConfig(
                source_format=self.bq.SourceFormat.PARQUET,
                write_disposition="WRITE_TRUNCATE" if i == 0 else "WRITE_APPEND")
            with open(part, "rb") as fh:
                self.client.load_table_from_file(
                    fh, f"{self.dataset}.{table}", job_config=job_config).result()
        print(f"loaded {table}: {len(parts)} part files")

    def load_csv(self, table, path: Path):
        job_config = self.bq.LoadJobConfig(source_format=self.bq.SourceFormat.CSV,
                                           skip_leading_rows=1, autodetect=True,
                                           write_disposition="WRITE_TRUNCATE")
        with open(path, "rb") as fh:
            self.client.load_table_from_file(fh, f"{self.dataset}.{table}",
                                             job_config=job_config).result()
        print(f"loaded {table} from {path.name}")

    def query(self, sql) -> dict:
        return dict(next(iter(self.client.query(sql).result())).items())


class DuckDBBackend:
    """Runs the identical reconciliation SQL over the local parquet files."""
    def __init__(self, cfg):
        import duckdb
        self.con = duckdb.connect()
        self.mart = Path(cfg["paths"]["mart_dir"])
        self.abt = Path(cfg["paths"]["abt_path"])

    def ref(self, table):
        path = self.abt if table == "abt" else self.mart / table
        glob = f"{path}/*.parquet" if path.is_dir() else str(path)
        return f"read_parquet('{glob}')"

    def load(self, table, parquet_dir):
        pass

    def query(self, sql) -> dict:
        res = self.con.sql(sql)
        return dict(zip(res.columns, res.fetchone()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["bigquery", "duckdb"], default="bigquery")
    ap.add_argument("--reconcile-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    results = Path(cfg["paths"]["results_dir"])
    controls = load_json(results / "control_totals.json")
    backend = BigQueryBackend(cfg) if args.backend == "bigquery" else DuckDBBackend(cfg)

    tables = list(TABLES)
    abt_controls = results / "abt_control_totals.json"
    if abt_controls.exists():
        controls["abt"] = load_json(abt_controls)
        tables.append("abt")

    if not args.reconcile_only:
        for t in tables:
            src = Path(cfg["paths"]["abt_path"]) if t == "abt" else Path(cfg["paths"]["mart_dir"]) / t
            backend.load(t, src)
        points = Path(cfg["paths"]["model_dir"]) / "scorecard_points.csv"
        if args.backend == "bigquery" and points.exists():
            backend.load_csv("scorecard_points", points)   # for sql/score_scorecard_bigquery.sql

    rows = []
    for t in tables:
        actual = backend.query(reconciliation_sql(backend.ref(t), controls[t]))
        rows += compare(t, controls[t], actual)
    rec = pd.DataFrame(rows)
    out = results / f"warehouse_reconciliation_{args.backend}.csv"
    rec.to_csv(out, index=False)
    summary = rec.groupby("table")["status"].value_counts().unstack(fill_value=0)
    print(summary.to_string())
    print(rec[rec.metric == "row_count"].to_string(index=False))
    if (rec.status == "FAIL").any():
        print(rec[rec.status == "FAIL"].to_string(index=False))
        sys.exit("warehouse reconciliation FAILED")
    print(f"all {len(rec)} checks pass -> {out}")


if __name__ == "__main__":
    main()
