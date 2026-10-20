"""Step 5 — implementation test: rescore with independent production code and
reconcile against the development scores.

  A. Scorecard: production/score_scorecard.sql run in DuckDB over the mart
     parquet and the exported points table. Integer scores must match exactly,
     characteristic by characteristic.
  B. Challenger: production/score_challenger.py run as a separate process from
     the model file + manifest. PDs must match to 1e-9.
  C. Negative control: the same SQL run against a deliberately corrupted points
     table (one bin boundary moved). The test must detect it — proof it has power.

Also writes sql/score_scorecard_bigquery.sql, a BigQuery translation of the same
SQL for running against the mirrored tables.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import numpy as np
import pandas as pd

from lcscore.config import REPO, load_config, load_json, save_json


def run_sql(cfg, points_csv: Path, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the production SQL; return its scores and per-characteristic points
    mismatches against development. The characteristic comparison runs inside
    DuckDB so ~10M point rows never have to be materialised in pandas."""
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    pop = cfg["population"]
    sql = (REPO / "production/score_scorecard.sql").read_text().format(
        loans=f"read_parquet('{P['mart_dir']}/loans/*.parquet')",
        bureau=f"read_parquet('{P['mart_dir']}/bureau/*.parquet')",
        points=f"read_csv('{points_csv}', header=true)",
        min_issue_date=pop["min_issue_date"], application_type=pop["application_type"],
        policy_code=pop["policy_code"])
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'")
    con.execute(f"SET temp_directory = '{P['model_dir'] / 'duckdb_tmp'}'")
    con.execute(sql)
    dev = f"read_parquet('{P['model_dir'] / 'scores_scorecard.parquet'}')"
    by_char = []
    for f in features:
        n = con.sql(f"""
            SELECT COUNT(*) FROM {dev} AS d
            LEFT JOIN prod_points AS p ON p.id = d.id AND p.feature = '{f}'
            WHERE p.points IS NULL OR p.points <> d.pts_{f}""").fetchone()[0]
        by_char.append({"feature": f, "points_mismatches": int(n)})
    scores = con.sql("SELECT id, score, characteristics_scored FROM prod_scores").df()
    con.close()
    return scores, pd.DataFrame(by_char)


def reconcile_scorecard(dev: pd.DataFrame, prod: pd.DataFrame, by_char: pd.DataFrame,
                        features: list[str], spec: dict) -> tuple[list[dict], pd.DataFrame]:
    m = dev.merge(prod, on="id", how="outer", indicator=True)
    both = m[m._merge == "both"]
    diff = (both.score_x - both.score_y).abs()
    prod_pd = 1 / (1 + np.exp((both.score_y - spec["offset"]) / spec["factor"]))
    checks = [
        ("A", "loans scored — development", len(dev), None),
        ("A", "loans scored — production SQL", len(prod), len(dev)),
        ("A", "in development only", int((m._merge == "left_only").sum()), 0),
        ("A", "in production only", int((m._merge == "right_only").sum()), 0),
        ("A", "loans not scored on every characteristic",
         int((prod.characteristics_scored != len(features)).sum()), 0),
        ("A", "score mismatches (exact integer match required)", int((diff > 0).sum()), 0),
        ("A", "max |score difference|", float(diff.max()), 0),
        ("A", "max |PD difference|", float((both.pd - prod_pd).abs().max()), None),
        ("A", "characteristic-level points mismatches", int(by_char.points_mismatches.sum()), 0),
    ]
    return [dict(zip(["part", "check", "value", "expected"], c)) for c in checks], by_char


def main():
    cfg = load_config()
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    tol = cfg["implementation_test"]
    spec = load_json(P["model_dir"] / "scorecard_spec.json")
    features = spec["features"]
    points_csv = P["model_dir"] / "scorecard_points.csv"

    dev = pd.read_parquet(P["model_dir"] / "scores_scorecard.parquet")
    dev = dev[["id", "score", "pd"]]

    # ---- A. scorecard via SQL ---------------------------------------------
    prod, prod_chars = run_sql(cfg, points_csv, features)
    checks, by_char = reconcile_scorecard(dev, prod, prod_chars, features, spec)

    # ---- B. challenger via standalone production script ---------------------
    prod_ch_path = P["model_dir"] / "prod_scores_challenger.parquet"
    pop = cfg["population"]
    subprocess.run([sys.executable, str(REPO / "production/score_challenger.py"),
                    "--mart", str(P["mart_dir"]),
                    "--model", str(P["model_dir"] / "challenger_lgbm.txt"),
                    "--manifest", str(P["model_dir"] / "challenger_manifest.json"),
                    "--min-issue-date", pop["min_issue_date"],
                    "--application-type", pop["application_type"],
                    "--policy-code", str(pop["policy_code"]),
                    "--out", str(prod_ch_path)], check=True)
    dev_ch = pd.read_parquet(P["model_dir"] / "scores_challenger.parquet")[["id", "pd"]]
    prod_ch = pd.read_parquet(prod_ch_path)
    mc = dev_ch.merge(prod_ch, on="id", how="outer", indicator=True)
    both = mc[mc._merge == "both"]
    pd_diff = (both.pd_x - both.pd_y).abs()
    checks += [dict(zip(["part", "check", "value", "expected"], c)) for c in [
        ("B", "loans scored — development", len(dev_ch), None),
        ("B", "loans scored — production script", len(prod_ch), len(dev_ch)),
        ("B", "in development only", int((mc._merge == "left_only").sum()), 0),
        ("B", "in production only", int((mc._merge == "right_only").sum()), 0),
        ("B", f"PD mismatches beyond {tol['pd_tolerance']:g}", int((pd_diff > tol["pd_tolerance"]).sum()), 0),
        ("B", "max |PD difference|", float(pd_diff.max()), None),
    ]]

    # ---- C. negative control ----------------------------------------------
    pts = pd.read_csv(points_csv)
    fico_rows = pts[(pts.feature == features[0]) & pts.hi.notna()]
    target = fico_rows.index[len(fico_rows) // 2]
    edge = pts.loc[target, "hi"]
    nxt = target + 1
    # The classic implementation bug: "<=" coded as "<" at a bin edge. Moving the
    # edge to the next float below it reproduces exactly that error.
    moved = float(np.nextafter(edge, -np.inf))
    pts.loc[target, "hi"] = moved
    pts.loc[nxt, "lo"] = moved
    broken = P["model_dir"] / "scorecard_points_NEGATIVE_CONTROL.csv"
    pts.to_csv(broken, index=False)
    bad_prod, bad_chars = run_sql(cfg, broken, features)
    bad_checks, _ = reconcile_scorecard(dev, bad_prod, bad_chars, features, spec)
    caught = next(c["value"] for c in bad_checks if c["check"].startswith("score mismatches"))
    checks.append({"part": "C", "check": f"negative control: '{features[0]}' bin edge {edge:g} coded as < "
                   "instead of <=; mismatches detected", "value": caught, "expected": "> 0"})
    broken.unlink()

    res = pd.DataFrame(checks)

    def status(r):
        if r.expected is None or (isinstance(r.expected, float) and np.isnan(r.expected)):
            return "info"
        if r.expected == "> 0":
            return "PASS" if r.value > 0 else "FAIL"
        return "PASS" if r.value == r.expected else "FAIL"

    res["status"] = res.apply(status, axis=1)
    res.to_csv(P["results_dir"] / "implementation_test.csv", index=False)
    by_char.to_csv(P["results_dir"] / "implementation_test_by_characteristic.csv", index=False)
    save_json({"passed": bool((res.status != "FAIL").all())}, P["results_dir"] / "implementation_test.json")

    # BigQuery translation of the scoring SQL for the mirrored tables
    try:
        import sqlglot
        ds = f'{cfg["bigquery"]["project"]}.{cfg["bigquery"]["dataset"]}'
        tmpl = (REPO / "production/score_scorecard.sql").read_text().format(
            loans="lcds.loans", bureau="lcds.bureau", points="lcds.scorecard_points",
            min_issue_date=pop["min_issue_date"], application_type=pop["application_type"],
            policy_code=pop["policy_code"])
        stmts = sqlglot.transpile(tmpl, read="duckdb", write="bigquery", pretty=True)
        stmts = [s.replace("lcds.", f"`{ds}`.")
                  .replace("CREATE OR REPLACE TABLE prod_", f"CREATE OR REPLACE TABLE `{ds}`.prod_")
                  .replace("FROM prod_", f"FROM `{ds}`.prod_") for s in stmts]
        header = ("-- GENERATED from production/score_scorecard.sql by pipeline/08_implementation_test.py\n"
                  "-- Load outputs/model/scorecard_points.csv into the dataset as `scorecard_points` first.\n"
                  "-- DDL (CREATE TABLE ... AS SELECT) is allowed in the BigQuery sandbox; DML is not.\n\n")
        (REPO / "sql/score_scorecard_bigquery.sql").write_text(header + ";\n\n".join(stmts) + ";\n")
    except Exception as exc:  # translation is a convenience, not part of the test
        print(f"BigQuery translation skipped: {exc}")

    print(res.to_string(index=False))
    print(by_char.to_string(index=False))
    if (res.status == "FAIL").any():
        sys.exit("implementation test FAILED")


if __name__ == "__main__":
    main()
