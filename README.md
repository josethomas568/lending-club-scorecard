# Lending Club application scorecard with monitoring

A 12-month application PD model built the way a bank credit-risk team would build it:
a weight-of-evidence logistic **scorecard** as the benchmark, a **LightGBM** challenger,
out-of-time validation, **monitoring** (PSI, characteristic stability, vintage curves, score drift),
an independent **implementation test**, and a **model development document**.

## Results (Kaggle `accepted_2007_to_2018Q4`, 2.26M loans)

Development 2012–2015 (844k loans, 3.1% bad); out-of-time Jan 2016–Mar 2018 (918k loans, 3.9% bad).

| Out-of-time | AUC | Gini | KS |
|---|---|---|---|
| WoE scorecard (10 characteristics) | 0.661 | 0.321 | 0.229 |
| LightGBM challenger | 0.684 | 0.367 | 0.263 |

* Every row count reconciles from raw file to warehouse; the SQL rescoring matches all 2,097,423 development scores exactly.
* Both models rank-order well out of time but under-predict the level (3.9% realised vs 2.8% predicted): risk at a given score rose in the 2016–17 vintages, so PDs need recalibration.
* Full write-up, assumptions and decision log: [`docs/MDD.md`](docs/MDD.md).

![Back-test](outputs/figures/backtest_by_quarter.png)

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `pipeline/01_ingest_spark.py` | PySpark load of the Kaggle file → typed `loans`, `bureau`, `outcomes` parquet tables → join; row counts reconciled at every hop, anchored by an independent DuckDB count |
| 1b | `pipeline/02_bigquery_mirror.py` | Mirrors the tables into the BigQuery sandbox (load jobs only — the sandbox has no DML) and reconciles counts, checksums, null counts, sums and date ranges against Spark |
| 2 | `pipeline/03_target_and_split.py` | Observation point = issue date; bad = charged off / 90+ DPD within 12 months; features; out-of-time split by issue date |
| 3 | `pipeline/04_scorecard.py` | Monotonic WoE binning, IV screen, correlation screen, sign/significance pruning, points scaling (600 @ 50:1, PDO 20) |
| 3 | `pipeline/05_challenger_lgbm.py` | LightGBM with monotone constraints; early stopping on a slice of train |
| 3 | `pipeline/06_evaluate.py` | KS, AUC/Gini, lift tables, calibration plots, scorecard and SHAP reason codes |
| 4 | `pipeline/07_monitoring.py` | Quarterly PSI, characteristic PSI and points shift, score drift, back-test, vintage curves |
| 5 | `pipeline/08_implementation_test.py` | Rescores with `production/score_scorecard.sql` and `production/score_challenger.py`; reconciles exactly; runs a negative control |
| 6 | `pipeline/09_render_mdd.py` | Renders [`docs/MDD.md`](docs/MDD.md) from the results |

## Run it

Requires Python 3.11+ and Java 17 or 21 for Spark (`brew install openjdk@17` on a Mac).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Kaggle: wordsforthewise/lending-club → accepted_2007_to_2018Q4.csv.gz
kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
# point paths.raw_csv in config.yaml at the extracted .csv.gz if its folder differs

make all            # ingest → target → local mirror check → models → evaluate → monitor → implementation test → MDD
python -m pytest tests
```

BigQuery sandbox (no billing account needed):

```bash
gcloud auth application-default login
# set bigquery.project in config.yaml, then
make mirror          # loads mart tables, ABT and the points table; reconciles against Spark
# optional: run sql/score_scorecard_bigquery.sql in the console to rescore inside BigQuery
```

Sandbox limits worth knowing: 10 GiB storage cap, 1 TiB of queries a month, tables expire after 60 days,
no DML or streaming — so everything here is written with load jobs and `CREATE TABLE AS SELECT`.

### Smoke test without the Kaggle file

`make synthetic` writes a schema-faithful synthetic file (same columns, string formats such as `Dec-2015`
and `" 36 months"`, status labels and footer rows) with a known default process and built-in vintage drift.
It exists to test the code. The MDD renders a **"synthetic smoke test — not Lending Club results"** banner
whenever it was built from that file; the banner disappears automatically once the real file is in place.

## Key design choices

* **Target without payment history.** The file is one status snapshot, so the 90+ DPD date is estimated as
  `last_pymnt_d + 4 months`. See the MDD for the assumption and its limits.
* **Grade and interest rate are excluded** — they are Lending Club's own model output.
* **The points table is the hand-off.** Production SQL scores from it alone; integer points make the
  implementation test an exact match, and the negative control (a `<=` bin edge coded as `<`) proves the
  test can fail.
* **Champion rule.** LightGBM replaces the scorecard only with ≥ 3 Gini points out of time and calibration no worse.

## Layout

```
config.yaml            every threshold, window and path
src/lcscore/           binning, metrics, features, plotting, Spark session
pipeline/              numbered steps (see table)
production/            independent scoring: SQL for the scorecard, standalone Python for LightGBM
sql/                   generated BigQuery version of the scoring SQL
scripts/               synthetic smoke-test data
tests/                 binning conventions the SQL depends on
docs/MDD.md            model development document (generated)
outputs/               results CSVs, figures, model artifacts
```
