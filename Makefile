PY ?= python

.PHONY: all synthetic ingest mirror mirror-local target scorecard challenger evaluate monitor implement mdd

all: ingest target mirror-local scorecard challenger evaluate monitor implement mdd

synthetic:      ## smoke-test data only — NOT Lending Club results
	$(PY) scripts/make_synthetic_sample.py --scale 0.1

ingest:
	$(PY) pipeline/01_ingest_spark.py

mirror:         ## needs `gcloud auth application-default login` and bigquery.project in config.yaml
	$(PY) pipeline/02_bigquery_mirror.py --backend bigquery

mirror-local:   ## same reconciliation SQL, run on DuckDB over the local parquet
	$(PY) pipeline/02_bigquery_mirror.py --backend duckdb

target:
	$(PY) pipeline/03_target_and_split.py

scorecard:
	$(PY) pipeline/04_scorecard.py

challenger:
	$(PY) pipeline/05_challenger_lgbm.py

evaluate:
	$(PY) pipeline/06_evaluate.py

monitor:
	$(PY) pipeline/07_monitoring.py

implement:
	$(PY) pipeline/08_implementation_test.py

mdd:
	$(PY) pipeline/09_render_mdd.py
