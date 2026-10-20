-- GENERATED from production/score_scorecard.sql by pipeline/08_implementation_test.py
-- Load outputs/model/scorecard_points.csv into the dataset as `scorecard_points` first.
-- DDL (CREATE TABLE ... AS SELECT) is allowed in the BigQuery sandbox; DML is not.

/* Production-style scorecard scoring, written independently of the Python */
/* development code. Its only inputs are the mart tables and the exported points */
/* table (model/scorecard_points.csv) — the implementation hand-off spec. */
/* Placeholders (filled by pipeline/08_implementation_test.py for DuckDB, or by */
/* hand for BigQuery — see sql/score_scorecard_bigquery.sql): */
/*   `YOUR_GCP_PROJECT_ID.lending_club`.loans `YOUR_GCP_PROJECT_ID.lending_club`.bureau `YOUR_GCP_PROJECT_ID.lending_club`.scorecard_points                 table references */
/*   2012-01-01 Individual 1   population rules */
/* Bin rules from the spec: numeric bins are lo < x <= hi (NULL bound = open); */
/* NULL numeric -> the 'missing' row; NULL category -> 'MISSING'; a level not in */
/* the spec -> the row flagged is_else. */
CREATE OR REPLACE TABLE `YOUR_GCP_PROJECT_ID.lending_club`.prod_features AS
SELECT
  l.id,
  l.issue_d,
  (
    b.fico_range_low + b.fico_range_high
  ) / 2.0 AS fico,
  b.revol_util,
  b.inq_last_6mths,
  b.delinq_2yrs,
  b.mths_since_last_delinq,
  b.open_acc,
  b.total_acc,
  b.pub_rec,
  b.pub_rec_bankruptcies,
  b.mort_acc,
  l.dti,
  l.annual_inc,
  l.loan_amnt,
  b.revol_bal,
  CAST(DATE_DIFF(l.issue_d, b.earliest_cr_line, MONTH) AS FLOAT64) AS credit_hist_months,
  CASE
    WHEN l.emp_length = '< 1 year'
    THEN 0.0
    WHEN l.emp_length = '10+ years'
    THEN 10.0
    ELSE SAFE_CAST(NULLIF(REGEXP_EXTRACT(l.emp_length, '^([0-9]+)'), '') AS FLOAT64)
  END AS emp_length_yrs,
  CASE WHEN l.annual_inc > 0 THEN l.loan_amnt / l.annual_inc END AS loan_to_income,
  CAST(l.term_months AS STRING) AS term,
  CASE
    WHEN l.home_ownership IN ('MORTGAGE', 'RENT', 'OWN')
    THEN l.home_ownership
    ELSE 'OTHER'
  END AS home_ownership,
  l.purpose,
  l.verification_status
FROM `YOUR_GCP_PROJECT_ID.lending_club`.loans AS l
JOIN `YOUR_GCP_PROJECT_ID.lending_club`.bureau AS b
  ON b.id = l.id
WHERE
  l.issue_d >= CAST('2012-01-01' AS DATE)
  AND l.application_type = 'Individual'
  AND l.policy_code = 1
  AND NOT b.fico_range_low IS NULL
  AND NOT b.fico_range_high IS NULL
  AND NOT l.annual_inc IS NULL;

CREATE OR REPLACE TABLE `YOUR_GCP_PROJECT_ID.lending_club`.prod_long AS
SELECT
  id,
  'fico' AS feature,
  fico AS num_value,
  CAST(NULL AS STRING) AS cat_value
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'revol_util',
  revol_util,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'inq_last_6mths',
  inq_last_6mths,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'delinq_2yrs',
  delinq_2yrs,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'mths_since_last_delinq',
  mths_since_last_delinq,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'open_acc',
  open_acc,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'total_acc',
  total_acc,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'pub_rec',
  pub_rec,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'pub_rec_bankruptcies',
  pub_rec_bankruptcies,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'mort_acc',
  mort_acc,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'dti',
  dti,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'annual_inc',
  annual_inc,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'loan_amnt',
  loan_amnt,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'revol_bal',
  revol_bal,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'credit_hist_months',
  credit_hist_months,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'emp_length_yrs',
  emp_length_yrs,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'loan_to_income',
  loan_to_income,
  NULL
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'term',
  NULL,
  term
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'home_ownership',
  NULL,
  home_ownership
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'purpose',
  NULL,
  purpose
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features
UNION ALL
SELECT
  id,
  'verification_status',
  NULL,
  verification_status
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_features;

CREATE OR REPLACE TABLE `YOUR_GCP_PROJECT_ID.lending_club`.prod_points AS
WITH spec AS (
  SELECT
    feature,
    bin_no,
    type,
    lo,
    hi,
    category,
    is_else,
    points
  FROM `YOUR_GCP_PROJECT_ID.lending_club`.scorecard_points
), cat_map AS (
  SELECT
    s.feature,
    s.bin_no,
    s.points,
    cat
  FROM spec AS s, UNNEST(SPLIT(s.category, '|')) AS cat
  WHERE
    s.type = 'category'
), numeric_points AS (
  SELECT
    x.id,
    x.feature,
    s.bin_no,
    s.points
  FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_long AS x
  JOIN spec AS s
    ON s.feature = x.feature
    AND s.type IN ('range', 'missing')
    AND (
      (
        x.num_value IS NULL AND s.type = 'missing'
      )
      OR (
        NOT x.num_value IS NULL
        AND s.type = 'range'
        AND (
          s.lo IS NULL OR x.num_value > s.lo
        )
        AND (
          s.hi IS NULL OR x.num_value <= s.hi
        )
      )
    )
), category_points AS (
  SELECT
    x.id,
    x.feature,
    COALESCE(m.bin_no, e.bin_no) AS bin_no,
    COALESCE(m.points, e.points) AS points
  FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_long AS x
  JOIN spec AS e
    ON e.feature = x.feature AND e.type = 'category' AND e.is_else
  LEFT JOIN cat_map AS m
    ON m.feature = x.feature AND m.cat = COALESCE(TRIM(x.cat_value), 'MISSING')
)
SELECT
  *
FROM numeric_points
UNION ALL
SELECT
  *
FROM category_points;

CREATE OR REPLACE TABLE `YOUR_GCP_PROJECT_ID.lending_club`.prod_scores AS
SELECT
  id,
  SUM(points) AS score,
  COUNT(*) AS characteristics_scored,
  COUNT(DISTINCT feature) AS distinct_characteristics
FROM `YOUR_GCP_PROJECT_ID.lending_club`.prod_points
GROUP BY
  id;
