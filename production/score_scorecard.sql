-- Production-style scorecard scoring, written independently of the Python
-- development code. Its only inputs are the mart tables and the exported points
-- table (model/scorecard_points.csv) — the implementation hand-off spec.
--
-- Placeholders (filled by pipeline/08_implementation_test.py for DuckDB, or by
-- hand for BigQuery — see sql/score_scorecard_bigquery.sql):
--   {loans} {bureau} {points}                 table references
--   {min_issue_date} {application_type} {policy_code}   population rules
--
-- Bin rules from the spec: numeric bins are lo < x <= hi (NULL bound = open);
-- NULL numeric -> the 'missing' row; NULL category -> 'MISSING'; a level not in
-- the spec -> the row flagged is_else.

CREATE OR REPLACE TABLE prod_features AS
SELECT
  l.id,
  l.issue_d,
  (b.fico_range_low + b.fico_range_high) / 2.0                         AS fico,
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
  CAST(date_diff('month', b.earliest_cr_line, l.issue_d) AS DOUBLE)  AS credit_hist_months,
  CASE WHEN l.emp_length = '< 1 year'  THEN 0.0
       WHEN l.emp_length = '10+ years' THEN 10.0
       ELSE TRY_CAST(NULLIF(regexp_extract(l.emp_length, '^([0-9]+)', 1), '') AS DOUBLE)
  END                                                                   AS emp_length_yrs,
  CASE WHEN l.annual_inc > 0 THEN l.loan_amnt / l.annual_inc END        AS loan_to_income,
  CAST(l.term_months AS VARCHAR)                                        AS term,
  CASE WHEN l.home_ownership IN ('MORTGAGE', 'RENT', 'OWN') THEN l.home_ownership
       ELSE 'OTHER' END                                                 AS home_ownership,
  l.purpose,
  l.verification_status
FROM {loans} AS l
JOIN {bureau} AS b ON b.id = l.id
WHERE l.issue_d >= DATE '{min_issue_date}'
  AND l.application_type = '{application_type}'
  AND l.policy_code = {policy_code}
  AND b.fico_range_low IS NOT NULL
  AND b.fico_range_high IS NOT NULL
  AND l.annual_inc IS NOT NULL;

CREATE OR REPLACE TABLE prod_long AS
SELECT id, 'fico' AS feature, fico AS num_value, CAST(NULL AS VARCHAR) AS cat_value FROM prod_features
UNION ALL SELECT id, 'revol_util', revol_util, NULL FROM prod_features
UNION ALL SELECT id, 'inq_last_6mths', inq_last_6mths, NULL FROM prod_features
UNION ALL SELECT id, 'delinq_2yrs', delinq_2yrs, NULL FROM prod_features
UNION ALL SELECT id, 'mths_since_last_delinq', mths_since_last_delinq, NULL FROM prod_features
UNION ALL SELECT id, 'open_acc', open_acc, NULL FROM prod_features
UNION ALL SELECT id, 'total_acc', total_acc, NULL FROM prod_features
UNION ALL SELECT id, 'pub_rec', pub_rec, NULL FROM prod_features
UNION ALL SELECT id, 'pub_rec_bankruptcies', pub_rec_bankruptcies, NULL FROM prod_features
UNION ALL SELECT id, 'mort_acc', mort_acc, NULL FROM prod_features
UNION ALL SELECT id, 'dti', dti, NULL FROM prod_features
UNION ALL SELECT id, 'annual_inc', annual_inc, NULL FROM prod_features
UNION ALL SELECT id, 'loan_amnt', loan_amnt, NULL FROM prod_features
UNION ALL SELECT id, 'revol_bal', revol_bal, NULL FROM prod_features
UNION ALL SELECT id, 'credit_hist_months', credit_hist_months, NULL FROM prod_features
UNION ALL SELECT id, 'emp_length_yrs', emp_length_yrs, NULL FROM prod_features
UNION ALL SELECT id, 'loan_to_income', loan_to_income, NULL FROM prod_features
UNION ALL SELECT id, 'term', NULL, term FROM prod_features
UNION ALL SELECT id, 'home_ownership', NULL, home_ownership FROM prod_features
UNION ALL SELECT id, 'purpose', NULL, purpose FROM prod_features
UNION ALL SELECT id, 'verification_status', NULL, verification_status FROM prod_features;

CREATE OR REPLACE TABLE prod_points AS
WITH spec AS (
  SELECT feature, bin_no, type, lo, hi, category, is_else, points FROM {points}
),
cat_map AS (
  SELECT s.feature, s.bin_no, s.points, c.cat
  FROM spec AS s, UNNEST(string_split(s.category, '|')) AS c(cat)
  WHERE s.type = 'category'
),
numeric_points AS (
  SELECT x.id, x.feature, s.bin_no, s.points
  FROM prod_long AS x
  JOIN spec AS s
    ON s.feature = x.feature
   AND s.type IN ('range', 'missing')
   AND ( (x.num_value IS NULL AND s.type = 'missing')
      OR (x.num_value IS NOT NULL AND s.type = 'range'
          AND (s.lo IS NULL OR x.num_value > s.lo)
          AND (s.hi IS NULL OR x.num_value <= s.hi)) )
),
category_points AS (
  SELECT x.id, x.feature,
         COALESCE(m.bin_no, e.bin_no) AS bin_no,
         COALESCE(m.points, e.points) AS points
  FROM prod_long AS x
  JOIN spec AS e
    ON e.feature = x.feature AND e.type = 'category' AND e.is_else
  LEFT JOIN cat_map AS m
    ON m.feature = x.feature AND m.cat = COALESCE(TRIM(x.cat_value), 'MISSING')
)
SELECT * FROM numeric_points
UNION ALL
SELECT * FROM category_points;

CREATE OR REPLACE TABLE prod_scores AS
SELECT id,
       SUM(points)                AS score,
       COUNT(*)                   AS characteristics_scored,
       COUNT(DISTINCT feature)    AS distinct_characteristics
FROM prod_points
GROUP BY id;
