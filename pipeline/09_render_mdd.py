"""Step 6 — render the model development document (docs/MDD.md) from the
pipeline's own results, so every number in it is reproducible.
The narrative (assumptions, limitations, decision log) lives here; tables
and figures are filled from outputs/.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from lcscore.config import REPO, load_config, load_json
from lcscore.features import LABELS


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in r) + " |")
    return "\n".join(lines)


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and v.is_integer() and abs(v) >= 1:
        v = f"{int(v):,}"
    elif isinstance(v, int) and not isinstance(v, bool):
        v = f"{v:,}"
    return str(v).replace("|", "\\|")


def pct(x, d=1):
    return f"{x:.{d}%}"


def main():
    cfg = load_config()
    P = {k: Path(v) for k, v in cfg["paths"].items()}
    R = P["results_dir"]
    synthetic = cfg["data_source"] == "synthetic"

    ingest = pd.read_csv(R / "ingest_reconciliation.csv")
    wh_file = R / "warehouse_reconciliation_bigquery.csv"
    wh_backend = "BigQuery"
    if not wh_file.exists():
        wh_file, wh_backend = R / "warehouse_reconciliation_duckdb.csv", "DuckDB dry run; BigQuery not yet run"
    wh = pd.read_csv(wh_file)
    meta = load_json(R / "target_meta.json")
    samples = pd.read_csv(R / "sample_summary.csv")
    status = pd.read_csv(R / "status_to_target.csv")
    iv = pd.read_csv(R / "iv_table.csv")
    sel = pd.read_csv(R / "selection_log.csv")
    perf = pd.read_csv(R / "performance.csv")
    spec = load_json(P["model_dir"] / "scorecard_spec.json")
    points = pd.read_csv(P["model_dir"] / "scorecard_points.csv")
    lift_sc = pd.read_csv(R / "lift_oot_scorecard.csv")
    lift_lg = pd.read_csv(R / "lift_oot_lgbm.csv")
    rc = load_json(R / "reason_code_meta.json")
    rc_sample = pd.read_csv(R / "reason_codes_sample.csv")
    mon = pd.read_csv(R / "monitoring_quarterly.csv")
    csi = pd.read_csv(R / "characteristic_stability.csv")
    impl = pd.read_csv(R / "implementation_test.csv")
    manifest = load_json(P["model_dir"] / "challenger_manifest.json")

    def m(model, sample, col):
        return perf.query("model == @model and sample == @sample")[col].iloc[0]

    SC, LG = "Scorecard (WoE logistic)", "LightGBM challenger"
    gini_up = m(LG, "oot", "gini") - m(SC, "oot", "gini")
    champion = LG if gini_up >= 0.03 and m(LG, "oot", "ece") <= m(SC, "oot", "ece") * 1.1 else SC
    oot_ratio = m(SC, "oot", "bad_rate") / m(SC, "oot", "mean_pd")
    red_q = mon[mon.status_score == "red"].quarter.tolist()
    amber_q = mon[mon.status_score == "amber"].quarter.tolist()
    worst_csi = (csi.sort_values("csi", ascending=False).drop_duplicates("feature").head(3))
    impl_pass = (impl.status != "FAIL").all()
    it_ratio = m(SC, "test_in_time", "bad_rate") / m(SC, "test_in_time", "mean_pd")
    calib_text = (f"In time, realised bad rate / mean PD = {it_ratio:.2f} "
                  f"({'well calibrated' if abs(it_ratio - 1) < 0.1 else 'miscalibrated'}). "
                  f"Out of time the ratio is {oot_ratio:.2f} ({pct(m(SC, 'oot', 'bad_rate'))} realised vs "
                  f"{pct(m(SC, 'oot', 'mean_pd'))} predicted). ")
    if abs(oot_ratio - 1) >= 0.1:
        calib_text += ("Rank ordering holds out of time but the PD level does not: changes in the applicant mix "
                       "(section 7) show up in the scores, but the realised bad rate "
                       f"{'exceeded' if oot_ratio > 1 else 'fell short of'} what they imply, so risk "
                       "at a given score has also moved (see the back-test by quarter). PDs used for pricing "
                       "or loss forecasting need a recalibration (intercept shift) on the latest mature "
                       "vintages; the score ranking can stay.")
    else:
        calib_text += "PD levels hold out of time; no recalibration is indicated."

    perf_tbl = perf.copy()
    perf_tbl["sample"] = perf_tbl["sample"].map({"train": "Train", "test_in_time": "In-time test",
                                                "oot": "Out-of-time"})
    perf_tbl = perf_tbl.assign(
        **{"Loans": perf_tbl.n.map("{:,}".format), "Bad rate": perf_tbl.bad_rate.map(pct),
           "Mean PD": perf_tbl.mean_pd.map(pct), "AUC": perf_tbl.auc.round(3),
           "Gini": perf_tbl.gini.round(3), "KS": perf_tbl.ks.round(3),
           "ECE": perf_tbl.ece.map(lambda v: f"{v:.4f}")})[
        ["model", "sample", "Loans", "Bad rate", "Mean PD", "AUC", "Gini", "KS", "ECE"]]
    perf_tbl.columns = ["Model", "Sample"] + list(perf_tbl.columns[2:])

    def lift_fmt(t):
        return t.assign(**{"Bad rate": t.bad_rate.map(pct), "Cum. bads captured": t.cum_bads_captured.map(pct),
                           "Lift": t.lift.round(2), "KS": t.ks.round(3), "bads": t.bads.astype(int)})[
            ["decile", "loans", "bads", "Bad rate", "Cum. bads captured", "Lift", "KS"]].rename(
            columns={"decile": "Decile (1 = riskiest)", "loans": "Loans", "bads": "Bads"})

    sample_tbl = samples.assign(bad_rate=samples.bad_rate.map(lambda v: "" if pd.isna(v) else pct(v, 2)),
                                loans=samples.loans.map("{:,}".format),
                                bads=samples.bads.map(lambda v: "" if pd.isna(v) else f"{int(v):,}"))
    sample_tbl["sample"] = sample_tbl["sample"].map({"train": "Train", "test_in_time": "In-time test",
        "oot": "Out-of-time", "monitor_only": "Monitoring only (immature)", "excluded": "Excluded"})
    sample_tbl.columns = ["Sample", "Loans", "Bads", "Bad rate", "First issue", "Last issue"]

    pts_tbl = points.assign(
        Characteristic=points.feature.map(LABELS),
        Attribute=points.apply(lambda r: r.category if r.type == "category" else (
            "Missing" if r.type == "missing" else
            f"{'' if pd.isna(r.lo) else f'> {r.lo:g}'}{' and ' if not pd.isna(r.lo) and not pd.isna(r.hi) else ''}"
            f"{'' if pd.isna(r.hi) else f'<= {r.hi:g}'}" or "All"), axis=1),
        Count=points["count"].map("{:,}".format), WoE=points.woe.round(3),
        Points=points.points)[["Characteristic", "Attribute", "Count", "WoE", "Points"]]
    pts_tbl["Attribute"] = pts_tbl.Attribute.str.replace("|", ", ", regex=False)

    iv_tbl = iv.merge(sel.drop_duplicates("feature", keep="last"), on="feature", how="left")
    iv_tbl = iv_tbl.assign(IV=iv_tbl.iv.round(4))[["label", "IV", "decision", "reason"]]
    iv_tbl.columns = ["Characteristic", "IV", "Decision", "Reason"]

    impl_tbl = impl.assign(value=impl.value.map(lambda v: f"{v:,.0f}" if float(v) >= 1 or float(v) == 0 else f"{float(v):.2e}"),
                           expected=impl.expected.map(lambda v: "" if pd.isna(v) else (f"{int(float(v)):,}" if str(v).replace('.', '').isdigit() else v)))
    impl_tbl.columns = ["Part", "Check", "Value", "Expected", "Status"]

    wh_sum = wh.groupby("table").status.apply(lambda s: f"{(s == 'PASS').sum()}/{len(s)} pass").reset_index()
    rows = wh[wh.metric == "row_count"][["table", "spark", "warehouse"]].merge(wh_sum, on="table")
    rows.columns = ["Table", "Spark rows", "Warehouse rows", "All checks"]

    banner = ("> **SYNTHETIC SMOKE TEST — these figures are not Lending Club results.** This copy was rendered "
              "from `scripts/make_synthetic_sample.py`, a schema-faithful stand-in used to test the pipeline. "
              "Run `make all` on the Kaggle file to regenerate every table and figure below.\n\n"
              if synthetic else "")

    doc = f"""# Model development document — Lending Club application scorecard

{banner}| | |
|---|---|
| Model | 12-month application PD scorecard (champion candidate) with LightGBM challenger |
| Data | Lending Club accepted loans 2007–2018 (Kaggle `accepted_2007_to_2018Q4`), snapshot {meta['snapshot_date']} |
| Developer | Jose Thomas |
| Version / date | v1.0 / {date.today():%Y-%m-%d} |
| Status | Development — not approved for use |

## 1. Purpose and scope

Rank and price new unsecured personal-loan applications by their probability of going bad within 12 months of issue,
using only information known at application. The scorecard is the interpretable benchmark (points per attribute,
adverse-action reasons by construction); LightGBM is the challenger that tests how much discrimination the
scorecard leaves on the table. Scope: individual (non-joint) applications, policy code 1, issued from
{cfg['population']['min_issue_date'][:4]} onward.

## 2. Data and reconciliation

The raw file is loaded in PySpark, typed, and split into three mart tables (`loans`, `bureau`, `outcomes`) that are
joined back into an analytic base table. Each hop is counted; an independent DuckDB count of the raw file anchors the chain.

{md_table(ingest.rename(columns={'check': 'Check', 'rows': 'Rows', 'expected': 'Expected', 'status': 'Status'}).fillna(''))}

Warehouse mirror ({wh_backend}): row counts, distinct ids, an id checksum, null counts, column sums and date ranges
are compared table by table against Spark's control totals.

{md_table(rows)}

## 3. Target definition and windows

* **Observation point:** issue date. All characteristics are as reported at application.
* **Performance window:** {cfg['target']['performance_months']} months from issue.
* **Bad:** charged off / default, or 90+ days past due, with the 90+ DPD date inside the window.
* **Good:** every other loan with a complete window — including loans repaid early and loans that went bad after month 12.
* **Immature:** issued less than 12 months before the {meta['snapshot_date']} snapshot; scored for monitoring, never labelled.

The file is a single status snapshot with no monthly payment history, so the 90+ DPD date is estimated as
`last_pymnt_d + {cfg['target']['dpd90_lag_months']} months` (the missed instalment falls due a month after the last
payment; 90 days later the loan is 90+ DPD). How snapshot statuses map to the 12-month flag in the modelling samples:

{md_table(status.rename(columns={'loan_status': 'Snapshot status', 'bad_12m': 'Bad within 12m', 'count': 'Loans'}))}

### Samples (out-of-time split by issue date)

{md_table(sample_tbl)}

Train and in-time test split the {cfg['split']['dev_start'][:4]}–{cfg['split']['dev_end'][:4]} development window
{int(cfg['split']['train_share']*100)}/{100-int(cfg['split']['train_share']*100)} on a hash of the loan id. Out-of-time
covers mature loans issued after the development window. Exclusions: {', '.join(f"{k} ({v:,})" for k, v in meta['exclusions'].items() if k != 'none')}.

## 4. Methodology

**Benchmark scorecard.** Each candidate characteristic is fine-classed into {cfg['scorecard']['fine_bins']} quantile bins,
then coarse-classed by merging adjacent bins until weight of evidence is monotonic, every bin holds at least
{pct(cfg['scorecard']['min_bin_share'], 0)} of training loans, and no characteristic has more than {cfg['scorecard']['max_bins']} bins.
Characteristics with IV below {cfg['scorecard']['min_iv']} are dropped, the weaker of any pair with WoE correlation above
{cfg['scorecard']['max_corr']} is dropped, and the logistic regression is pruned one characteristic at a time for wrong-signed
coefficients or p-values above {cfg['scorecard']['max_pvalue']}. Scores are scaled to {spec['base_score']} points at
{spec['base_odds']}:1 good:bad odds with {spec['pdo']} points to double the odds; attribute points are rounded to integers.

{md_table(iv_tbl)}

Final scorecard: {len(spec['features'])} characteristics, pseudo-R² {spec['pseudo_r2']:.3f} on {spec['train_n']:,} training loans.
Full points table in Appendix A.

**Challenger.** LightGBM on all {len(manifest['features'])} candidate characteristics, un-binned, with native handling of
categoricals and missing values. Early stopping (best iteration {manifest['best_iteration']}) uses a 10% slice of the
training sample so the in-time test and out-of-time samples are never seen during fitting. Monotone constraints are
imposed on {', '.join(LABELS[f] for f in manifest['monotone_constraints'])}, so a better value can never raise PD
and reason codes stay defensible.

## 5. Performance

{md_table(perf_tbl)}

ECE = expected calibration error across PD deciles. Out-of-time, the challenger's Gini is
{'higher' if gini_up > 0 else 'lower'} by {abs(gini_up)*100:.1f} points.

![ROC and capture](../outputs/figures/roc_capture_oot.png)

### Lift — out-of-time, scorecard

{md_table(lift_fmt(lift_sc))}

### Lift — out-of-time, LightGBM

{md_table(lift_fmt(lift_lg))}

![Score distribution](../outputs/figures/score_distribution_oot.png)

### Calibration

![Calibration](../outputs/figures/calibration.png)

{calib_text}

## 6. Reason codes

* **Scorecard:** for each characteristic, the points shortfall against the maximum attainable points; the four largest shortfalls are the reasons.
* **LightGBM:** exact TreeSHAP contributions in log-odds (LightGBM `pred_contrib`; additivity verified to {rc['shap_additivity_max_abs_error']:.1e}); the four largest positive contributions are the reasons.

In the decline zone (riskiest 20% by score, out-of-time) the two methods give the same primary reason for
{pct(rc['reason_code_top1_agreement_decline_zone'], 0)} of applicants and share {pct(rc['reason_code_top4_overlap_decline_zone'], 0)} of their top-four reasons.
Sample (lowest out-of-time scores):

{md_table(rc_sample.head(8)[['id', 'score', 'scorecard_reason_1', 'scorecard_reason_2', 'lgbm_reason_1', 'lgbm_reason_2']].rename(columns={'id': 'Loan id', 'score': 'Score', 'scorecard_reason_1': 'Scorecard #1', 'scorecard_reason_2': 'Scorecard #2', 'lgbm_reason_1': 'SHAP #1', 'lgbm_reason_2': 'SHAP #2'}))}

![SHAP importance](../outputs/figures/shap_importance.png)

## 7. Monitoring

Every issue quarter is scored and compared with the training baseline: score PSI on baseline deciles, characteristic
PSI and points shift for each scorecard characteristic, score percentiles, and predicted vs realised bad rate for
mature quarters. Thresholds: PSI < {cfg['monitoring']['psi_amber']} green, {cfg['monitoring']['psi_amber']}–{cfg['monitoring']['psi_red']} amber (investigate), ≥ {cfg['monitoring']['psi_red']} red (escalate; recalibrate or redevelop).

* Amber quarters: {', '.join(amber_q) or 'none'}.
* Red quarters: {', '.join(red_q) or 'none'}.
* Largest characteristic shifts: {'; '.join(f"{LABELS[r.feature]} PSI {r.csi:.2f} in {r.quarter} ({r.points_shift:+.1f} points)" for r in worst_csi.itertuples())}.

![Stability and drift](../outputs/figures/stability_and_drift.png)

![Characteristic stability](../outputs/figures/characteristic_stability.png)

![Back-test](../outputs/figures/backtest_by_quarter.png)

![Vintage curves](../outputs/figures/vintage_curves.png)

## 8. Implementation test

The scorecard was rescored by `production/score_scorecard.sql` (DuckDB; a BigQuery translation is generated alongside)
from the mart tables and the exported points table alone, and the challenger by `production/score_challenger.py`, a
standalone script that rebuilds every feature itself. A negative control re-runs the SQL with one bin edge coded as
`<` instead of `<=`; the test must catch it. Overall: **{'PASS' if impl_pass else 'FAIL'}**.

{md_table(impl_tbl)}

## 9. Recommendation

Champion: **{champion}**. {'The challenger clears the adoption bar (≥ 3 Gini points out of time with no worse calibration).' if champion == LG else f'The challenger adds {gini_up*100:.1f} Gini points out of time, below the 3-point bar for giving up a points-based, directly auditable model; it is retained as an ongoing benchmark.'}
Before use: recalibrate PD levels on the most recent mature vintages and confirm the monitoring thresholds with the model owner.

## 10. Assumptions

1. The 90+ DPD date is `last_pymnt_d + {cfg['target']['dpd90_lag_months']}` months. Loans that make partial payments while delinquent will have their bad date estimated late.
2. A loan that reached 90+ DPD and later cured is invisible in a snapshot; it is treated as good. Cures from 90+ DPD are rare in unsecured consumer lending, so the bad rate is understated only slightly.
3. Bureau attributes in the file are those reported at application, as Lending Club documents them.
4. Accepted loans are representative of future applicants who would be accepted (no reject inference; see limitations).
5. Snapshot date = latest `last_pymnt_d` in the file ({meta['snapshot_date']}).

## 11. Limitations

1. **Accepted-only sample.** Rejected applicants' outcomes are unknown, so the model is trained on a population already filtered by Lending Club's policy and scoring. Discrimination on the full through-the-door population is likely overstated. Reject inference (e.g. parcelling with the Kaggle `rejected` file, which carries only a few attributes) is the next step.
2. **No payment history.** The target uses an estimated 90+ DPD date rather than observed monthly delinquency; roll rates and cure behaviour cannot be modelled.
3. **Lending-policy changes** inside the window (credit-policy tightening, new fields added mid-period, joint applications from late 2015) mean some characteristics are not comparable across years. Null rates by vintage (in `binning_detail.csv` and the CSI heatmap) should be checked before trusting any characteristic whose population changes mid-period.
4. **Out-of-time level shift.** PDs under-predict out-of-time vintages; see section 5.
5. **Fair lending.** No protected attributes are present, but home ownership and income can proxy for them; disparate-impact testing is outside this project's data.
6. **Public data.** Lending Club's own grade and interest rate are excluded, so this is not a replication of Lending Club's model.

## 12. Decision log

| # | Decision | Alternatives considered | Rationale |
|---|---|---|---|
| 1 | Bad = CO/default or 90+ DPD within 12 months | 18/24-month windows; lifetime charge-off | 12 months is the standard application-scorecard horizon and leaves the most recent vintages usable for out-of-time testing. |
| 2 | Estimate the 90+ DPD date from `last_pymnt_d` | Use charge-off status only, ignoring timing | Without timing, loans charged off at month 30 would count as 12-month bads. |
| 3 | Out-of-time split by issue date; in-time split by id hash | Random split only | Out-of-time is the honest test for a model that will score future vintages; the id hash keeps splits reproducible across Spark and pandas. |
| 4 | Start the population in {cfg['population']['min_issue_date'][:4]}; individual applications; policy code 1 | Use all years and application types | Pre-2012 vintages are small and pre-date current policy; joint applications carry a second borrower's bureau file. |
| 5 | Exclude grade, sub-grade and interest rate | Include them | They are outputs of Lending Club's own risk model; including them would turn the scorecard into a re-weighting of that model. |
| 6 | Exclude post-origination fields (last FICO, payments, recoveries) | — | Target leakage. |
| 7 | Monotonic WoE bins, ≥ {pct(cfg['scorecard']['min_bin_share'], 0)} per bin, ≤ {cfg['scorecard']['max_bins']} bins | Free-form or tree-based bins | Monotone points are explainable to applicants and stable over time. |
| 8 | High IV flagged for review, not auto-dropped | Drop IV > {cfg['scorecard']['max_iv']} | FICO legitimately carries high IV; leakage is a question about timing, which a human must answer. |
| 9 | Integer points; SQL scoring from the points table | Score from coefficients | The points table is the hand-off artefact; integer points make the implementation test an exact match. |
| 10 | Early stopping on a slice of train | Early stopping on in-time test or OOT | Keeps both test samples untouched. |
| 11 | Monotone constraints in LightGBM | Unconstrained | Defensible reason codes and behaviour; costs little discrimination. |
| 12 | Scorecard reasons by points shortfall; LightGBM by TreeSHAP | Permutation importance; LIME | Both are exact and per-applicant. |
| 13 | PSI thresholds 0.10 / 0.25 | Other cut-offs | Industry convention; revisit with the model owner. |
| 14 | Challenger adoption bar: +3 Gini points out of time, calibration no worse | Adopt whichever scores higher | Interpretability and implementation cost justify a margin. |

## Appendix A — scorecard points table

{md_table(pts_tbl)}

Missing values take the riskiest bin's points unless they are frequent enough to earn their own bin; an unseen
category takes the riskiest category group's points.
"""
    out = REPO / "docs/MDD.md"
    out.write_text(doc)
    print(f"wrote {out} ({len(doc.split()):,} words)")


if __name__ == "__main__":
    main()
