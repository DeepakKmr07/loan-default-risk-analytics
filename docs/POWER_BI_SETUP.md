# Power BI Setup Guide

This guide connects Power BI to the curated Parquet marts in `data/curated/` and assembles the
three-page executive risk dashboard on top of them.

## 1. Getting the data in

1. **Get Data > More > Text/CSV or Parquet** (Power BI Desktop 2023+ has a native Parquet
   connector under *Get Data > File > Parquet*). Point it at each file in `data/curated/`:
   - `Fact_Loan_Risk_Portfolio.parquet`
   - `Fact_Stress_Test_Scenarios.parquet` (after running `stress_test.py` or
     `run_pipeline.py --stress-test`)
   - `Dim_Borrower.parquet`
   - `Dim_Vintage.parquet`
   - `Dim_Credit_Grade.parquet`
2. Load all five as separate tables (do not append/merge them in Power Query — the star schema
   relationships are built in the Power BI model, not by flattening in the query layer).
3. In each table's column view, hide the surrogate key columns from report view *after* wiring
   relationships (Section 2) — reports should filter through slicers/visuals bound to descriptive
   dimension columns (grade, vintage, state, etc.), not raw keys.

## 2. Table relationships

Both fact tables share the same three conformed dimensions (a fact constellation / galaxy
schema), so each dimension only needs to be loaded once and related to both facts:

| From (fact) | Column | To (dimension) | Column | Cardinality | Cross-filter |
|---|---|---|---|---|---|
| `Fact_Loan_Risk_Portfolio` | `borrower_key` | `Dim_Borrower` | `borrower_key` | 1:many (Dim → Fact) | Single |
| `Fact_Loan_Risk_Portfolio` | `vintage_key` | `Dim_Vintage` | `vintage_key` | 1:many | Single |
| `Fact_Loan_Risk_Portfolio` | `credit_grade_key` | `Dim_Credit_Grade` | `credit_grade_key` | 1:many | Single |
| `Fact_Stress_Test_Scenarios` | `borrower_key` | `Dim_Borrower` | `borrower_key` | 1:many | Single |
| `Fact_Stress_Test_Scenarios` | `vintage_key` | `Dim_Vintage` | `vintage_key` | 1:many | Single |
| `Fact_Stress_Test_Scenarios` | `credit_grade_key` | `Dim_Credit_Grade` | `credit_grade_key` | 1:many | Single |

`Fact_Stress_Test_Scenarios` also has a `scenario` text column (`Baseline` / `Adverse` /
`Severely Adverse`) with no dedicated dimension table. Add one manually so the scenario slicer
sorts in severity order rather than alphabetically:

1. **Modeling > New Table**:
   ```
   Dim_Scenario = DATATABLE(
       "scenario", STRING, "sort_order", INTEGER,
       {
           {"Baseline", 1},
           {"Adverse", 2},
           {"Severely Adverse", 3}
       }
   )
   ```
2. Relate `Dim_Scenario[scenario]` (1) to `Fact_Stress_Test_Scenarios[scenario]` (many).
3. On `Dim_Scenario[scenario]`, **Column tool > Sort by column > sort_order**.

Load `dax/credit_risk_measures.dax` into a new Measures table (or paste each measure directly
into `Fact_Loan_Risk_Portfolio`) via **Modeling > New Measure**.

### Known modeling limitation: no true calendar date table

`Dim_Vintage` is grain-limited to `issue_year` / `issue_quarter` (one row per origination
quarter) — it is **not** a continuous calendar date table. This is fine for quarter-over-quarter
(QoQ) trending, which is what the vintage dimension actually supports, but Power BI's built-in
time-intelligence functions (`DATEADD`, `SAMEPERIODLASTYEAR`, true month-over-month deltas)
require a marked date table with one row per calendar day or month. If month-over-month trends
are required, extend `build_marts.py`'s `DIM_VINTAGE_SQL` to also emit `issue_month`, mark the
resulting table as a date table (**Modeling > Mark as date table**), and build MoM measures
against that — the current schema only supports QoQ out of the box.

## 3. Page 1 — Executive Summary

**Layout:** a top row of 4 KPI cards, a trend chart beneath, and a grade breakdown table.

| Visual | Type | Fields |
|---|---|---|
| Total EAD | Card | `[Total EAD]` |
| Weighted PD | Card | `[Weighted PD]` |
| Expected Loss ($) | Card | `[Total Expected Loss]` |
| EL Rate (%) | Card | `[EL Rate %]` |
| EAD & EL trend | Line/clustered column combo | Axis: `Dim_Vintage[vintage]` (sorted by `vintage_key`); Values: `[Total EAD]` (columns), `[Total Expected Loss]` (line, secondary axis) |
| QoQ EL change | Line chart | Axis: `Dim_Vintage[vintage]`; Values: a QoQ delta measure — `EL QoQ Delta = [Total Expected Loss] - CALCULATE([Total Expected Loss], FILTER(ALL(Dim_Vintage), Dim_Vintage[vintage_key] = MAX(Dim_Vintage[vintage_key]) - 1))` |
| Expected Loss by Grade | Bar chart | Axis: `Dim_Credit_Grade[grade]`; Values: `[Total Expected Loss]` |
| Interest Rate vs PD | Scatter chart | see note below |
| Grade breakdown | Table/matrix | Rows: `Dim_Credit_Grade[grade]`; Values: `[Total EAD]`, `[Weighted PD]`, `[Total Expected Loss]`, `[EL Rate %]` |

**Scatter chart note:** a scatter needs one point per *something* — plotting `[Weighted PD]`
(a portfolio-level aggregate) with no other dimension collapses to a single dot, not a scatter.
Two ways to do it properly:
- **One point per sub-grade** (recommended, ~35 points, readable): X = average `int_rate` by
  `Dim_Credit_Grade[sub_grade]`, Y = `[Weighted PD]`, both aggregated with `Dim_Credit_Grade[sub_grade]`
  in the visual's Details field — this is the grade-level risk-pricing curve.
- **One point per loan** (matches the Streamlit app's version, but 2.26M points will choke the
  visual): use `Fact_Loan_Risk_Portfolio[int_rate]` and `Fact_Loan_Risk_Portfolio[pd]` directly
  (not the `[Weighted PD]` measure) with **Sampling** enabled (Format visual > General > a top-N
  or fixed sample), or pre-aggregate a sampled table in Power Query first.

Add slicers for `Dim_Vintage[vintage]` and `Dim_Credit_Grade[grade]` so the whole page can be
filtered to a specific cohort.

## 4. Page 2 — Credit Migration & Vintage Curves

**Important scope note:** this dataset has one row per loan at origination — it is a snapshot,
not a monthly loan-performance panel. There is no literal month-by-month delinquency roll-rate
history to build a true transition matrix (e.g. "% of loans that moved from 30dpd to 60dpd").
What the DAX file's `[Grade Mix Shift (pp)]` and `[Risk Migration (PD Delta)]` measures give you
instead is a **cohort-migration view**: how credit mix and modeled risk shift vintage-over-vintage
— which is the standard proxy used when granular loan-performance history isn't available. Build
the page around that, not a literal delinquency transition matrix.

| Visual | Type | Fields |
|---|---|---|
| Grade mix shift heatmap | Matrix (conditional formatting) | Rows: `Dim_Vintage[vintage]`; Columns: `Dim_Credit_Grade[sub_grade]`; Values: `[Grade Mix Shift (pp)]`, color scale red (loosening toward that grade) to blue (tightening away) |
| Sub-Grade vs Loan Status | Matrix (conditional formatting) | Rows: `Dim_Credit_Grade[sub_grade]`; Columns: `Fact_Loan_Risk_Portfolio[loan_status]`; Values: `[Loan Count]` — right-click the value → **Show value as → Percent of row total** so each row reads as that sub-grade's status mix (%), not raw counts. Same caveat as the DAX file's design notes: this is a status-composition snapshot, not a month-over-month transition matrix. |
| Weighted PD by vintage & sub-grade | Line chart, small multiples by grade | Axis: `Dim_Vintage[vintage]`; Values: `[Weighted PD]`; Small multiple: `Dim_Credit_Grade[grade]` |
| Vintage curve | Line chart | Axis: `Dim_Vintage[vintage]`; Values: `[Weighted PD]`, `[Realized Default Rate]` (both series, to compare modeled vs realized risk by cohort) |
| PD Calibration Gap | Card / gauge | `[PD Calibration Gap]` — pulls in the governance metric from `reports/model_risk_summary.json`; enter it as a manual card or a small disconnected table if you want it live in the model |

## 5. Page 3 — Stress Testing & What-If Planner

**Data:** `Fact_Stress_Test_Scenarios` (already contains the three pre-computed scenarios) plus
the What-If parameters from `dax/credit_risk_measures.dax`.

1. **Modeling > New Parameter > Numeric range** twice, exactly matching the DAX file's comments:
   - `PD Stress Multiplier`: min 1.0, max 3.0, increment 0.1, default 1.0
   - `LGD Stress Multiplier`: min 1.0, max 2.0, increment 0.1, default 1.0

   Each creates a disconnected table and a `[<Name> Value]` measure automatically — these feed
   `[Stressed Expected Loss]`, `[Stress EL Delta]`, and `[Stress EL Rate %]` from the DAX file.

2. **Scenario slicer:** a slicer visual on `Fact_Stress_Test_Scenarios[scenario]` — note the
   column is named `scenario`, not `scenario_name`. Sort it via `Dim_Scenario[sort_order]`
   (Section 2) if you built that table, so it reads Baseline → Adverse → Severely Adverse rather
   than alphabetically. Add cards for `[Scenario Total EAD]`, `[Scenario Weighted PD]`, and
   `[Scenario Total Expected Loss]` — these already reflect the Baseline/Adverse/Severely Adverse
   shocks, no What-If multiplier needed on top of this table.

3. **What-If sliders:** place the two What-If parameter slicers alongside the scenario slicer so
   a user can further stress the *currently selected* scenario interactively via
   `[Stressed Expected Loss]` (computed over `Fact_Loan_Risk_Portfolio`, not the scenario table).

4. **Waterfall — incremental capital loss by scenario:** add a **Waterfall chart** visual with
   `Fact_Stress_Test_Scenarios[scenario]` (sorted Baseline → Adverse → Severely Adverse) as the
   **Category** and `[Scenario Incremental EL vs Baseline]` as the **Value**. This renders as
   Baseline sitting at $0, then each scenario's incremental Expected Loss stacking on top —
   $2.35B added by the Adverse shock, a further $2.98B by Severely Adverse (on top of Adverse),
   for a $10.68B severely-adverse total. A clustered bar chart with `[Scenario Total Expected
   Loss]` by scenario works too if you'd rather show absolute totals side by side instead of the
   cascading breakdown.

5. **Layout:** scenario slicer + two What-If slider slicers across the top; the waterfall chart
   from step 4 below that; and a card row showing the live `[Stressed Expected Loss]` /
   `[Stress EL Delta]` from the slider-driven measures for interactive what-if exploration on top
   of whichever scenario is selected.

### Optional extension: interest-rate shock slider

An interest-rate What-If parameter (e.g. `Rate Shock (bps)`, range -100 to +300, step 25) can be
added the same way as the PD/LGD parameters above. Be upfront with report viewers about what it
actually does, though: the current stress-test engine (`src/models/stress_test.py`) only shocks
PD, LGD, and EAD — `int_rate` is a **model input feature**, not a lever in the EL formula, so a
rate-shock slider has no wired effect on Expected Loss unless the PD model is retrained with a
macro overlay (e.g. a rate-sensitivity coefficient) or a satellite model is added. Until that
exists, treat a rate slider as illustrative/for future coupling, not a live driver of the
`[Stressed Expected Loss]` measure.

## 6. Refreshing the dashboard

Re-running `python run_pipeline.py --stress-test` regenerates every Parquet file this report
reads. In Power BI Desktop, **Home > Refresh** picks up the new files as long as the file paths
haven't changed. For a scheduled refresh via Power BI Service, publish to a workspace with a
gateway pointed at wherever `data/curated/` lives, or move the Parquet outputs to cloud storage
(OneLake/ADLS/S3) the service can reach directly.
