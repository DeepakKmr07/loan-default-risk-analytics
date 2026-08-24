# Banking Credit Portfolio Default & Expected Loss Intelligence Platform

[![Tests](https://github.com/DeepakKmr07/loan-default-risk-analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/DeepakKmr07/loan-default-risk-analytics/actions/workflows/tests.yml)

An end-to-end credit risk intelligence pipeline analyzing a large historical consumer lending
portfolio (LendingClub, ~2.26M accepted loans, 2007–2018) and calculating **Expected Loss**
(EL = PD × LGD × EAD) in line with Basel / IFRS 9 expected-loss principles.

The pipeline runs raw loan-level data through a calibrated machine-learning core, out to a
curated star schema and DAX measure library ready for Power BI, plus a self-serve Streamlit app.

## Table of contents

1. [Architecture](#architecture)
2. [Financial risk methodology](#financial-risk-methodology)
3. [Key features](#key-features)
4. [Results](#results-full-20072018-accepted-loans-portfolio)
5. [Tech stack](#tech-stack)
6. [Setup & execution guide](#setup--execution-guide)
7. [Power BI integration guide](#power-bi-integration-guide)
8. [Model governance & diagnostics](#model-governance--diagnostics)
9. [Hyperparameter tuning](#hyperparameter-tuning---tune-via-optuna)
10. [Explainability](#explainability-reportsloan_reason_codesparquet)
11. [Project structure](#project-structure)
12. [Design notes & known limitations](#design-notes--known-limitations)

## Architecture

```
┌────────────────────────┐
│      Raw Loan Data      │
│  LendingClub CSV/Parquet │
│   (~2.26M loans, 2007–2018)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│         DuckDB ETL Layer               │
│      src/data/etl_duckdb.py            │
│  vectorized SQL on Parquet — no full   │
│  in-memory pandas load of 2.26M rows   │
│  • cleaning & type coercion            │
│  • DTI / utilization feature engineering│
│  • default_flag / actual_lgd labeling  │
└────────────┬────────────────────────────┘
             │ data/processed/clean_loans.parquet
             ▼
┌─────────────────────────────────────────────────────┐
│                  ML Modeling Core                      │
│                                                         │
│   ┌────────────────────┐     ┌───────────────────────┐ │
│   │      PD Model        │     │       LGD Model        │ │
│   │  LightGBM Classifier │     │  LightGBM Regressor     │ │
│   │  + Isotonic          │     │  on realized recoveries │ │
│   │    Calibration        │     │  from resolved defaults │ │
│   │  (Optuna-tunable)     │     │  (Optuna-tunable)        │ │
│   └──────────┬───────────┘     └────────────┬─────────────┘ │
│              └───────────────┬───────────────┘               │
│                               ▼                               │
│                   src/models/inference.py                     │
│              EL = PD × LGD × EAD, scored per loan              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ data/processed/scored_portfolio.parquet
                               ▼
┌────────────────────────────────────────────────────────┐
│           Star Schema Marts (src/bi/build_marts.py)       │
│                                                            │
│   Fact_Loan_Risk_Portfolio    Fact_Stress_Test_Scenarios    │
│         │        │       │            (Baseline/Adverse/    │
│         ▼        ▼       ▼             Severely Adverse)     │
│  Dim_Borrower  Dim_Vintage  Dim_Credit_Grade                │
└──────────────┬─────────────────────────────┬─────────────────┘
               │                             │
               ▼                             ▼
   ┌─────────────────────────┐   ┌────────────────────────────┐
   │  Streamlit + Plotly        │   │      Power BI Desktop        │
   │  app/dashboard.py           │   │  + dax/credit_risk_measures.dax│
   │  self-serve web app,        │   │  star-schema model,           │
   │  live macro stress sliders  │   │  enterprise reporting layer   │
   └─────────────────────────┘   └────────────────────────────┘
```

Both consumption layers (Streamlit and Power BI) read the *same* curated Parquet marts — there
is no divergent business logic between the two, only presentation.

## Financial risk methodology

### Expected Loss: EL = PD × LGD × EAD

The core calculation follows the Basel II/III internal-ratings-based (IRB) capital framework and
IFRS 9's expected-credit-loss (ECL) accounting model, both of which decompose loss on a credit
exposure into three independently-estimated components:

| Component | Definition | This project's estimator |
|---|---|---|
| **PD** — Probability of Default | Likelihood the borrower defaults over the relevant horizon | Calibrated LightGBM classifier (`src/models/train_pd.py`) |
| **LGD** — Loss Given Default | Share of exposure not recovered after a default, net of recoveries | LightGBM regressor trained on realized post-default recoveries (`src/models/train_lgd.py`) |
| **EAD** — Exposure at Default | Outstanding exposure amount at the point of default | Funded loan amount (appropriate for fully-disbursed installment loans; see [Design notes](#design-notes--known-limitations)) |

`EL = PD × LGD × EAD` is then computed **per loan** and aggregated across the portfolio,
by credit grade, and by macroeconomic scenario. One honest simplification worth stating plainly:
this project produces a single lifetime EL per loan rather than a full IFRS 9 three-stage
model (12-month ECL for Stage 1 vs. lifetime ECL for Stage 2/3 credit-impaired assets) — it's
built as a portfolio-level risk-intelligence and stress-testing tool, not a full accounting
provisioning engine.

### Why probability calibration matters here

A model can have excellent *discrimination* (it ranks risky loans above safe ones) while still
being *poorly calibrated* (its raw scores don't match true probabilities). That distinction is
invisible to ROC-AUC/Gini/KS, but it matters enormously for `EL = PD × LGD × EAD`: if predicted
PD is systematically too high or too low, the dollar-denominated Expected Loss is wrong even
when the risk *ranking* is fine. So predicted probabilities are explicitly recalibrated with
isotonic regression (`CalibratedClassifierCV`) after training, and checked with:

- **Brier score** — mean squared error between predicted probability and the realized binary
  outcome; lower is better, and it (unlike ROC-AUC) actually penalizes miscalibration.
- **Reliability curve / calibration diagram** — predictions are bucketed into deciles and mean
  predicted PD is plotted against the empirical default rate in that decile; a perfectly
  calibrated model sits on the 45° diagonal (`reports/calibration_curve.png`).

### Macroeconomic stress testing

`src/models/stress_test.py` implements CCAR/IFRS9-style scenario shocks — a transparent,
multiplicative approach (rather than a full macro-econometric satellite model), applying PD and
LGD multipliers and a credit-conversion-factor (CCF) shock to undrawn exposure per scenario:

| Scenario | PD multiplier | LGD multiplier | EAD shock |
|---|---:|---:|---|
| Baseline | ×1.00 | ×1.00 | none |
| Adverse | ×1.25 | ×1.15 | none |
| Severely Adverse | ×1.60 | ×1.25 | +20% CCF on undrawn exposure |

This recomputes portfolio-level Expected Loss under each scenario without retraining the
underlying models — see [Results](#stress-testing-datacuratedfact_stress_test_scenariosparquet)
below for actual output.

## Key features

- **ETL (DuckDB):** cleans ~2.26M raw loan records and engineers debt-to-income and credit
  utilization features directly with vectorized SQL on Parquet — no full in-memory pandas loads.
- **PD model:** a LightGBM classifier predicting Probability of Default, validated with a strict
  Out-of-Time (OOT) split (never randomly shuffled across vintages) and calibrated with isotonic
  regression so predicted risk matches empirical default rates. Hyperparameters are tunable via
  Optuna (`--tune`), with the best-found configuration persisted and reused automatically.
- **Rolling-window backtesting** (`src/models/rolling_backtest.py`): a single OOT cutoff only
  proves a model worked on one held-out period. This walks the whole timeline forward across
  multiple expanding train/test windows to check whether performance actually holds up over time.
- **LGD model:** a LightGBM regressor trained on realized recoveries from resolved, charged-off
  loans to estimate Loss Given Default, tunable the same way.
- **Explainability:** per-loan SHAP reason codes on the PD model (`src/models/explain.py`) —
  the top risk-driving features per loan, in the direction adverse-action-style credit
  explanations need.
- **Portfolio scoring:** every loan in the portfolio (including still-active accounts) is scored
  with PD, LGD, EAD, and Expected Loss.
- **Star schema marts:** a curated `Fact_Loan_Risk_Portfolio` table plus `Dim_Borrower`,
  `Dim_Vintage`, and `Dim_Credit_Grade` dimensions, exported as Parquet for Power BI.
- **DAX measure library:** production-ready measures for Total EAD, Weighted PD, Total Expected
  Loss, EL Rate %, vintage-over-vintage Risk Migration, and What-If stress-testing scenarios.
- **Model governance:** a calibration reliability diagram, Gini/KS/ROC-AUC discrimination
  metrics, and Population Stability Index (PSI) monitoring for both predicted PD and key credit
  features across origination vintages — the diagnostics Basel/IFRS 9 model-risk review expects.
- **Macroeconomic stress testing:** Baseline / Adverse / Severely Adverse scenario simulation
  (CCAR-style PD/LGD/EAD shocks) exported as its own Parquet fact table for Power BI.
- **Automated test suite + CI:** pytest coverage for ETL data contracts, star-schema referential
  integrity, the PSI/KS/calibration statistics, and the walk-forward backtest splitter itself,
  run automatically on every push via GitHub Actions against a synthetic dataset (so
  `test_etl.py`/`test_star_schema.py` actually execute in CI instead of skipping for lack of the
  real, uncommitted multi-GB data, and the rolling backtest runs end-to-end too).
- **Interactive Streamlit dashboard:** a self-serve web app (`app/dashboard.py`) reading straight
  from the curated Parquet marts — executive KPIs, vintage/migration views, and a live macro
  stress simulator with sliders, alongside the enterprise Power BI layer.

## Results (full 2007–2018 accepted-loans portfolio)

| Grade | Total EAD | Weighted PD | Expected Loss |
|---|---:|---:|---:|
| A | $6.32B | 7.20% | $229.3M |
| B | $9.40B | 17.92% | $953.8M |
| C | $9.77B | 28.90% | $1.73B |
| D | $5.10B | 37.52% | $1.26B |
| E | $2.37B | 46.03% | $770.0M |
| F | $0.80B | 53.87% | $320.3M |
| G | $0.25B | 56.89% | $108.1M |
| **Total** | **$34.0B** | **25.11%** | **$5.38B** |

Overall portfolio EL rate: **15.81%**. The PD model scores **0.713 ROC-AUC / 0.390 PR-AUC / 0.1544
Brier score** on the out-of-time test set (loans issued Oct 2016–Dec 2018, held out entirely from
training and calibration). The LGD model scores **0.205 RMSE / 0.173 MAE** on realized recoveries.

### Stress testing (`data/curated/Fact_Stress_Test_Scenarios.parquet`)

| Scenario | Shock | Weighted PD | Total Expected Loss |
|---|---|---:|---:|
| Baseline | none | 25.11% | $5.38B |
| Adverse | PD ×1.25, LGD ×1.15 | 31.38% | $7.73B |
| Severely Adverse | PD ×1.60, LGD ×1.25, +20% CCF on undrawn exposure | 40.05% | $10.71B |

EAD is unchanged across scenarios: LendingClub loans are fully-disbursed installment loans with
no revolving/undrawn commitment, so the CCF term has no effect on this particular portfolio — the
mechanism is implemented generally in `stress_test.py` for portfolios that do carry revolving
exposure.

## Tech stack

- **Storage / ETL:** DuckDB, PyArrow
- **Modeling:** Python, scikit-learn, LightGBM, XGBoost, Optuna (hyperparameter tuning), SHAP (explainability)
- **BI:** Star-schema Parquet marts + a full DAX measure library, designed for Power BI
- **Interactive app:** Streamlit + Plotly, reading the same curated Parquet marts directly
- **Testing / CI:** pytest, GitHub Actions

## Setup & execution guide

### 1. Get the data

Get the LendingClub accepted-loans dataset (`accepted_2007_to_2018Q4.csv`) — either via
`src/data/download.py` (requires a Kaggle API token) or by dropping the CSV into `data/` yourself.

### 2. Run the full pipeline

```bash
python run_pipeline.py                  # ETL -> train PD/LGD -> score -> build marts
```

This runs ETL, trains and calibrates both models, scores the full portfolio, and builds the
star-schema marts — printing a Total EAD / Weighted PD / Expected Loss summary by grade at the
end (the table in [Results](#results-full-20072018-accepted-loans-portfolio) above).

```bash
python run_pipeline.py --stress-test    # same, plus Baseline/Adverse/Severely Adverse scenarios
python run_pipeline.py --test           # run the pytest suite instead of the pipeline
python run_pipeline.py --dashboard      # launch the Streamlit app (needs the marts built already)
```

### 3. Model governance diagnostics

A separate step, since it evaluates already-trained models rather than producing pipeline outputs:

```bash
python -m src.models.evaluate_governance
```

Writes `reports/calibration_curve.png`, `reports/psi_report.csv`, and
`reports/model_risk_summary.json` — see [Model governance & diagnostics](#model-governance--diagnostics).

### 4. Hyperparameter tuning (optional, opt-in)

A full Optuna search takes minutes, so it's never part of the default fast pipeline run:

```bash
python -m src.models.train_pd --tune --n-trials 25   # tunes, saves models/pd_best_params.json
python -m src.models.train_lgd --tune --n-trials 25  # tunes, saves models/lgd_best_params.json
```

Once a `*_best_params.json` exists, every subsequent normal run (`train_pd.main()` /
`train_lgd.main()` / `run_pipeline.py`) automatically picks it up and trains with those
hyperparameters instead of the fixed defaults — no flag needed. Re-run with `--tune` any time
to re-search (e.g. after the underlying data changes materially).

### 5. Explainability and rolling-window backtest (optional, opt-in)

```bash
python -m src.models.explain                       # SHAP reason codes on a representative sample
python -m src.models.rolling_backtest --folds 5     # walk-forward multi-window OOT stability check
```

Both are separate steps from the default pipeline: `explain.py` runs SHAP over a sample rather
than the full 2.26M-row portfolio, and `rolling_backtest.py` retrains fresh models per fold.

### 6. The dashboard

```bash
python run_pipeline.py --dashboard
```

`app/dashboard.py` reads `Fact_Loan_Risk_Portfolio.parquet` and `Fact_Stress_Test_Scenarios.parquet`
straight off disk via DuckDB (no separate data step) and gives three views, filterable by credit
sub-grade and vintage quarter from the sidebar:

1. **Executive Portfolio Health** — Total EAD / Weighted PD / Expected Loss / EL Rate KPI cards,
   an EL-by-grade bar chart, and an interest-rate-vs-PD scatter (sampled to 5,000 loans for
   responsiveness — the full 2.26M-row plot isn't something a browser should render live).
2. **Vintage Analysis & Migration** — a cumulative-default-rate-by-vintage line chart and a
   loan-status-mix-by-grade heatmap. Both are captioned in-app with what they actually represent:
   this dataset is a loan-level snapshot, not a monthly performance panel, so there's no literal
   delinquency roll-rate history to show.
3. **Dynamic Macro Stress Simulator** — live sliders for a PD multiplier (1.0×–2.5×) and an
   LGD/collateral haircut (0%–50%) recompute Expected Loss on the fly against the currently
   filtered population, with a before/after capital-impact delta, plus a reference table pulling
   in the three pre-computed CCAR-style scenarios for comparison.

## Power BI integration guide

The full page-by-page dashboard build guide lives in
[docs/POWER_BI_SETUP.md](docs/POWER_BI_SETUP.md); this section summarizes the data model and
core measures.

### Star schema data model

Both fact tables share the same three conformed dimensions (a fact constellation), so each
dimension is loaded once and related to both facts:

| From (fact) | Column | To (dimension) | Column | Cardinality |
|---|---|---|---|---|
| `Fact_Loan_Risk_Portfolio` | `borrower_key` | `Dim_Borrower` | `borrower_key` | 1:many |
| `Fact_Loan_Risk_Portfolio` | `vintage_key` | `Dim_Vintage` | `vintage_key` | 1:many |
| `Fact_Loan_Risk_Portfolio` | `credit_grade_key` | `Dim_Credit_Grade` | `credit_grade_key` | 1:many |
| `Fact_Stress_Test_Scenarios` | `borrower_key` / `vintage_key` / `credit_grade_key` | *(same dimensions)* | | 1:many |

`Fact_Stress_Test_Scenarios` also carries a `scenario` text column (`Baseline` / `Adverse` /
`Severely Adverse`); the setup guide shows how to add a small `Dim_Scenario` table so it sorts
by severity instead of alphabetically.

Three report pages are documented in detail in the setup guide: **Executive Summary** (KPI cards,
EL-by-grade, interest-rate-vs-PD scatter), **Credit Migration & Vintage Curves** (grade mix-shift
heatmap, vintage curves, PD calibration gap), and **Stress Testing & What-If Planner** (scenario
slicer, live PD/LGD What-If sliders, waterfall chart of incremental EL by scenario).

### Core DAX measures

```dax
// EL = PD x LGD x EAD (IFRS 9 / Basel expected-loss formula), aggregated at loan grain.
Total Expected Loss =
SUMX (
    Fact_Loan_Risk_Portfolio,
    Fact_Loan_Risk_Portfolio[pd] * Fact_Loan_Risk_Portfolio[lgd] * Fact_Loan_Risk_Portfolio[ead]
)

// EAD-weighted average PD — the correct portfolio-level PD, not a simple row average.
Weighted PD =
DIVIDE (
    SUMX ( Fact_Loan_Risk_Portfolio, Fact_Loan_Risk_Portfolio[pd] * Fact_Loan_Risk_Portfolio[ead] ),
    [Total EAD]
)

Total EAD = SUM ( Fact_Loan_Risk_Portfolio[ead] )

EL Rate % = DIVIDE ( [Total Expected Loss], [Total EAD] )

// Calibration check: predicted portfolio PD vs. observed default rate on resolved loans.
PD Calibration Gap = [Weighted PD] - [Realized Default Rate]

// Percentage-point shift in a grade's share of originations, vintage over vintage —
// positive = credit box loosening toward this grade, negative = tightening away from it.
Grade Mix Shift (pp) = ( [Grade Mix %] - [Grade Mix % (Prior Vintage)] ) * 100

// Live What-If stress measure, driven by two Power BI numeric-range parameters
// ('PD Stress Multiplier', 'LGD Stress Multiplier').
Stressed Expected Loss =
SUMX (
    Fact_Loan_Risk_Portfolio,
    MIN ( Fact_Loan_Risk_Portfolio[pd] * SELECTEDVALUE ( 'PD Stress Multiplier'[PD Stress Multiplier Value], 1.0 ), 1.0 )
        * MIN ( Fact_Loan_Risk_Portfolio[lgd] * SELECTEDVALUE ( 'LGD Stress Multiplier'[LGD Stress Multiplier Value], 1.0 ), 1.0 )
        * Fact_Loan_Risk_Portfolio[ead]
)
```

The full library (`dax/credit_risk_measures.dax`) also includes realized-vs-expected validation
measures, vintage-over-vintage Risk Migration, and pre-computed scenario measures for the
`Fact_Stress_Test_Scenarios` table — 20+ measures in total, each with an inline comment
explaining the business logic.

## Model governance & diagnostics

`reports/model_risk_summary.json`, generated by `src/models/evaluate_governance.py`:

| Metric | Value | What it measures |
|---|---:|---|
| Gini coefficient (2×AUC−1) | 0.4267 | Discriminatory power, rescaled from ROC-AUC to the [0, 1] range conventionally used in credit scoring (0 = random, 1 = perfect separation) |
| KS statistic | 0.3109 | Maximum separation between the cumulative score distributions of defaulters vs. non-defaulters — a standard credit-scoring discrimination metric alongside Gini |
| ROC-AUC (OOT) | 0.7133 | Probability the model ranks a random defaulter above a random non-defaulter |
| Brier score (OOT) | 0.1544 | Mean squared error between predicted PD and realized outcome — penalizes miscalibration, not just ranking |
| LGD RMSE / MAE (OOT) | 0.2048 / 0.1725 | LGD regression error on realized recoveries |

The calibration reliability diagram (`reports/calibration_curve.png`) tracks the perfect-
calibration diagonal closely across all 10 deciles.

### Population Stability Index (PSI)

`compute_psi()` (`src/models/governance_utils.py`) buckets a baseline (development-sample)
distribution into deciles and compares a monitoring-period population against those same bucket
edges: `PSI = Σ (actual_pct − expected_pct) × ln(actual_pct / expected_pct)`, using the
conventional credit-risk thresholds:

| PSI | Classification |
|---:|---|
| < 0.10 | Stable |
| 0.10 – 0.25 | Moderate shift |
| > 0.25 | Significant shift |

PSI monitoring (`reports/psi_report.csv`, baseline = training/development sample vs. each
post-training vintage) flags **27 moderate-or-worse shifts**, concentrated in `int_rate`
(moderate drift from 2016 onward — LendingClub's pricing shifted over time) and
`revol_util`/`credit_utilization` (significant drift from 2017Q4 onward — a known reporting
change in LendingClub's later vintages). This is exactly the kind of real, explainable
population drift a PSI monitor is supposed to catch.

### Rolling-window (walk-forward) out-of-time validation

The headline OOT numbers above come from one cutoff (train through mid-2016, test on loans
issued Oct 2016–Dec 2018). That only proves the model worked on *one* held-out period — a
sharper question is whether it holds up *consistently* across time. `rolling_backtest.py`
answers that with a walk-forward design: the timeline is cut into 6 equal chronological
segments, and each of 5 folds trains on all segments before it and tests on the next one — an
expanding window that never touches future data, with hyperparameters held fixed so this
measures stability of one modeling choice, not the effect of re-tuning per fold.

| Fold | Test window | ROC-AUC | Brier | LGD RMSE |
|---|---|---:|---:|---:|
| 1 | 2013-12 – 2014-12 | 0.7159 | 0.1371 | 0.2030 |
| 2 | 2014-12 – 2015-08 | 0.7383 | 0.1426 | 0.1980 |
| 3 | 2015-08 – 2016-03 | 0.7501 | 0.1389 | 0.1961 |
| 4 | 2016-03 – 2017-01 | 0.7161 | 0.1672 | 0.1788 |
| 5 | 2017-01 – 2018-12 | 0.7231 | 0.1500 | 0.2064 |

PD ROC-AUC holds in a **0.716–0.750 band (mean 0.729, std 0.015)** across five different
out-of-time windows spanning 2013–2018 — the single-cutoff number isn't a lucky draw. Fold 4
(2016-03–2017-01) is the one soft spot, with Brier score jumping to 0.167: this lines up with the
same `int_rate`/`revol_util` drift the PSI monitor independently flags starting in that period, a
good cross-check that both diagnostics are catching the same real phenomenon rather than noise.
See `reports/rolling_backtest_pd.png` for the fold-by-fold plot and `reports/rolling_backtest_pd.csv`
/ `rolling_backtest_lgd.csv` for the full per-fold breakdown.

## Hyperparameter tuning (`--tune`, via Optuna)

Tuning is opt-in and the result is genuinely mixed — worth stating plainly rather than only
reporting the win:

- **PD:** a 25-trial search found a configuration that generalizes marginally better on the true
  OOT test set (ROC-AUC 0.7131→0.7134, Brier 0.1545→0.1544). Small, but real — kept and persisted
  to `models/pd_best_params.json`.
- **LGD:** a 25-trial search found a configuration with a *better* internal tuning-validation
  score, but it scored *worse* on the true OOT test (RMSE 0.2046→0.2061) — the search overfit its
  own validation slice. I reverted to the original hand-picked hyperparameters rather than ship a
  model that looked better in tuning and measurably wasn't. No `models/lgd_best_params.json` is
  committed, on purpose — its absence is the record of that decision, not an oversight.

This is the actual value of wiring up a real tuning loop instead of just listing Optuna in a tech
stack: it only helps if you check the *right* metric before trusting it.

## Explainability (`reports/loan_reason_codes.parquet`)

SHAP reason codes over a 2,000-loan sample: `sub_grade` is the #1 risk driver for about half the
sample (unsurprising — it's LendingClub's own risk assessment baked into the loan terms),
followed by `term_months` (60-month loans carry more risk than 36-month) and
`acc_open_past_24mths` (recent credit-seeking behavior) — all textbook credit-risk signals, which
is the actual point of checking this rather than assuming a tree model's internals are sane.

## Project structure

```
├── data/
│   ├── raw/          # loans_raw.parquet (converted from the LendingClub CSV export)
│   ├── processed/     # clean_loans.parquet, scored_portfolio.parquet
│   └── curated/       # Star schema: Fact_Loan_Risk_Portfolio, Fact_Stress_Test_Scenarios, Dim_*
├── app/
│   └── dashboard.py   # Streamlit + Plotly interactive dashboard (3 tabs, see above)
├── src/
│   ├── data/
│   │   ├── download.py     # Kaggle -> Parquet ingestion
│   │   └── etl_duckdb.py   # cleaning, feature engineering, labeling
│   ├── models/
│   │   ├── features.py            # shared feature lists (train/inference stay in sync)
│   │   ├── categorical_utils.py   # consistent categorical encoding across train & inference
│   │   ├── train_pd.py            # calibrated LightGBM PD model, OOT validation
│   │   ├── train_lgd.py           # LightGBM LGD regression model
│   │   ├── inference.py           # portfolio-wide PD/LGD/EAD/EL scoring
│   │   ├── governance_utils.py    # PSI, KS-statistic, calibration reliability curve
│   │   ├── evaluate_governance.py # calibration diagnostics + PSI + model_risk_summary.json
│   │   ├── stress_test.py         # Baseline/Adverse/Severely Adverse scenario engine
│   │   ├── tuning.py              # Optuna search for both models (opt-in via --tune)
│   │   ├── explain.py             # SHAP per-loan reason codes for the PD model
│   │   └── rolling_backtest.py    # walk-forward multi-window OOT stability backtest
│   └── bi/
│       └── build_marts.py  # star schema generation
├── dax/
│   └── credit_risk_measures.dax  # EAD, Weighted PD, EL, EL Rate %, Risk Migration, Stress Testing
├── docs/
│   └── POWER_BI_SETUP.md   # table relationships + page-by-page dashboard build guide
├── reports/                 # model_risk_summary.json, psi_report.csv, calibration_curve.png, loan_reason_codes.parquet,
│                             # rolling_backtest_summary.json, rolling_backtest_pd.csv/.png, rolling_backtest_lgd.csv
├── tests/
│   ├── fixtures/make_synthetic_raw.py  # synthetic dataset generator, used locally and in CI
│   └── test_*.py             # ETL contracts, star-schema referential integrity, PSI/KS unit tests
├── .github/workflows/tests.yml  # CI: generates synthetic data, runs the pipeline, runs pytest
├── run_pipeline.py    # orchestrates ETL -> train -> score -> marts (+ --test / --stress-test / --dashboard)
└── DEVELOPMENT.md     # project rules and modeling constraints set for this build
```

## Design notes & known limitations

- **No random shuffling across time.** Every train/test split is chronological, based on
  `issue_d`, to avoid look-ahead bias — a model trained on 2018 vintages would leak information
  a real underwriting model would never have.
- **Calibration matters more than raw discrimination for EL.** A PD model with good ROC-AUC but
  poorly calibrated probabilities will produce a systematically wrong Expected Loss, so predicted
  probabilities are calibrated against empirical default rates with isotonic regression.
- **EAD proxy.** Funded amount is used as the Exposure at Default proxy, appropriate for these
  fixed-amortization installment loans (as opposed to revolving lines of credit, which need a
  utilization-based EAD model).
- **Single lifetime EL, not full IFRS 9 staging.** This project computes one lifetime EL per
  loan rather than formally bucketing exposures into IFRS 9's Stage 1 (12-month ECL) / Stage 2 /
  Stage 3 (lifetime ECL) — appropriate for a portfolio risk-intelligence tool, not a full
  accounting provisioning engine.
- **Dim_Borrower is 1:1 with loans**, not deduplicated — this dataset has no persistent customer
  identifier across loans, so each row represents one borrower snapshot at application time.
- **PSI baseline = the training/development sample**, not the immediately-preceding vintage —
  that's what a governance review actually wants to know: has the population drifted since the
  model was built, not just since last quarter.
- **No revolving-exposure CCF in practice.** The severely-adverse scenario's credit-conversion-
  factor shock is implemented as a general `EAD + CCF × undrawn_amount` formula, but this dataset
  has no undrawn/revolving exposure (installment loans are fully disbursed at origination), so
  `undrawn_amount` is 0 for every loan and EAD doesn't move under stress here.
- **No true calendar date table yet.** `Dim_Vintage` is grain-limited to `issue_year`/`issue_quarter`,
  which supports quarter-over-quarter trending but not Power BI's built-in time-intelligence
  functions (`DATEADD`, `SAMEPERIODLASTYEAR`) — see [docs/POWER_BI_SETUP.md](docs/POWER_BI_SETUP.md)
  for how to extend this if month-over-month trending is required.
- **"Migration" means vintage-cohort mix shift, not a literal delinquency transition matrix.**
  This dataset is a loan-level snapshot, not a monthly loan-performance panel, so there's no
  granular roll-rate history (e.g. "% of loans that moved from 30dpd to 60dpd") to build a true
  transition matrix from — the Risk Migration measures are the standard proxy used when that
  history isn't available.
