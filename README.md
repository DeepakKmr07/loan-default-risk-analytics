# Banking Credit Portfolio Default & Expected Loss Intelligence Platform

An end-to-end credit risk intelligence pipeline I built to analyze a large historical consumer
lending portfolio (LendingClub, ~2.26M accepted loans, 2007–2018) and calculate **Expected Loss**
(EL = PD × LGD × EAD) following IFRS 9 / Basel principles.

The pipeline goes from raw loan-level data all the way to a curated star schema and DAX measure
library ready for Power BI, with a calibrated machine-learning core in the middle.

## What it does

- **ETL (DuckDB):** cleans ~2.26M raw loan records and engineers debt-to-income and credit
  utilization features directly with vectorized SQL on Parquet — no full in-memory pandas loads.
- **PD model:** a LightGBM classifier predicting Probability of Default, validated with a strict
  Out-of-Time (OOT) split (never randomly shuffled across vintages) and calibrated with isotonic
  regression so predicted risk matches empirical default rates.
- **LGD model:** a LightGBM regressor trained on realized recoveries from resolved, charged-off
  loans to estimate Loss Given Default.
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
- **Automated test suite:** pytest coverage for ETL data contracts, star-schema referential
  integrity, and the PSI/KS/calibration statistics themselves.
- **Interactive Streamlit dashboard:** a self-serve web app (`app/dashboard.py`) reading straight
  from the curated Parquet marts — executive KPIs, vintage/migration views, and a live macro
  stress simulator with sliders, alongside the enterprise Power BI layer.

## Results (full 2007–2018 accepted-loans portfolio)

| Grade | Total EAD | Weighted PD | Expected Loss |
|---|---:|---:|---:|
| A | $6.32B | 7.37% | $234.7M |
| B | $9.40B | 18.14% | $964.1M |
| C | $9.77B | 28.75% | $1.73B |
| D | $5.10B | 37.21% | $1.25B |
| E | $2.37B | 45.41% | $759.8M |
| F | $0.80B | 54.00% | $321.2M |
| G | $0.25B | 57.00% | $108.3M |
| **Total** | **$34.0B** | **25.07%** | **$5.36B** |

Overall portfolio EL rate: **15.77%**. The PD model scores **0.713 ROC-AUC / 0.389 PR-AUC / 0.1545
Brier score** on the out-of-time test set (loans issued Oct 2016–Dec 2018, held out entirely from
training and calibration). The LGD model scores **0.205 RMSE / 0.172 MAE** on realized recoveries.

## Model governance & stability (`reports/model_risk_summary.json`)

| Metric | Value |
|---|---:|
| Gini coefficient (2×AUC−1) | 0.4262 |
| KS statistic | 0.3097 |
| ROC-AUC (OOT) | 0.7131 |
| Brier score (OOT) | 0.1545 |
| LGD RMSE / MAE (OOT) | 0.2046 / 0.1724 |

The calibration reliability diagram (`reports/calibration_curve.png`) tracks the perfect-
calibration diagonal closely across all 10 deciles. PSI monitoring (`reports/psi_report.csv`,
baseline = training/development sample vs. each post-training vintage) flags **27 moderate-or-
worse shifts**, concentrated in `int_rate` (moderate drift from 2016 onward — LendingClub's
pricing shifted over time) and `revol_util`/`credit_utilization` (significant drift from 2017Q4
onward — a known reporting change in LendingClub's later vintages). This is exactly the kind of
real, explainable population drift a PSI monitor is supposed to catch.

## Stress testing (`data/curated/Fact_Stress_Test_Scenarios.parquet`)

| Scenario | Shock | Weighted PD | Total Expected Loss |
|---|---|---:|---:|
| Baseline | none | 25.07% | $5.36B |
| Adverse | PD ×1.25, LGD ×1.15 | 31.33% | $7.71B |
| Severely Adverse | PD ×1.60, LGD ×1.25, +20% CCF on undrawn exposure | 40.01% | $10.68B |

EAD is unchanged across scenarios: LendingClub loans are fully-disbursed installment loans with
no revolving/undrawn commitment, so the CCF term has no effect on this particular portfolio — the
mechanism is implemented generally in `stress_test.py` for portfolios that do carry revolving
exposure.

## Tech stack

- **Storage / ETL:** DuckDB, PyArrow
- **Modeling:** Python, scikit-learn, LightGBM, XGBoost, Optuna
- **BI:** Star-schema Parquet marts + a full DAX measure library, designed for Power BI
- **Interactive app:** Streamlit + Plotly, reading the same curated Parquet marts directly
- **Testing:** pytest

## Project structure

```
├── data/
│   ├── raw/          # loans_raw.parquet (converted from the LendingClub CSV export)
│   ├── processed/     # clean_loans.parquet, scored_portfolio.parquet
│   └── curated/       # Star schema: Fact_Loan_Risk_Portfolio, Fact_Stress_Test_Scenarios, Dim_*
├── app/
│   └── dashboard.py   # Streamlit + Plotly interactive dashboard (3 tabs, see below)
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
│   │   └── stress_test.py         # Baseline/Adverse/Severely Adverse scenario engine
│   └── bi/
│       └── build_marts.py  # star schema generation
├── dax/
│   └── credit_risk_measures.dax  # EAD, Weighted PD, EL, EL Rate %, Risk Migration, Stress Testing
├── docs/
│   └── POWER_BI_SETUP.md   # table relationships + page-by-page dashboard build guide
├── reports/                 # model_risk_summary.json, psi_report.csv, calibration_curve.png
├── tests/                    # pytest: ETL contracts, star-schema referential integrity, PSI/KS unit tests
├── run_pipeline.py    # orchestrates ETL -> train -> score -> marts (+ --test / --stress-test)
└── DEVELOPMENT.md     # project rules and modeling constraints I set for this build
```

## Running it

1. Get the LendingClub accepted-loans dataset (`accepted_2007_to_2018Q4.csv`) — either via
   `src/data/download.py` (requires a Kaggle API token) or by dropping the CSV into `data/`.
2. Run the full pipeline:

```
python run_pipeline.py                  # ETL -> train PD/LGD -> score -> build marts
python run_pipeline.py --stress-test    # same, plus Baseline/Adverse/Severely Adverse scenarios
python run_pipeline.py --test           # run the pytest suite instead of the pipeline
python run_pipeline.py --dashboard       # launch the Streamlit app (needs the marts built already)
```

The default run prints a summary of Total EAD, Weighted-Average PD, and Total Expected Loss,
broken down by credit grade. Model governance diagnostics are a separate step, since they
evaluate already-trained models rather than producing pipeline outputs:

```
python -m src.models.evaluate_governance
```

### The dashboard

`app/dashboard.py` reads `Fact_Loan_Risk_Portfolio.parquet` and `Fact_Stress_Test_Scenarios.parquet`
straight off disk via DuckDB (no separate data step) and gives three views, filterable by credit
sub-grade and vintage quarter from the sidebar:

1. **Executive Portfolio Health** — Total EAD / Weighted PD / Expected Loss / EL Rate KPI cards,
   an EL-by-grade bar chart, and an interest-rate-vs-PD scatter (sampled to 5,000 loans for
   responsiveness — the full 2.26M-row plot isn't something a browser should render live).
2. **Vintage Analysis & Migration** — a cumulative-default-rate-by-vintage line chart and a
   loan-status-mix-by-grade heatmap. Both are captioned in-app with what they actually represent:
   this dataset is a loan-level snapshot, not a monthly performance panel, so there's no literal
   delinquency roll-rate history to show — see the Design notes below.
3. **Dynamic Macro Stress Simulator** — live sliders for a PD multiplier (1.0×–2.5×) and an
   LGD/collateral haircut (0%–50%) recompute Expected Loss on the fly against the currently
   filtered population, with a before/after capital-impact delta, plus a reference table pulling
   in the three pre-computed CCAR-style scenarios for comparison.

## Design notes

- **No random shuffling across time.** Every train/test split is chronological, based on
  `issue_d`, to avoid look-ahead bias — a model trained on 2018 vintages would leak information
  a real underwriting model would never have.
- **Calibration matters more than raw discrimination for EL.** A PD model with good ROC-AUC but
  poorly calibrated probabilities will produce a systematically wrong Expected Loss, so predicted
  probabilities are calibrated against empirical default rates with isotonic regression.
- **EAD proxy.** Funded amount is used as the Exposure at Default proxy, appropriate for these
  fixed-amortization installment loans (as opposed to revolving lines of credit, which need a
  utilization-based EAD model).
- **Dim_Borrower is 1:1 with loans**, not deduplicated — this dataset has no persistent customer
  identifier across loans, so each row represents one borrower snapshot at application time.
- **PSI baseline = the training/development sample**, not the immediately-preceding vintage —
  that's what a governance review actually wants to know: has the population drifted since the
  model was built, not just since last quarter.
- **No revolving-exposure CCF in practice.** The severely-adverse scenario's credit-conversion-
  factor shock is implemented as a general `EAD + CCF × undrawn_amount` formula, but this dataset
  has no undrawn/revolving exposure (installment loans are fully disbursed at origination), so
  `undrawn_amount` is 0 for every loan and EAD doesn't move under stress here.

See [docs/POWER_BI_SETUP.md](docs/POWER_BI_SETUP.md) for the full dashboard build guide,
including two things I was upfront about rather than overstating: `Dim_Vintage` only supports
quarter-level trending (no true calendar date table yet), and "migration" in this dataset means
vintage-cohort mix shift, not a literal month-by-month delinquency roll-rate matrix (there's no
loan-performance panel here, just one snapshot per loan).
