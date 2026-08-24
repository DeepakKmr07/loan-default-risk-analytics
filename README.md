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

## Rolling-window out-of-time validation (`reports/rolling_backtest_summary.json`)

The headline numbers above come from one OOT cutoff (train through mid-2016, test on loans
issued Oct 2016–Dec 2018). That's the standard split for the deployed model, but it only answers
"did this work on one held-out period" — a credit-risk reviewer's sharper question is whether it
holds up *consistently* across time. `rolling_backtest.py` answers that with a walk-forward
design: the timeline is cut into 6 equal chronological segments, and each of 5 folds trains on
all segments before it and tests on the next one — an expanding window that never touches future
data, with hyperparameters held fixed so this measures stability of one modeling choice, not the
effect of re-tuning per fold.

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
same `int_rate`/`revol_util` drift the PSI monitor below independently flags starting in that
period, which is a good cross-check that both diagnostics are catching the same real phenomenon
rather than noise. See `reports/rolling_backtest_pd.png` for the fold-by-fold plot and
`reports/rolling_backtest_pd.csv` / `rolling_backtest_lgd.csv` for the full per-fold breakdown.

## Model governance & stability (`reports/model_risk_summary.json`)

| Metric | Value |
|---|---:|
| Gini coefficient (2×AUC−1) | 0.4267 |
| KS statistic | 0.3109 |
| ROC-AUC (OOT) | 0.7133 |
| Brier score (OOT) | 0.1544 |
| LGD RMSE / MAE (OOT) | 0.2048 / 0.1725 |

The calibration reliability diagram (`reports/calibration_curve.png`) tracks the perfect-
calibration diagonal closely across all 10 deciles. PSI monitoring (`reports/psi_report.csv`,
baseline = training/development sample vs. each post-training vintage) flags **27 moderate-or-
worse shifts**, concentrated in `int_rate` (moderate drift from 2016 onward — LendingClub's
pricing shifted over time) and `revol_util`/`credit_utilization` (significant drift from 2017Q4
onward — a known reporting change in LendingClub's later vintages). This is exactly the kind of
real, explainable population drift a PSI monitor is supposed to catch.

## Explainability (`reports/loan_reason_codes.parquet`)

SHAP reason codes over a 2,000-loan sample: `sub_grade` is the #1 risk driver for about half the
sample (unsurprising — it's LendingClub's own risk assessment baked into the loan terms),
followed by `term_months` (60-month loans carry more risk than 36-month) and
`acc_open_past_24mths` (recent credit-seeking behavior) — all textbook credit-risk signals, which
is the actual point of checking this rather than assuming a tree model's internals are sane.

## Stress testing (`data/curated/Fact_Stress_Test_Scenarios.parquet`)

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

### Tuning and explainability

Hyperparameter tuning is opt-in, not part of the default pipeline run — a full Optuna search
takes minutes, and the default run should stay fast and reproducible:

```
python -m src.models.train_pd --tune --n-trials 25   # tunes, saves models/pd_best_params.json
python -m src.models.train_lgd --tune --n-trials 25  # tunes, saves models/lgd_best_params.json
```

Once a `*_best_params.json` exists, every subsequent normal run (`train_pd.main()` /
`train_lgd.main()` / `run_pipeline.py`) automatically picks it up and trains with those
hyperparameters instead of the fixed defaults — no flag needed. Re-run with `--tune` any time
to re-search (e.g. after the underlying data changes materially).

Per-loan reason codes (SHAP, on a representative sample rather than the full portfolio — see
the module docstring for why) are computed separately from training:

```
python -m src.models.explain
```

The rolling-window backtest is also a separate step — it retrains fresh models per fold, so it's
not part of the default fast pipeline run either:

```
python -m src.models.rolling_backtest --folds 5   # 5 is the default; adjust for smaller datasets
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
