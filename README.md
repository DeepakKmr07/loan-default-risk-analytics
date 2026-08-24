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

## Tech stack

- **Storage / ETL:** DuckDB, PyArrow
- **Modeling:** Python, scikit-learn, LightGBM, XGBoost, Optuna
- **BI:** Star-schema Parquet marts + a full DAX measure library, designed for Power BI

## Project structure

```
├── data/
│   ├── raw/          # loans_raw.parquet (converted from the LendingClub CSV export)
│   ├── processed/     # clean_loans.parquet, scored_portfolio.parquet
│   └── curated/       # Star schema: Fact_Loan_Risk_Portfolio + Dim_* tables
├── src/
│   ├── data/
│   │   ├── download.py     # Kaggle -> Parquet ingestion
│   │   └── etl_duckdb.py   # cleaning, feature engineering, labeling
│   ├── models/
│   │   ├── features.py            # shared feature lists (train/inference stay in sync)
│   │   ├── categorical_utils.py   # consistent categorical encoding across train & inference
│   │   ├── train_pd.py            # calibrated LightGBM PD model, OOT validation
│   │   ├── train_lgd.py           # LightGBM LGD regression model
│   │   └── inference.py           # portfolio-wide PD/LGD/EAD/EL scoring
│   └── bi/
│       └── build_marts.py  # star schema generation
├── dax/
│   └── credit_risk_measures.dax  # EAD, Weighted PD, EL, EL Rate %, Risk Migration, Stress Testing
├── run_pipeline.py    # orchestrates the full ETL -> train -> score -> marts workflow
└── DEVELOPMENT.md     # project rules and modeling constraints I set for this build
```

## Running it

1. Get the LendingClub accepted-loans dataset (`accepted_2007_to_2018Q4.csv`) — either via
   `src/data/download.py` (requires a Kaggle API token) or by dropping the CSV into `data/`.
2. Run the full pipeline:

```
python run_pipeline.py
```

This runs ETL → PD/LGD training → portfolio scoring → star-schema mart generation, and prints a
summary of Total EAD, Weighted-Average PD, and Total Expected Loss, broken down by credit grade.

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
