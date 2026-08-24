# Project: Banking Credit Portfolio Default & Expected Loss Intelligence Platform

## Project Overview & Domain Objectives
Build an end-to-end credit risk intelligence pipeline analyzing lending portfolios.
- Objective: Calculate Expected Loss (EL = PD * LGD * EAD) under IFRS 9 / Basel principles.
- Dataset: LendingClub historical loan dataset (accepted_2007_to_2018Q4.csv / raw Parquet).
- Deliverables: DuckDB ETL pipeline, ML modeling engine (PD + LGD), curated Star Schema data marts (Parquet), and DAX calculation model.

## Environment Setup
This machine has Windows Smart App Control enabled, which blocks native DLLs for packages
installed into a project-local `.venv` or conda env (reputation-based, not source-based —
conda-forge builds are blocked identically to PyPI wheels). Do **not** create a `.venv` or
conda environment for this project. Instead use the system interpreter directly with
`--user`-scoped installs:
```
"C:\Program Files\Python313\python.exe" -m pip install --user <package>
"C:\Program Files\Python313\python.exe" script.py
```

## Tech Stack & Tooling
- Storage/ETL: DuckDB (columnar, vectorized), PyArrow
- ML Stack: Python 3.10+, Scikit-Learn, LightGBM / XGBoost, Optuna (tuning)
- BI/Reporting: Star Schema Parquet outputs compatible with Power BI, full DAX measure scripts.

## Project Structure

```
├── data/
│   ├── raw/          # Raw downloaded CSV/Parquet
│   ├── processed/    # Cleaned DuckDB tables / intermediate Parquet
│   └── curated/       # Final Star Schema (Fact_Loans, Dim_*)
├── notebooks/         # Exploratory analysis
├── src/
│   ├── data/
│   │   ├── download.py     # Automated Kaggle ingestion
│   │   └── etl_duckdb.py   # DuckDB cleaning, feature engineering, and labeling
│   ├── models/
│   │   ├── train_pd.py     # Probability of Default (LightGBM/XGBoost + calibration)
│   │   ├── train_lgd.py    # Loss Given Default (Two-stage or Regressor)
│   │   └── inference.py    # Portfolio scoring (PD, LGD, EAD -> EL)
│   └── bi/
│       └── build_marts.py  # Generates Star Schema tables for Power BI
├── dax/
│   └── credit_risk_measures.dax  # Documented DAX business logic
├── tests/
└── run_pipeline.py    # Master orchestration script
```

## Implementation Rules & Constraints
1. **Data Leakage & Target Definition:**
   - Target `default_flag`: 1 if `loan_status` in ('Charged Off', 'Default', 'Does not meet credit policy: Status Charged Off'), 0 if 'Fully Paid'. Exclude active/current loans from training.
   - Use Out-of-Time (OOT) validation based on `issue_d` (vintage quarters). Never do random shuffle splitting across time.
2. **DuckDB Optimization:**
   - Always prefer SQL transformations directly on columnar Parquet streams. Avoid converting full tables to in-memory Pandas dataframes unless required for model fitting.
3. **Probability Calibration:**
   - PD model predictions must be calibrated (e.g., `CalibratedClassifierCV` / Isotonic Regression) so predicted risk matches empirical default rates per rating bucket.
4. **Code Quality:**
   - Use type hints, modular functions, structured logging, and docstrings.
   - Place reproducible seeds (`random_state=42`) across all modeling modules.
