"""Build a Star Schema from the scored portfolio for Power BI consumption.

Note on Dim_Borrower: the LendingClub dataset has no persistent customer identifier —
each row is an independent loan application, not a repeat-customer history. Dim_Borrower
is therefore keyed 1:1 by loan id (there is no natural grain to deduplicate borrowers on),
capturing the borrower-profile attributes captured at application time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORED_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "processed" / "scored_portfolio.parquet"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"

DIM_VINTAGE_SQL = """
CREATE OR REPLACE TABLE dim_vintage AS
SELECT
    ROW_NUMBER() OVER (ORDER BY issue_year, issue_quarter) AS vintage_key,
    issue_year,
    issue_quarter,
    vintage
FROM (SELECT DISTINCT issue_year, issue_quarter, vintage FROM scored_portfolio)
"""

DIM_CREDIT_GRADE_SQL = """
CREATE OR REPLACE TABLE dim_credit_grade AS
SELECT
    ROW_NUMBER() OVER (ORDER BY grade, sub_grade) AS credit_grade_key,
    grade,
    sub_grade
FROM (SELECT DISTINCT grade, sub_grade FROM scored_portfolio)
"""

DIM_BORROWER_SQL = """
CREATE OR REPLACE TABLE dim_borrower AS
SELECT
    id AS borrower_key,
    home_ownership,
    verification_status,
    addr_state,
    purpose,
    application_type,
    emp_length_years,
    annual_inc,
    dti
FROM scored_portfolio
"""

FACT_LOAN_RISK_PORTFOLIO_SQL = """
CREATE OR REPLACE TABLE fact_loan_risk_portfolio AS
SELECT
    s.id AS loan_id,
    s.id AS borrower_key,
    v.vintage_key,
    g.credit_grade_key,
    s.loan_amnt,
    s.funded_amnt,
    s.term_months,
    s.int_rate,
    s.installment,
    s.ead,
    s.pd,
    s.lgd,
    s.expected_loss,
    s.default_flag,
    s.actual_lgd,
    s.loan_status
FROM scored_portfolio s
JOIN dim_vintage v USING (issue_year, issue_quarter, vintage)
JOIN dim_credit_grade g USING (grade, sub_grade)
"""


def build_marts(
    scored_path: Path = SCORED_PORTFOLIO_PATH, output_dir: Path = CURATED_DIR
) -> None:
    """Build the star schema tables and write each as a Parquet file in `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(
            f"CREATE OR REPLACE VIEW scored_portfolio AS "
            f"SELECT * FROM read_parquet('{scored_path.as_posix()}')"
        )
        con.execute(DIM_VINTAGE_SQL)
        con.execute(DIM_CREDIT_GRADE_SQL)
        con.execute(DIM_BORROWER_SQL)
        con.execute(FACT_LOAN_RISK_PORTFOLIO_SQL)

        tables = {
            "Fact_Loan_Risk_Portfolio": "fact_loan_risk_portfolio",
            "Dim_Borrower": "dim_borrower",
            "Dim_Vintage": "dim_vintage",
            "Dim_Credit_Grade": "dim_credit_grade",
        }
        for file_name, table_name in tables.items():
            out_path = output_dir / f"{file_name}.parquet"
            con.execute(f"COPY {table_name} TO '{out_path.as_posix()}' (FORMAT PARQUET)")
            n_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            logger.info("Wrote %s (%s rows) -> %s", table_name, f"{n_rows:,}", out_path)
    finally:
        con.close()


if __name__ == "__main__":
    build_marts()
