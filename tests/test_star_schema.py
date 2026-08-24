"""Referential integrity tests for the curated star schema: every foreign key on
Fact_Loan_Risk_Portfolio must resolve to a row in its dimension table, and dimension
keys must be unique.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
FACT_PATH = CURATED_DIR / "Fact_Loan_Risk_Portfolio.parquet"
DIM_BORROWER_PATH = CURATED_DIR / "Dim_Borrower.parquet"
DIM_VINTAGE_PATH = CURATED_DIR / "Dim_Vintage.parquet"
DIM_CREDIT_GRADE_PATH = CURATED_DIR / "Dim_Credit_Grade.parquet"

pytestmark = pytest.mark.skipif(
    not FACT_PATH.exists(),
    reason="Star schema not found - run `python -m src.bi.build_marts` first",
)


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW fact_loans AS SELECT * FROM read_parquet('{FACT_PATH.as_posix()}')")
    connection.execute(f"CREATE VIEW dim_borrower AS SELECT * FROM read_parquet('{DIM_BORROWER_PATH.as_posix()}')")
    connection.execute(f"CREATE VIEW dim_vintage AS SELECT * FROM read_parquet('{DIM_VINTAGE_PATH.as_posix()}')")
    connection.execute(
        f"CREATE VIEW dim_credit_grade AS SELECT * FROM read_parquet('{DIM_CREDIT_GRADE_PATH.as_posix()}')"
    )
    yield connection
    connection.close()


def test_borrower_key_referential_integrity(con):
    n_orphans = con.execute(
        """
        SELECT COUNT(*) FROM fact_loans f
        LEFT JOIN dim_borrower d ON f.borrower_key = d.borrower_key
        WHERE d.borrower_key IS NULL
        """
    ).fetchone()[0]
    assert n_orphans == 0


def test_vintage_key_referential_integrity(con):
    n_orphans = con.execute(
        """
        SELECT COUNT(*) FROM fact_loans f
        LEFT JOIN dim_vintage d ON f.vintage_key = d.vintage_key
        WHERE d.vintage_key IS NULL
        """
    ).fetchone()[0]
    assert n_orphans == 0


def test_credit_grade_key_referential_integrity(con):
    n_orphans = con.execute(
        """
        SELECT COUNT(*) FROM fact_loans f
        LEFT JOIN dim_credit_grade d ON f.credit_grade_key = d.credit_grade_key
        WHERE d.credit_grade_key IS NULL
        """
    ).fetchone()[0]
    assert n_orphans == 0


@pytest.mark.parametrize(
    "table,key",
    [("dim_borrower", "borrower_key"), ("dim_vintage", "vintage_key"), ("dim_credit_grade", "credit_grade_key")],
)
def test_dimension_keys_are_unique(con, table, key):
    n_rows, n_distinct = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {key}) FROM {table}").fetchone()
    assert n_rows == n_distinct


def test_fact_loan_count_matches_dim_borrower(con):
    """Dim_Borrower is 1:1 with loans by design (no persistent customer id in this dataset)."""
    n_fact = con.execute("SELECT COUNT(*) FROM fact_loans").fetchone()[0]
    n_borrower = con.execute("SELECT COUNT(*) FROM dim_borrower").fetchone()[0]
    assert n_fact == n_borrower
