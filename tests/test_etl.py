"""Data-contract tests for the ETL output (`clean_loans.parquet`) that every downstream
model and mart depends on: non-null/unique ids, a valid default_flag domain, and a valid
actual_lgd range.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean_loans.parquet"

pytestmark = pytest.mark.skipif(
    not CLEAN_PARQUET.exists(),
    reason="clean_loans.parquet not found - run `python -m src.data.etl_duckdb` first",
)


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW clean_loans AS SELECT * FROM read_parquet('{CLEAN_PARQUET.as_posix()}')")
    yield connection
    connection.close()


def test_no_null_ids(con):
    n_null = con.execute("SELECT COUNT(*) FROM clean_loans WHERE id IS NULL").fetchone()[0]
    assert n_null == 0


def test_ids_are_unique(con):
    n_rows, n_distinct = con.execute("SELECT COUNT(*), COUNT(DISTINCT id) FROM clean_loans").fetchone()
    assert n_rows == n_distinct


def test_default_flag_domain(con):
    """default_flag must be 0, 1, or NULL (still-active loans excluded from training)."""
    n_invalid = con.execute(
        "SELECT COUNT(*) FROM clean_loans WHERE default_flag IS NOT NULL AND default_flag NOT IN (0, 1)"
    ).fetchone()[0]
    assert n_invalid == 0


def test_actual_lgd_range(con):
    n_invalid = con.execute(
        "SELECT COUNT(*) FROM clean_loans WHERE actual_lgd IS NOT NULL AND (actual_lgd < 0 OR actual_lgd > 1)"
    ).fetchone()[0]
    assert n_invalid == 0


def test_actual_lgd_only_present_for_defaults(con):
    """actual_lgd is a realized-loss measure, so it should only be populated for defaulted loans."""
    n_mismatched = con.execute(
        "SELECT COUNT(*) FROM clean_loans WHERE actual_lgd IS NOT NULL AND default_flag != 1"
    ).fetchone()[0]
    assert n_mismatched == 0


def test_ead_non_negative(con):
    n_invalid = con.execute("SELECT COUNT(*) FROM clean_loans WHERE ead < 0").fetchone()[0]
    assert n_invalid == 0
