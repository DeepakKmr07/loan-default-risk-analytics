"""DuckDB ETL: clean raw LendingClub loans, engineer risk features, and label outcomes.

All transformations run as vectorized SQL directly against the raw Parquet file — the
data is never materialized into a pandas DataFrame here.

Target definition:
    default_flag = 1   if loan_status indicates a realized default/charge-off
    default_flag = 0   if loan_status == 'Fully Paid'
    default_flag = NULL for loans still active (Current, Issued, Late, In Grace Period, ...)
Active loans are kept in the output (needed for portfolio-level EL scoring downstream)
but must be filtered out (`WHERE default_flag IS NOT NULL`) before model training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PARQUET = PROJECT_ROOT / "data" / "raw" / "loans_raw.parquet"
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean_loans.parquet"

DEFAULT_STATUSES = (
    "Charged Off",
    "Default",
    "Does not meet credit policy: Status Charged Off",
)
PAID_STATUSES = ("Fully Paid",)

# Curated subset of raw columns used downstream. try_cast is used throughout because the
# raw Parquet stores every column as VARCHAR (see download.py) and LendingClub's own export
# mixes formats/garbage footer rows across vintages.
CLEAN_SQL_TEMPLATE = """
WITH base AS (
    SELECT
        id,
        try_cast(loan_amnt AS DOUBLE) AS loan_amnt,
        try_cast(funded_amnt AS DOUBLE) AS funded_amnt,
        try_cast(regexp_extract(term, '(\\d+)', 1) AS INTEGER) AS term_months,
        try_cast(trim(replace(int_rate, '%', '')) AS DOUBLE) AS int_rate,
        try_cast(installment AS DOUBLE) AS installment,
        grade,
        sub_grade,
        CASE
            WHEN emp_length ILIKE '%< 1%' THEN 0
            WHEN emp_length ILIKE '%10+%' THEN 10
            ELSE try_cast(regexp_extract(emp_length, '(\\d+)', 1) AS INTEGER)
        END AS emp_length_years,
        home_ownership,
        try_cast(annual_inc AS DOUBLE) AS annual_inc,
        verification_status,
        try_strptime(issue_d, '%b-%Y') AS issue_date,
        loan_status,
        purpose,
        addr_state,
        try_cast(dti AS DOUBLE) AS dti,
        try_cast(delinq_2yrs AS DOUBLE) AS delinq_2yrs,
        try_strptime(earliest_cr_line, '%b-%Y') AS earliest_cr_line_date,
        try_cast(inq_last_6mths AS DOUBLE) AS inq_last_6mths,
        try_cast(open_acc AS DOUBLE) AS open_acc,
        try_cast(pub_rec AS DOUBLE) AS pub_rec,
        try_cast(revol_bal AS DOUBLE) AS revol_bal,
        try_cast(trim(replace(revol_util, '%', '')) AS DOUBLE) AS revol_util,
        try_cast(total_acc AS DOUBLE) AS total_acc,
        initial_list_status,
        application_type,
        try_cast(mort_acc AS DOUBLE) AS mort_acc,
        try_cast(pub_rec_bankruptcies AS DOUBLE) AS pub_rec_bankruptcies,
        try_cast(tot_cur_bal AS DOUBLE) AS tot_cur_bal,
        try_cast(total_bal_ex_mort AS DOUBLE) AS total_bal_ex_mort,
        try_cast(total_bc_limit AS DOUBLE) AS total_bc_limit,
        try_cast(tot_hi_cred_lim AS DOUBLE) AS tot_hi_cred_lim,
        try_cast(avg_cur_bal AS DOUBLE) AS avg_cur_bal,
        try_cast(bc_util AS DOUBLE) AS bc_util,
        try_cast(bc_open_to_buy AS DOUBLE) AS bc_open_to_buy,
        try_cast(num_actv_bc_tl AS DOUBLE) AS num_actv_bc_tl,
        try_cast(num_tl_90g_dpd_24m AS DOUBLE) AS num_tl_90g_dpd_24m,
        try_cast(mo_sin_old_rev_tl_op AS DOUBLE) AS mo_sin_old_rev_tl_op,
        try_cast(percent_bc_gt_75 AS DOUBLE) AS percent_bc_gt_75,
        try_cast(acc_open_past_24mths AS DOUBLE) AS acc_open_past_24mths,
        try_cast(out_prncp AS DOUBLE) AS out_prncp,
        try_cast(total_pymnt AS DOUBLE) AS total_pymnt,
        try_cast(total_rec_prncp AS DOUBLE) AS total_rec_prncp,
        try_cast(total_rec_int AS DOUBLE) AS total_rec_int,
        try_cast(total_rec_late_fee AS DOUBLE) AS total_rec_late_fee,
        try_cast(recoveries AS DOUBLE) AS recoveries,
        try_cast(collection_recovery_fee AS DOUBLE) AS collection_recovery_fee,
        try_cast(last_pymnt_amnt AS DOUBLE) AS last_pymnt_amnt
    FROM read_parquet('{raw_path}')
    -- Drop footer/summary rows the LendingClub export appends (they have no numeric loan_amnt)
    WHERE id IS NOT NULL AND try_cast(loan_amnt AS DOUBLE) IS NOT NULL
)
SELECT
    *,
    date_part('year', issue_date) AS issue_year,
    date_part('quarter', issue_date) AS issue_quarter,
    date_part('year', issue_date)::VARCHAR || 'Q' || date_part('quarter', issue_date)::VARCHAR AS vintage,
    date_diff('month', earliest_cr_line_date, issue_date) AS credit_history_months,

    -- Debt-to-income features
    (installment * 12.0) / NULLIF(annual_inc, 0) AS installment_to_income,
    funded_amnt / NULLIF(annual_inc, 0) AS loan_to_income,
    revol_bal / NULLIF(annual_inc, 0) AS revol_bal_to_income,

    -- Credit utilization features
    revol_util / 100.0 AS credit_utilization,
    bc_util / 100.0 AS bankcard_utilization,

    -- Target: default_flag (NULL = still active, excluded from model training)
    CASE
        WHEN loan_status IN {default_statuses} THEN 1
        WHEN loan_status IN {paid_statuses} THEN 0
        ELSE NULL
    END AS default_flag,

    -- Exposure at Default: funded amount is the standard EAD proxy for fixed-amortization
    -- installment loans (as opposed to revolving lines, where EAD needs a utilization model).
    funded_amnt AS ead,

    -- Realized LGD on resolved defaults: 1 - net recovery rate, net of collection fees, clipped to [0, 1].
    CASE
        WHEN loan_status IN {default_statuses} THEN
            LEAST(GREATEST(
                1.0 - (total_rec_prncp + recoveries - collection_recovery_fee) / NULLIF(funded_amnt, 0),
                0.0
            ), 1.0)
        ELSE NULL
    END AS actual_lgd
FROM base
WHERE issue_date IS NOT NULL
"""


def _sql_string_list(values: tuple[str, ...]) -> str:
    """Render a tuple of strings as a valid SQL `(...)` literal list for an IN clause."""
    escaped = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"({escaped})"


def build_clean_view(con: duckdb.DuckDBPyConnection, raw_path: Path) -> None:
    """Register the cleaned/engineered dataset as a DuckDB view named `clean_loans`."""
    sql = CLEAN_SQL_TEMPLATE.format(
        raw_path=raw_path.as_posix(),
        default_statuses=_sql_string_list(DEFAULT_STATUSES),
        paid_statuses=_sql_string_list(PAID_STATUSES),
    )
    con.execute(f"CREATE OR REPLACE VIEW clean_loans AS {sql}")


def run_etl(raw_path: Path = RAW_PARQUET, output_path: Path = CLEAN_PARQUET) -> None:
    """Clean the raw loan Parquet, engineer features/labels, and write the result to Parquet."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run download.py first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running DuckDB ETL: %s -> %s", raw_path, output_path)

    con = duckdb.connect()
    try:
        build_clean_view(con, raw_path)
        con.execute(f"COPY clean_loans TO '{output_path.as_posix()}' (FORMAT PARQUET)")

        stats = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,
                SUM(CASE WHEN default_flag = 1 THEN 1 ELSE 0 END) AS n_default,
                SUM(CASE WHEN default_flag = 0 THEN 1 ELSE 0 END) AS n_paid,
                SUM(CASE WHEN default_flag IS NULL THEN 1 ELSE 0 END) AS n_active,
                SUM(CASE WHEN actual_lgd IS NOT NULL THEN 1 ELSE 0 END) AS n_with_lgd
            FROM clean_loans
            """
        ).fetchone()
    finally:
        con.close()

    n_rows, n_default, n_paid, n_active, n_with_lgd = stats
    logger.info(
        "Wrote %s rows (default=%s, fully_paid=%s, active/excluded=%s, with actual_lgd=%s)",
        f"{n_rows:,}", f"{n_default:,}", f"{n_paid:,}", f"{n_active:,}", f"{n_with_lgd:,}",
    )


if __name__ == "__main__":
    run_etl()
