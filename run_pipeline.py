"""Master orchestration script: ETL -> Train PD/LGD -> Inference -> Curated Star Schema.

Run with the project's system Python (see DEVELOPMENT.md "Environment Setup" — no venv/conda here):
    "C:\\Program Files\\Python313\\python.exe" run_pipeline.py

Note: this assumes `data/raw/loans_raw.parquet` already exists (run `src/data/download.py`
once, with Kaggle API credentials configured, to produce it).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from src.bi.build_marts import build_marts
from src.data.etl_duckdb import run_etl
from src.models import inference, train_lgd, train_pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SCORED_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "processed" / "scored_portfolio.parquet"


def print_portfolio_summary(path: Path = SCORED_PORTFOLIO_PATH) -> None:
    """Print overall and per-grade Total EAD / Weighted-Average PD / Total Expected Loss."""
    con = duckdb.connect()
    try:
        overall = con.execute(
            f"""
            SELECT
                SUM(ead) AS total_ead,
                SUM(pd * ead) / NULLIF(SUM(ead), 0) AS weighted_pd,
                SUM(pd * lgd * ead) AS total_expected_loss
            FROM read_parquet('{path.as_posix()}')
            """
        ).fetchone()
        by_grade = con.execute(
            f"""
            SELECT
                grade,
                SUM(ead) AS total_ead,
                SUM(pd * ead) / NULLIF(SUM(ead), 0) AS weighted_pd,
                SUM(pd * lgd * ead) AS total_expected_loss
            FROM read_parquet('{path.as_posix()}')
            GROUP BY grade
            ORDER BY grade
            """
        ).fetchall()
    finally:
        con.close()

    total_ead, weighted_pd, total_el = overall
    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"{'Total EAD':<28}: ${total_ead:,.0f}")
    print(f"{'Weighted-Average PD':<28}: {weighted_pd:.2%}")
    print(f"{'Total Expected Loss':<28}: ${total_el:,.0f}")
    print(f"{'EL Rate':<28}: {total_el / total_ead:.2%}")
    print("-" * 60)
    print(f"{'Grade':<8}{'Total EAD':>16}{'Weighted PD':>16}{'Expected Loss':>18}")
    for grade, g_ead, g_pd, g_el in by_grade:
        print(f"{grade:<8}{g_ead:>16,.0f}{g_pd:>16.2%}{g_el:>18,.0f}")
    print("=" * 60 + "\n")


def main() -> None:
    logger.info("STEP 1/4: ETL")
    run_etl()

    logger.info("STEP 2/4: Train PD and LGD models")
    train_pd.main()
    train_lgd.main()

    logger.info("STEP 3/4: Score portfolio (inference)")
    inference.main()

    logger.info("STEP 4/4: Build curated Star Schema marts")
    build_marts()

    print_portfolio_summary()


if __name__ == "__main__":
    main()
