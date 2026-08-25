"""Master orchestration script: ETL -> Train PD/LGD -> Inference -> Curated Star Schema.

    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --stress-test    # + stress-test scenarios
    python run_pipeline.py --test           # run the pytest suite only
    python run_pipeline.py --dashboard      # launch the Streamlit app

Note: this assumes `data/raw/loans_raw.parquet` already exists (run `src/data/download.py`
once, with Kaggle API credentials configured, to produce it).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import duckdb

from src.bi.build_marts import build_marts
from src.data.etl_duckdb import run_etl
from src.models import inference, train_lgd, train_pd
from src.models.stress_test import run_stress_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SCORED_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "processed" / "scored_portfolio.parquet"
TESTS_DIR = PROJECT_ROOT / "tests"
DASHBOARD_PATH = PROJECT_ROOT / "app" / "dashboard.py"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit risk pipeline orchestration")
    parser.add_argument(
        "--test", action="store_true", help="Run the automated pytest suite instead of the pipeline"
    )
    parser.add_argument(
        "--stress-test", action="store_true",
        help="Also run macroeconomic stress-test scenarios (Baseline/Adverse/Severely Adverse) after the marts are built",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch the Streamlit dashboard (app/dashboard.py) instead of running the pipeline",
    )
    return parser.parse_args()


def run_test_suite() -> int:
    """Run the pytest suite and return its exit code (0 = all passed)."""
    import pytest

    return pytest.main([str(TESTS_DIR), "-v"])


def run_dashboard() -> int:
    """Launch the Streamlit dashboard. Blocks until the user stops the server (Ctrl+C)."""
    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(DASHBOARD_PATH)])
    return result.returncode


def main() -> None:
    args = parse_args()

    if args.test:
        sys.exit(run_test_suite())

    if args.dashboard:
        sys.exit(run_dashboard())

    logger.info("STEP 1/4: ETL")
    run_etl()

    logger.info("STEP 2/4: Train PD and LGD models")
    train_pd.main()
    train_lgd.main()

    logger.info("STEP 3/4: Score portfolio (inference)")
    inference.main()

    logger.info("STEP 4/4: Build curated Star Schema marts")
    build_marts()

    if args.stress_test:
        logger.info("STEP 5/5: Run macroeconomic stress-test scenarios")
        run_stress_test()

    print_portfolio_summary()


if __name__ == "__main__":
    main()
