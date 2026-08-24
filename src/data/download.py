"""Download the LendingClub accepted-loans dataset and stream it into Parquet via DuckDB.

Requires Kaggle API credentials to be configured (either a ``~/.kaggle/kaggle.json``
file or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables) so that
``kagglehub`` can authenticate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import kagglehub

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_DATASET = "wordsforthewise/lending-club"
RAW_FILE_GLOB = "accepted_2007_to_2018Q4.csv*"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "raw" / "loans_raw.parquet"


def download_dataset() -> Path:
    """Download the Kaggle LendingClub dataset and return the path to the accepted-loans CSV."""
    logger.info("Downloading dataset '%s' via kagglehub...", KAGGLE_DATASET)
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    logger.info("Dataset cached at %s", dataset_dir)

    candidates = sorted(dataset_dir.rglob(RAW_FILE_GLOB))
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate '{RAW_FILE_GLOB}' under {dataset_dir}. "
            "Check that the Kaggle dataset layout has not changed."
        )
    return candidates[0]


def convert_to_parquet(csv_path: Path, output_path: Path = OUTPUT_PARQUET) -> None:
    """Stream the raw CSV straight into Parquet with DuckDB, without materializing it in pandas.

    All columns are read as VARCHAR: the source CSV mixes numeric/percentage/date
    formatting across ~150 columns and years of vintages, so type casting is deferred
    to the ETL stage where it can be done explicitly and safely.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Streaming %s -> %s via DuckDB...", csv_path, output_path)

    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_csv_auto(
                    '{csv_path.as_posix()}',
                    ALL_VARCHAR = TRUE,
                    IGNORE_ERRORS = TRUE
                )
            ) TO '{output_path.as_posix()}' (FORMAT PARQUET)
            """
        )
        row_count = con.execute(f"SELECT COUNT(*) FROM '{output_path.as_posix()}'").fetchone()[0]
    finally:
        con.close()
    logger.info("Wrote %s rows to %s", f"{row_count:,}", output_path)


def main() -> None:
    csv_path = download_dataset()
    convert_to_parquet(csv_path)


if __name__ == "__main__":
    main()
