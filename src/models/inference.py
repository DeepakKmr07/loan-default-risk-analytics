"""Score the entire loan portfolio with PD, LGD, EAD, and Expected Loss.

Unlike training (which only uses resolved loans), inference scores every loan in
`clean_loans.parquet`, including still-active accounts, so the output represents the
full current portfolio's risk (IFRS 9 / Basel-style EL = PD * LGD * EAD).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from src.models.categorical_utils import apply_categorical_dtypes, build_categorical_dtypes
from src.models.features import CATEGORICAL_FEATURES, PD_FEATURES, LGD_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean_loans.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
PD_MODEL_PATH = MODEL_DIR / "pd_model.pkl"
LGD_MODEL_PATH = MODEL_DIR / "lgd_model.pkl"
SCORED_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "processed" / "scored_portfolio.parquet"

# Identifying/dimensional columns carried through for downstream BI marts (Phase 3),
# beyond the raw model features.
PASSTHROUGH_COLUMNS = [
    "id",
    "issue_date",
    "issue_year",
    "issue_quarter",
    "vintage",
    "loan_status",
    "default_flag",
    "actual_lgd",
    "ead",
]


def load_portfolio(path: Path = CLEAN_PARQUET) -> pd.DataFrame:
    """Load the full loan portfolio (all resolution statuses) with consistent categorical dtypes."""
    con = duckdb.connect()
    try:
        dtypes = build_categorical_dtypes(con, path, CATEGORICAL_FEATURES)
        columns = sorted(set(PASSTHROUGH_COLUMNS + PD_FEATURES + LGD_FEATURES))
        df = con.execute(
            f"SELECT {', '.join(columns)} FROM read_parquet('{path.as_posix()}')"
        ).df()
    finally:
        con.close()
    apply_categorical_dtypes(df, dtypes)
    return df


def score_portfolio(
    df: pd.DataFrame,
    pd_model,
    lgd_model,
) -> pd.DataFrame:
    """Attach calibrated PD, predicted LGD, EAD, and Expected Loss to every loan."""
    scored = df.copy()
    scored["pd"] = pd_model.predict_proba(scored[PD_FEATURES])[:, 1]
    scored["lgd"] = lgd_model.predict(scored[LGD_FEATURES]).clip(0.0, 1.0)
    scored["expected_loss"] = scored["pd"] * scored["lgd"] * scored["ead"]
    return scored


def save_scored_portfolio(scored: pd.DataFrame, output_path: Path = SCORED_PORTFOLIO_PATH) -> None:
    """Write the scored portfolio to Parquet via DuckDB."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("scored_portfolio", scored)
        con.execute(f"COPY scored_portfolio TO '{output_path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
    logger.info("Wrote %s scored loans to %s", f"{len(scored):,}", output_path)


def main() -> None:
    pd_model = joblib.load(PD_MODEL_PATH)
    lgd_model = joblib.load(LGD_MODEL_PATH)
    logger.info("Loaded PD model from %s and LGD model from %s", PD_MODEL_PATH, LGD_MODEL_PATH)

    portfolio = load_portfolio()
    logger.info("Scoring %s loans in the portfolio", f"{len(portfolio):,}")

    scored = score_portfolio(portfolio, pd_model, lgd_model)
    save_scored_portfolio(scored)

    total_ead = scored["ead"].sum()
    weighted_pd = (scored["pd"] * scored["ead"]).sum() / total_ead
    total_el = scored["expected_loss"].sum()
    logger.info(
        "Portfolio summary -> Total EAD: $%s | Weighted-avg PD: %.4f | Total Expected Loss: $%s",
        f"{total_ead:,.0f}", weighted_pd, f"{total_el:,.0f}",
    )


if __name__ == "__main__":
    main()
