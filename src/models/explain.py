"""Per-loan PD reason codes via SHAP, for adverse-action-style explainability.

Regulatory context: US consumer lending (Regulation B / ECOA) requires specific reasons when
credit is declined or priced adversely — "reason codes" derived from the factors that most
moved a model's decision. This computes SHAP values against the underlying (uncalibrated)
LightGBM booster: isotonic calibration is a monotonic rescaling of the final probability and
doesn't change which features drove the prediction, so explaining the raw booster gives the
same reason-code ranking as explaining the calibrated model, without needing SHAP support for
CalibratedClassifierCV directly.

Computing SHAP for the full ~2.26M-loan portfolio is unnecessary and slow: in production this
runs per-decision, on the specific loan being underwritten. Here it's computed for a
representative sample to demonstrate the capability.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import pandas as pd
import shap

from src.models.categorical_utils import apply_categorical_dtypes, build_categorical_dtypes
from src.models.features import CATEGORICAL_FEATURES, PD_FEATURES
from src.models.train_pd import CLEAN_PARQUET, PD_MODEL_PATH, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REASON_CODES_PATH = PROJECT_ROOT / "reports" / "loan_reason_codes.parquet"
TOP_K_REASONS = 5
SAMPLE_SIZE = 2000


def get_underlying_booster(calibrated_model) -> lgb.LGBMClassifier:
    """Unwrap CalibratedClassifierCV(FrozenEstimator(LGBMClassifier)) to the raw booster."""
    frozen_estimator = calibrated_model.calibrated_classifiers_[0].estimator
    return frozen_estimator.estimator


def sample_portfolio(path: Path = CLEAN_PARQUET, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """A reproducible sample of resolved loans to explain (not the full portfolio — see module docstring)."""
    con = duckdb.connect()
    try:
        dtypes = build_categorical_dtypes(con, path, CATEGORICAL_FEATURES)
        df = con.execute(
            f"""
            SELECT id, {", ".join(PD_FEATURES)} FROM (
                SELECT * FROM read_parquet('{path.as_posix()}') WHERE {TARGET} IS NOT NULL
            ) sub
            USING SAMPLE {n} (reservoir)
            """
        ).df()
    finally:
        con.close()
    apply_categorical_dtypes(df, dtypes)
    return df


def compute_reason_codes(df: pd.DataFrame, booster: lgb.LGBMClassifier, top_k: int = TOP_K_REASONS) -> pd.DataFrame:
    """Top-k SHAP-ranked features per loan, long-format (one row per loan/reason rank)."""
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(df[PD_FEATURES])
    if isinstance(shap_values, list):  # older shap/lightgbm combos return a 2-class list
        shap_values = shap_values[1]

    loan_ids = df["id"].to_numpy()
    feature_values = df[PD_FEATURES].to_numpy()
    records = []
    for row_idx, loan_id in enumerate(loan_ids):
        row_shap = shap_values[row_idx]
        ranked = pd.Series(row_shap, index=PD_FEATURES).abs().sort_values(ascending=False)
        for rank, feature in enumerate(ranked.index[:top_k], start=1):
            col_idx = PD_FEATURES.index(feature)
            value = row_shap[col_idx]
            records.append(
                {
                    "loan_id": loan_id,
                    "reason_rank": rank,
                    "feature": feature,
                    "feature_value": feature_values[row_idx, col_idx],
                    "shap_value": value,
                    "direction": "increases_risk" if value > 0 else "decreases_risk",
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    pd_model = joblib.load(PD_MODEL_PATH)
    booster = get_underlying_booster(pd_model)
    logger.info("Unwrapped calibrated PD model to its underlying LightGBM booster")

    sample = sample_portfolio()
    logger.info("Computing SHAP reason codes for a %s-loan sample...", len(sample))
    reasons = compute_reason_codes(sample, booster)

    REASON_CODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("reasons", reasons)
        con.execute(f"COPY reasons TO '{REASON_CODES_PATH.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
    logger.info("Wrote %s reason-code rows to %s", len(reasons), REASON_CODES_PATH)

    top_driver_counts = reasons.loc[reasons["reason_rank"] == 1, "feature"].value_counts().head(10)
    logger.info("Most common #1 risk driver across the sample:\n%s", top_driver_counts.to_string())


if __name__ == "__main__":
    main()
