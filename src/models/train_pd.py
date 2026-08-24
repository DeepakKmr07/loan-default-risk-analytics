"""Train a calibrated LightGBM Probability of Default (PD) model with Out-of-Time validation.

Splitting is strictly chronological (DEVELOPMENT.md rule #1: never random-shuffle across time):
train < calibration < out-of-time (OOT) test, in issue-date order throughout. Predicted
probabilities are calibrated with isotonic regression (DEVELOPMENT.md rule #3) so risk estimates
track empirical default rates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.models.categorical_utils import apply_categorical_dtypes, build_categorical_dtypes
from src.models.features import CATEGORICAL_FEATURES, PD_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean_loans.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
PD_MODEL_PATH = MODEL_DIR / "pd_model.pkl"

TARGET = "default_flag"
OOT_TEST_FRACTION = 0.2
CALIBRATION_FRACTION = 0.2  # carved from the most recent slice of the remaining training period


def load_labeled_data(path: Path = CLEAN_PARQUET) -> pd.DataFrame:
    """Load resolved loans (default_flag IS NOT NULL) needed for PD training, oldest first."""
    con = duckdb.connect()
    try:
        dtypes = build_categorical_dtypes(con, path, CATEGORICAL_FEATURES)
        df = con.execute(
            f"""
            SELECT issue_date, {", ".join(PD_FEATURES)}, {TARGET}
            FROM read_parquet('{path.as_posix()}')
            WHERE {TARGET} IS NOT NULL
            ORDER BY issue_date
            """
        ).df()
    finally:
        con.close()
    apply_categorical_dtypes(df, dtypes)
    return df


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train / calibration / out-of-time test sets."""
    n = len(df)
    oot_cutoff_idx = int(n * (1 - OOT_TEST_FRACTION))
    train_val, test_oot = df.iloc[:oot_cutoff_idx], df.iloc[oot_cutoff_idx:]

    calib_cutoff_idx = int(len(train_val) * (1 - CALIBRATION_FRACTION))
    train, calib = train_val.iloc[:calib_cutoff_idx], train_val.iloc[calib_cutoff_idx:]

    logger.info(
        "Split sizes -> train: %s [%s .. %s], calibration: %s [%s .. %s], OOT test: %s [%s .. %s]",
        len(train), train["issue_date"].min(), train["issue_date"].max(),
        len(calib), calib["issue_date"].min(), calib["issue_date"].max(),
        len(test_oot), test_oot["issue_date"].min(), test_oot["issue_date"].max(),
    )
    return train, calib, test_oot


def train_lightgbm(train: pd.DataFrame) -> lgb.LGBMClassifier:
    """Fit a LightGBM classifier on the training slice."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(train[PD_FEATURES], train[TARGET], categorical_feature=CATEGORICAL_FEATURES)
    return model


def calibrate_model(model: lgb.LGBMClassifier, calib: pd.DataFrame) -> CalibratedClassifierCV:
    """Calibrate predicted probabilities against empirical default rates via isotonic regression."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(calib[PD_FEATURES], calib[TARGET])
    return calibrated


def evaluate(model, test: pd.DataFrame, label: str) -> dict[str, float]:
    """Score ROC-AUC, PR-AUC, and Brier score on the OOT test set."""
    proba = model.predict_proba(test[PD_FEATURES])[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(test[TARGET], proba),
        "pr_auc": average_precision_score(test[TARGET], proba),
        "brier_score": brier_score_loss(test[TARGET], proba),
    }
    logger.info(
        "[%s] ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f",
        label, metrics["roc_auc"], metrics["pr_auc"], metrics["brier_score"],
    )
    return metrics


def main() -> None:
    df = load_labeled_data()
    train, calib, test_oot = time_based_split(df)

    raw_model = train_lightgbm(train)
    evaluate(raw_model, test_oot, "uncalibrated")

    calibrated_model = calibrate_model(raw_model, calib)
    evaluate(calibrated_model, test_oot, "calibrated")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, PD_MODEL_PATH)
    logger.info("Saved calibrated PD model to %s", PD_MODEL_PATH)


if __name__ == "__main__":
    main()
