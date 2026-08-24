"""Train a LightGBM regression model for Loss Given Default (LGD) on realized, resolved defaults.

Only loans with a realized `actual_lgd` (computed in the ETL stage from post-default recoveries
net of collection fees) are used. Splitting is chronological, matching DEVELOPMENT.md rule #1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.models.categorical_utils import apply_categorical_dtypes, build_categorical_dtypes
from src.models.features import CATEGORICAL_FEATURES, LGD_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean_loans.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
LGD_MODEL_PATH = MODEL_DIR / "lgd_model.pkl"

TARGET = "actual_lgd"
OOT_TEST_FRACTION = 0.2


def load_defaulted_loans(path: Path = CLEAN_PARQUET) -> pd.DataFrame:
    """Load resolved, charged-off/defaulted loans with a realized actual_lgd, oldest first."""
    con = duckdb.connect()
    try:
        dtypes = build_categorical_dtypes(con, path, CATEGORICAL_FEATURES)
        df = con.execute(
            f"""
            SELECT issue_date, {", ".join(LGD_FEATURES)}, {TARGET}
            FROM read_parquet('{path.as_posix()}')
            WHERE {TARGET} IS NOT NULL
            ORDER BY issue_date
            """
        ).df()
    finally:
        con.close()
    apply_categorical_dtypes(df, dtypes)
    return df


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train and out-of-time test sets (never random shuffle)."""
    n = len(df)
    cutoff_idx = int(n * (1 - OOT_TEST_FRACTION))
    train, test_oot = df.iloc[:cutoff_idx], df.iloc[cutoff_idx:]
    logger.info(
        "Split sizes -> train: %s [%s .. %s], OOT test: %s [%s .. %s]",
        len(train), train["issue_date"].min(), train["issue_date"].max(),
        len(test_oot), test_oot["issue_date"].min(), test_oot["issue_date"].max(),
    )
    return train, test_oot


def train_lightgbm(train: pd.DataFrame) -> lgb.LGBMRegressor:
    """Fit a LightGBM regressor to predict realized LGD on defaulted loans."""
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(train[LGD_FEATURES], train[TARGET], categorical_feature=CATEGORICAL_FEATURES)
    return model


def evaluate(model: lgb.LGBMRegressor, test: pd.DataFrame) -> dict[str, float]:
    """Score RMSE and MAE on the OOT test set. Predictions are clipped to the valid [0, 1] LGD range."""
    preds = model.predict(test[LGD_FEATURES]).clip(0.0, 1.0)
    metrics = {
        "rmse": root_mean_squared_error(test[TARGET], preds),
        "mae": mean_absolute_error(test[TARGET], preds),
    }
    logger.info("[LGD] RMSE=%.4f  MAE=%.4f", metrics["rmse"], metrics["mae"])
    return metrics


def main() -> None:
    df = load_defaulted_loans()
    train, test_oot = time_based_split(df)

    model = train_lightgbm(train)
    evaluate(model, test_oot)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LGD_MODEL_PATH)
    logger.info("Saved LGD model to %s", LGD_MODEL_PATH)


if __name__ == "__main__":
    main()
