"""Optuna-based hyperparameter tuning for the PD and LGD LightGBM models.

Tuning is opt-in (`--tune` on train_pd.py / train_lgd.py) rather than run on every pipeline
execution: a full Optuna search takes minutes, while the default pipeline run should stay fast
and deterministic. Tuning carves its own validation slice out of the *training* period only —
it never touches the calibration or out-of-time test sets used for final model evaluation, to
avoid leaking information into either.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

logging.getLogger("optuna").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TUNE_VALIDATION_FRACTION = 0.2


def time_based_tune_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a chronological validation slice out of a training set, for tuning only."""
    n = len(train)
    cutoff = int(n * (1 - TUNE_VALIDATION_FRACTION))
    return train.iloc[:cutoff], train.iloc[cutoff:]


def _suggest_common_params(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }


def tune_classifier(
    train: pd.DataFrame, features: list[str], categorical_features: list[str], target: str,
    n_trials: int = 25,
) -> dict:
    """Tune an LGBMClassifier via Optuna, maximizing ROC-AUC on a held-out training-period slice."""
    tune_train, tune_val = time_based_tune_split(train)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_common_params(trial)
        model = lgb.LGBMClassifier(objective="binary", random_state=RANDOM_STATE, verbosity=-1, **params)
        model.fit(tune_train[features], tune_train[target], categorical_feature=categorical_features)
        proba = model.predict_proba(tune_val[features])[:, 1]
        return roc_auc_score(tune_val[target], proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Best PD trial: ROC-AUC=%.4f  params=%s", study.best_value, study.best_params)
    return study.best_params


def tune_regressor(
    train: pd.DataFrame, features: list[str], categorical_features: list[str], target: str,
    n_trials: int = 25,
) -> dict:
    """Tune an LGBMRegressor via Optuna, minimizing RMSE on a held-out training-period slice."""
    tune_train, tune_val = time_based_tune_split(train)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_common_params(trial)
        model = lgb.LGBMRegressor(objective="regression", random_state=RANDOM_STATE, verbosity=-1, **params)
        model.fit(tune_train[features], tune_train[target], categorical_feature=categorical_features)
        preds = model.predict(tune_val[features]).clip(0.0, 1.0)
        return mean_squared_error(tune_val[target], preds) ** 0.5

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Best LGD trial: RMSE=%.4f  params=%s", study.best_value, study.best_params)
    return study.best_params


def save_best_params(params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))


def load_best_params(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())
