"""Rolling-window (expanding, walk-forward) out-of-time backtests for the PD and LGD models.

A single OOT cutoff (used by train_pd.py / train_lgd.py to produce the deployed model) only
answers "did the model work on one held-out period." This answers the sharper question a
credit-risk reviewer actually asks: does performance hold up consistently across *multiple*
out-of-time windows, or does it look good on one slice and fall apart on others?

Walk-forward design: the timeline is split into (folds + 1) equal chronological segments.
Fold i trains on all segments before it and tests on segment i -- an expanding window that
never touches future data, mirroring how the model would actually be retrained and evaluated
over time in production. Hyperparameters are held fixed (the tuned ones if available, else the
same defaults used for the deployed model) so this measures the stability of one modeling
choice across time, not the effect of re-tuning per fold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)

from src.models import train_lgd, train_pd
from src.models.features import LGD_FEATURES, PD_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PD_BACKTEST_CSV = REPORTS_DIR / "rolling_backtest_pd.csv"
LGD_BACKTEST_CSV = REPORTS_DIR / "rolling_backtest_lgd.csv"
PD_BACKTEST_PLOT = REPORTS_DIR / "rolling_backtest_pd.png"
BACKTEST_SUMMARY_PATH = REPORTS_DIR / "rolling_backtest_summary.json"

DEFAULT_N_FOLDS = 5


def expanding_windows(df: pd.DataFrame, n_folds: int = DEFAULT_N_FOLDS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split a chronologically-sorted `df` into n_folds expanding train/test windows.

    The timeline is cut into (n_folds + 1) equal-sized segments. Fold i trains on segments
    [0..i) and tests on segment i, so each fold's test window is strictly later in time than
    everything it trained on.
    """
    n_segments = n_folds + 1
    edges = [int(len(df) * i / n_segments) for i in range(n_segments + 1)]
    windows = []
    for i in range(1, n_segments):
        train = df.iloc[: edges[i]]
        test = df.iloc[edges[i] : edges[i + 1]]
        windows.append((train, test))
    return windows


def backtest_pd(n_folds: int = DEFAULT_N_FOLDS) -> pd.DataFrame:
    """Walk-forward backtest of the PD model's discrimination and calibration across time."""
    df = train_pd.load_labeled_data()
    best_params = train_pd.load_best_params(train_pd.PD_BEST_PARAMS_PATH)

    rows = []
    for fold, (train, test) in enumerate(expanding_windows(df, n_folds), start=1):
        model = train_pd.train_lightgbm(train, best_params)
        proba = model.predict_proba(test[PD_FEATURES])[:, 1]
        row = {
            "fold": fold,
            "train_start": str(train["issue_date"].min().date()),
            "train_end": str(train["issue_date"].max().date()),
            "test_start": str(test["issue_date"].min().date()),
            "test_end": str(test["issue_date"].max().date()),
            "n_train": len(train),
            "n_test": len(test),
            "roc_auc": round(float(roc_auc_score(test[train_pd.TARGET], proba)), 4),
            "pr_auc": round(float(average_precision_score(test[train_pd.TARGET], proba)), 4),
            "brier_score": round(float(brier_score_loss(test[train_pd.TARGET], proba)), 4),
        }
        logger.info(
            "[PD fold %s] test %s..%s (n=%s) -> ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f",
            fold, row["test_start"], row["test_end"], row["n_test"],
            row["roc_auc"], row["pr_auc"], row["brier_score"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def backtest_lgd(n_folds: int = DEFAULT_N_FOLDS) -> pd.DataFrame:
    """Walk-forward backtest of the LGD model's RMSE/MAE across time."""
    df = train_lgd.load_defaulted_loans()
    best_params = train_lgd.load_best_params(train_lgd.LGD_BEST_PARAMS_PATH)

    rows = []
    for fold, (train, test) in enumerate(expanding_windows(df, n_folds), start=1):
        model = train_lgd.train_lightgbm(train, best_params)
        preds = model.predict(test[LGD_FEATURES]).clip(0.0, 1.0)
        row = {
            "fold": fold,
            "train_start": str(train["issue_date"].min().date()),
            "train_end": str(train["issue_date"].max().date()),
            "test_start": str(test["issue_date"].min().date()),
            "test_end": str(test["issue_date"].max().date()),
            "n_train": len(train),
            "n_test": len(test),
            "rmse": round(float(root_mean_squared_error(test[train_lgd.TARGET], preds)), 4),
            "mae": round(float(mean_absolute_error(test[train_lgd.TARGET], preds)), 4),
        }
        logger.info(
            "[LGD fold %s] test %s..%s (n=%s) -> RMSE=%.4f  MAE=%.4f",
            fold, row["test_start"], row["test_end"], row["n_test"], row["rmse"], row["mae"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_pd_stability(pd_folds: pd.DataFrame, output_path: Path) -> None:
    """Plot ROC-AUC and Brier score across folds to visualize performance stability over time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pd_folds["fold"], pd_folds["roc_auc"], marker="o", label="ROC-AUC")
    ax.plot(pd_folds["fold"], pd_folds["brier_score"], marker="s", label="Brier score")
    ax.set_xlabel("Fold (expanding window, chronological)")
    ax.set_ylabel("Metric value")
    ax.set_title("PD Model Stability Across Rolling Out-of-Time Windows")
    ax.set_xticks(pd_folds["fold"])
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(n_folds: int = DEFAULT_N_FOLDS) -> None:
    logger.info("Running PD rolling-window backtest (%s folds)...", n_folds)
    pd_folds = backtest_pd(n_folds)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd_folds.to_csv(PD_BACKTEST_CSV, index=False)
    logger.info("Wrote PD rolling backtest to %s", PD_BACKTEST_CSV)
    plot_pd_stability(pd_folds, PD_BACKTEST_PLOT)
    logger.info("Wrote PD stability plot to %s", PD_BACKTEST_PLOT)

    logger.info("Running LGD rolling-window backtest (%s folds)...", n_folds)
    lgd_folds = backtest_lgd(n_folds)
    lgd_folds.to_csv(LGD_BACKTEST_CSV, index=False)
    logger.info("Wrote LGD rolling backtest to %s", LGD_BACKTEST_CSV)

    summary = {
        "n_folds": n_folds,
        "pd_model": {
            "roc_auc_mean": round(float(pd_folds["roc_auc"].mean()), 4),
            "roc_auc_std": round(float(pd_folds["roc_auc"].std()), 4),
            "brier_score_mean": round(float(pd_folds["brier_score"].mean()), 4),
            "brier_score_std": round(float(pd_folds["brier_score"].std()), 4),
        },
        "lgd_model": {
            "rmse_mean": round(float(lgd_folds["rmse"].mean()), 4),
            "rmse_std": round(float(lgd_folds["rmse"].std()), 4),
        },
        "artifacts": {
            "pd_csv": str(PD_BACKTEST_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "lgd_csv": str(LGD_BACKTEST_CSV.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "pd_plot": str(PD_BACKTEST_PLOT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    BACKTEST_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote rolling backtest summary to %s", BACKTEST_SUMMARY_PATH)
    logger.info(
        "PD stability: ROC-AUC %.4f +/- %.4f across %s folds",
        summary["pd_model"]["roc_auc_mean"], summary["pd_model"]["roc_auc_std"], n_folds,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rolling-window (walk-forward) OOT backtest for PD/LGD models")
    parser.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS, help="Number of expanding-window folds")
    args = parser.parse_args()
    main(n_folds=args.folds)
