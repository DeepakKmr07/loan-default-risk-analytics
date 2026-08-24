"""Reusable statistics for model governance: PSI, KS-statistic, and calibration reliability curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between a baseline (`expected`) and current (`actual`) distribution.

    Bucket edges are quantiles of `expected`, so bucket weights reflect the baseline
    (development-sample) population's shape; `actual` (a monitoring-period population) is
    then binned into those same edges. PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct)).
    """
    expected = np.asarray(expected, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = np.asarray(actual, dtype=float)
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0  # not enough distinct values in the baseline to bucket meaningfully
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = np.clip(expected_counts / expected_counts.sum(), 1e-6, None)
    actual_pct = np.clip(actual_counts / actual_counts.sum(), 1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def classify_psi(psi: float) -> str:
    """Label a PSI value per standard credit-risk monitoring thresholds."""
    if np.isnan(psi):
        return "insufficient data"
    if psi > PSI_SIGNIFICANT_THRESHOLD:
        return "significant shift"
    if psi > PSI_MODERATE_THRESHOLD:
        return "moderate shift"
    return "stable"


def compute_ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between score distributions of the two classes."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positive_scores = y_score[y_true == 1]
    negative_scores = y_score[y_true == 0]
    return float(stats.ks_2samp(positive_scores, negative_scores).statistic)


def reliability_curve(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions into deciles and compare mean predicted PD vs empirical default rate."""
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_score": np.asarray(y_score)})
    df["bin"] = pd.qcut(df["y_score"], q=n_bins, duplicates="drop")
    summary = (
        df.groupby("bin", observed=True)
        .agg(
            predicted_pd=("y_score", "mean"),
            empirical_default_rate=("y_true", "mean"),
            count=("y_true", "size"),
        )
        .reset_index(drop=True)
    )
    return summary
