"""Unit tests for the PSI / KS / calibration utilities used in model governance reporting.

These are self-contained (synthetic data only) so they run without the multi-hundred-MB
processed dataset or trained model artifacts being present.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.governance_utils import (
    classify_psi,
    compute_ks_statistic,
    compute_psi,
    reliability_curve,
)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    baseline = rng.normal(size=5000)
    assert compute_psi(baseline, baseline) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_large_mean_shift():
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=0.0, scale=1.0, size=5000)
    shifted = rng.normal(loc=2.0, scale=1.0, size=5000)
    assert compute_psi(baseline, shifted) > 0.25


def test_psi_small_shift_stays_below_significant_threshold():
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=0.0, scale=1.0, size=5000)
    slightly_shifted = rng.normal(loc=0.05, scale=1.0, size=5000)
    assert compute_psi(baseline, slightly_shifted) < 0.25


def test_psi_handles_empty_actual():
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    actual = np.array([])
    assert np.isnan(compute_psi(baseline, actual))


def test_classify_psi_thresholds():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate shift"
    assert classify_psi(0.30) == "significant shift"
    assert classify_psi(float("nan")) == "insufficient data"


def test_ks_statistic_perfect_separation():
    y_true = np.array([0] * 100 + [1] * 100)
    y_score = np.array([0.1] * 100 + [0.9] * 100)
    assert compute_ks_statistic(y_true, y_score) == pytest.approx(1.0)


def test_ks_statistic_near_zero_for_uninformative_scores():
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 500 + [1] * 500)
    y_score = rng.uniform(size=1000)  # random, unrelated to y_true
    assert compute_ks_statistic(y_true, y_score) < 0.15


def test_reliability_curve_shape_and_calibration_direction():
    rng = np.random.default_rng(42)
    n = 2000
    y_score = rng.uniform(size=n)
    y_true = (rng.uniform(size=n) < y_score).astype(int)  # well-calibrated by construction
    curve = reliability_curve(y_true, y_score, n_bins=10)
    assert len(curve) <= 10
    assert curve["count"].sum() == n
    assert np.corrcoef(curve["predicted_pd"], curve["empirical_default_rate"])[0, 1] > 0.8
