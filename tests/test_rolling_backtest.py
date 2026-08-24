"""Unit tests for the walk-forward expanding-window splitter used in rolling OOT backtests.

Self-contained (synthetic in-memory data only) so this runs without the processed dataset
or trained model artifacts being present.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.rolling_backtest import expanding_windows


def _make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"issue_date": pd.date_range("2015-01-01", periods=n, freq="D"), "value": range(n)})


def test_expanding_windows_returns_requested_fold_count():
    df = _make_df(600)
    windows = expanding_windows(df, n_folds=5)
    assert len(windows) == 5


def test_expanding_windows_train_grows_each_fold():
    df = _make_df(600)
    windows = expanding_windows(df, n_folds=5)
    train_sizes = [len(train) for train, _ in windows]
    assert train_sizes == sorted(train_sizes)
    assert all(b > a for a, b in zip(train_sizes, train_sizes[1:]))


def test_expanding_windows_never_leak_future_into_train():
    df = _make_df(600)
    for train, test in expanding_windows(df, n_folds=5):
        assert train["issue_date"].max() < test["issue_date"].min()


def test_expanding_windows_test_segments_are_contiguous_and_non_overlapping():
    df = _make_df(600)
    windows = expanding_windows(df, n_folds=5)
    all_test_indices = pd.concat([test for _, test in windows])["value"]
    assert all_test_indices.is_monotonic_increasing
    assert all_test_indices.nunique() == len(all_test_indices)


def test_expanding_windows_covers_whole_timeline_across_folds():
    df = _make_df(600)
    windows = expanding_windows(df, n_folds=5)
    last_train, last_test = windows[-1]
    assert len(last_train) + len(last_test) == len(df)


@pytest.mark.parametrize("n_folds", [1, 3, 10])
def test_expanding_windows_handles_various_fold_counts(n_folds):
    df = _make_df(1000)
    windows = expanding_windows(df, n_folds=n_folds)
    assert len(windows) == n_folds
    for train, test in windows:
        assert len(train) > 0
        assert len(test) > 0
