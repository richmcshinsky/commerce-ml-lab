"""Tests for forecasting/data.py — M5 data pipeline.

Uses the synthetic M5 fixture from conftest.py so tests run without
downloading the real 450 MB M5 dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from forecasting.data import (
    aggregate_by_date,
    get_active_items,
    make_series_pivot,
    train_test_split,
    zero_rate,
)


class TestTrainTestSplit:
    def test_test_window_has_correct_days(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        train, test = train_test_split(synthetic_m5_long, test_days=28)
        assert test["date"].nunique() == 28

    def test_no_overlap_between_splits(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        train, test = train_test_split(synthetic_m5_long, test_days=28)
        train_dates = set(train["date"].dt.date)
        test_dates = set(test["date"].dt.date)
        assert len(train_dates & test_dates) == 0

    def test_train_before_test(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        train, test = train_test_split(synthetic_m5_long, test_days=28)
        assert train["date"].max() < test["date"].min()

    def test_total_rows_preserved(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        train, test = train_test_split(synthetic_m5_long, test_days=28)
        assert len(train) + len(test) == len(synthetic_m5_long)

    def test_all_skus_in_test(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        train, test = train_test_split(synthetic_m5_long, test_days=28)
        assert set(test["id"]) == set(synthetic_m5_long["id"])


class TestZeroRate:
    def test_returns_float_in_range(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        rate = zero_rate(synthetic_m5_long)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_all_zeros_gives_one(self) -> None:
        df = pd.DataFrame({"sales": [0, 0, 0]})
        assert zero_rate(df) == pytest.approx(1.0)

    def test_no_zeros_gives_zero(self) -> None:
        df = pd.DataFrame({"sales": [1, 2, 3]})
        assert zero_rate(df) == pytest.approx(0.0)


class TestAggregateByDate:
    def test_returns_one_row_per_date(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        agg = aggregate_by_date(synthetic_m5_long)
        n_dates = synthetic_m5_long["date"].nunique()
        assert len(agg) == n_dates

    def test_group_aggregation_has_correct_rows(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        agg = aggregate_by_date(synthetic_m5_long, group_col="cat_id")
        n_dates = synthetic_m5_long["date"].nunique()
        n_cats = synthetic_m5_long["cat_id"].nunique()
        assert len(agg) == n_dates * n_cats

    def test_total_sales_preserved(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        agg = aggregate_by_date(synthetic_m5_long)
        assert agg["sales"].sum() == synthetic_m5_long["sales"].sum()


class TestMakeSeriesPivot:
    def test_pivot_shape(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        wide = make_series_pivot(synthetic_m5_long)
        n_dates = synthetic_m5_long["date"].nunique()
        n_series = synthetic_m5_long["id"].nunique()
        assert wide.shape == (n_dates, n_series)

    def test_pivot_has_datetime_index(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        wide = make_series_pivot(synthetic_m5_long)
        assert isinstance(wide.index, pd.DatetimeIndex)

    def test_pivot_no_nan(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        wide = make_series_pivot(synthetic_m5_long)
        assert wide.isna().sum().sum() == 0


class TestGetActiveItems:
    def test_filters_sparse_items(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        # All items should have at least 1 non-zero day in 120-day synthetic data
        active = get_active_items(synthetic_m5_long, min_nonzero_days=1)
        assert len(active) > 0

    def test_high_threshold_returns_fewer_items(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        active_loose = get_active_items(synthetic_m5_long, min_nonzero_days=5)
        active_strict = get_active_items(synthetic_m5_long, min_nonzero_days=60)
        assert len(active_strict) <= len(active_loose)

    def test_returns_list_of_strings(
        self, synthetic_m5_long: pd.DataFrame
    ) -> None:
        active = get_active_items(synthetic_m5_long)
        assert isinstance(active, list)
        if active:
            assert isinstance(active[0], str)
