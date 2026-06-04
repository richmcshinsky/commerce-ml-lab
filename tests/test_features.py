"""Tests for commerce_ml.features.time_features and session_features.

Tests focus on correctness and edge cases that would fail silently in
production: off-by-one in lag computation, NaN handling, group leakage.
"""

import numpy as np
import pandas as pd
import pytest

from commerce_ml.features.time_features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
)
from commerce_ml.features.session_features import build_session_features


class TestCalendarFeatures:
    def test_adds_expected_columns(self) -> None:
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10, freq="D")})
        result = add_calendar_features(df)
        expected_cols = [
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "week_of_year",
            "month",
            "quarter",
            "year",
            "is_weekend",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_is_weekend_correct(self) -> None:
        # 2023-01-07 is a Saturday, 2023-01-08 is a Sunday
        df = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09"])}
        )
        result = add_calendar_features(df)
        assert result["is_weekend"].tolist() == [0, 1, 1, 0]

    def test_fourier_columns_added_when_requested(self) -> None:
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5, freq="D")})
        result = add_calendar_features(df, include_fourier=True, fourier_order=2)
        assert "fourier_weekly_sin_1" in result.columns
        assert "fourier_annual_cos_2" in result.columns

    def test_no_fourier_when_disabled(self) -> None:
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5, freq="D")})
        result = add_calendar_features(df, include_fourier=False)
        assert not any("fourier" in c for c in result.columns)

    def test_original_columns_preserved(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "sales": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = add_calendar_features(df)
        assert "date" in result.columns
        assert "sales" in result.columns


class TestLagFeatures:
    def test_lag_1_correct_values(self) -> None:
        df = pd.DataFrame({"sales": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = add_lag_features(df, "sales", lags=[1])
        # Lag 1: first element is NaN, then shifted by 1
        assert np.isnan(result["sales_lag_1"].iloc[0])
        assert result["sales_lag_1"].iloc[1] == pytest.approx(10.0)
        assert result["sales_lag_1"].iloc[2] == pytest.approx(20.0)

    def test_multiple_lags(self) -> None:
        df = pd.DataFrame({"sales": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = add_lag_features(df, "sales", lags=[1, 2, 3])
        assert "sales_lag_1" in result.columns
        assert "sales_lag_2" in result.columns
        assert "sales_lag_3" in result.columns

    def test_group_lag_does_not_leak_across_groups(self) -> None:
        """Lag for group B should not see group A's values."""
        df = pd.DataFrame(
            {
                "sku": ["A", "A", "A", "B", "B", "B"],
                "sales": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            }
        )
        result = add_lag_features(df, "sales", lags=[1], group_col="sku")
        # First value of group B should be NaN (not 30 from group A)
        group_b = result[result["sku"] == "B"].reset_index(drop=True)
        assert np.isnan(group_b["sales_lag_1"].iloc[0])

    def test_original_column_unchanged(self) -> None:
        df = pd.DataFrame({"sales": [1.0, 2.0, 3.0]})
        result = add_lag_features(df, "sales", lags=[1])
        pd.testing.assert_series_equal(result["sales"], df["sales"])


class TestRollingFeatures:
    def test_adds_mean_and_std_by_default(self) -> None:
        df = pd.DataFrame({"sales": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = add_rolling_features(df, "sales", windows=[3])
        assert "sales_roll_3_mean" in result.columns
        assert "sales_roll_3_std" in result.columns

    def test_no_look_ahead(self) -> None:
        """Rolling mean at time t should not include time t itself."""
        df = pd.DataFrame({"sales": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = add_rolling_features(df, "sales", windows=[2], agg_funcs=["mean"])
        # At index 2, rolling mean of window 2 should use indices 0 and 1 (shift(1) applied)
        # shift(1) means: at index 2, we see [NaN, 10, 20] -> rolling(2) -> mean of [10, 20] = 15
        assert result["sales_roll_2_mean"].iloc[2] == pytest.approx(15.0)

    def test_group_rolling_does_not_leak(self) -> None:
        df = pd.DataFrame(
            {
                "sku": ["A", "A", "A", "B", "B", "B"],
                "sales": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            }
        )
        result = add_rolling_features(df, "sales", windows=[2], group_col="sku", agg_funcs=["mean"])
        group_b_mean = result[result["sku"] == "B"]["sales_roll_2_mean"]
        # No value from group B should exceed 200 (since max(B) is 300, rolling(2) of
        # shift(1) gives at most (100+200)/2=150 for index 2)
        assert group_b_mean.max() <= 200.0


class TestSessionFeatures:
    def _make_events(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "session_id": ["s1", "s1", "s1", "s1", "s2", "s2"],
                "event_type": [
                    "page_view",
                    "add_to_cart",
                    "page_view",
                    "purchase",
                    "page_view",
                    "page_view",
                ],
                "timestamp": [
                    "2023-01-01 10:00",
                    "2023-01-01 10:05",
                    "2023-01-01 10:10",
                    "2023-01-01 10:20",
                    "2023-01-01 11:00",
                    "2023-01-01 11:15",
                ],
                "price": [50.0, 50.0, 30.0, 50.0, 20.0, 40.0],
            }
        )

    def test_one_row_per_session(self) -> None:
        events = self._make_events()
        result = build_session_features(events)
        assert len(result) == 2
        assert set(result["session_id"]) == {"s1", "s2"}

    def test_conversion_label_correct(self) -> None:
        events = self._make_events()
        result = build_session_features(events).set_index("session_id")
        assert result.loc["s1", "converted"] == 1
        assert result.loc["s2", "converted"] == 0

    def test_page_view_counts_correct(self) -> None:
        events = self._make_events()
        result = build_session_features(events).set_index("session_id")
        assert result.loc["s1", "n_page_views"] == 2
        assert result.loc["s2", "n_page_views"] == 2

    def test_session_duration_positive(self) -> None:
        events = self._make_events()
        result = build_session_features(events).set_index("session_id")
        assert result.loc["s1", "session_duration_seconds"] == pytest.approx(1200.0)

    def test_price_features_computed(self) -> None:
        events = self._make_events()
        result = build_session_features(events, price_col="price").set_index("session_id")
        assert "price_max" in result.columns
        assert result.loc["s2", "price_max"] == pytest.approx(40.0)
