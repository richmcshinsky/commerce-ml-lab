"""Tests for forecasting/features.py and forecasting/lgbm_model.py.

Tests are designed to run without lightgbm installed where possible.
The LGBMForecaster tests are skipped when lightgbm is unavailable, so
the test suite remains green in CI environments that haven't installed
the optional ML dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from forecasting.features import (
    MIN_LAG,
    add_calendar_features,
    add_lag_features,
    add_price_features,
    add_rolling_features,
    build_lgbm_features,
    encode_categoricals,
    get_feature_columns,
)

# Short lags used in all tests — the 120-day synthetic fixture can't satisfy lag_364.
# In production (real M5 data: 5+ years), the full LAG_DAYS list is used.
_TEST_LAGS: list[int] = [7, 14, 28]
_TEST_WINDOWS: list[int] = [7, 14]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_df() -> pd.DataFrame:
    """Minimal long-format DataFrame: 3 SKUs × 60 days."""
    rng = np.random.default_rng(0)
    skus = ["A", "B", "C"]
    dates = pd.date_range("2023-01-01", periods=60)
    rows = [
        {"id": sku, "date": d, "sales": int(rng.poisson(3.0)), "sell_price": 9.99}
        for sku in skus
        for d in dates
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def lgbm_train_test(synthetic_m5_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split of the synthetic M5 fixture for LightGBM tests."""
    from forecasting.data import train_test_split
    return train_test_split(synthetic_m5_long, test_days=28)


# ── Calendar features ─────────────────────────────────────────────────────────


class TestCalendarFeatures:
    def test_adds_day_of_week(self, small_df: pd.DataFrame) -> None:
        result = add_calendar_features(small_df)
        assert "day_of_week" in result.columns

    def test_day_of_week_range(self, small_df: pd.DataFrame) -> None:
        result = add_calendar_features(small_df)
        assert result["day_of_week"].between(0, 6).all()

    def test_is_weekend_correct(self) -> None:
        # 2023-01-07 is Sat, 2023-01-08 is Sun
        df = pd.DataFrame({"date": pd.to_datetime(["2023-01-06", "2023-01-07", "2023-01-08"])})
        result = add_calendar_features(df)
        assert result["is_weekend"].tolist() == [0, 1, 1]

    def test_fourier_columns_present(self, small_df: pd.DataFrame) -> None:
        result = add_calendar_features(small_df, include_fourier=True, fourier_order=2)
        assert "fourier_weekly_sin_1" in result.columns
        assert "fourier_annual_cos_2" in result.columns

    def test_no_fourier_when_disabled(self, small_df: pd.DataFrame) -> None:
        result = add_calendar_features(small_df, include_fourier=False)
        assert not any("fourier" in c for c in result.columns)

    def test_original_columns_preserved(self, small_df: pd.DataFrame) -> None:
        result = add_calendar_features(small_df)
        assert "date" in result.columns
        assert "sales" in result.columns


# ── Lag features ──────────────────────────────────────────────────────────────


class TestLagFeatures:
    def test_lag_columns_created(self, small_df: pd.DataFrame) -> None:
        result = add_lag_features(small_df, lags=_TEST_LAGS)
        assert "sales_lag_7" in result.columns
        assert "sales_lag_14" in result.columns

    def test_lag_7_correct_value(self, small_df: pd.DataFrame) -> None:
        """lag_7 at row k should equal sales at row k-7 within the same SKU."""
        df = small_df[small_df["id"] == "A"].reset_index(drop=True)
        result = add_lag_features(df, lags=[7], id_col="id")
        for i in range(7, len(df)):
            expected = df.loc[i - 7, "sales"]
            assert result.loc[i, "sales_lag_7"] == expected, f"Row {i}: expected {expected}"

    def test_lag_boundary_is_nan(self, small_df: pd.DataFrame) -> None:
        """First lag rows of each SKU should be NaN."""
        result = add_lag_features(small_df, lags=_TEST_LAGS)
        for sku, group in result.groupby("id"):
            first_7 = group.head(7)["sales_lag_7"]
            assert first_7.isna().all(), f"SKU {sku}: expected NaN in first 7 rows"

    def test_no_cross_group_leakage(self, small_df: pd.DataFrame) -> None:
        """First lag value for group B should not equal last value of group A."""
        result = add_lag_features(small_df, lags=[1])
        group_b_first = result[result["id"] == "B"].iloc[0]["sales_lag_1"]
        assert np.isnan(group_b_first), "First lag of group B should be NaN, not group A's last value"


# ── Rolling features ──────────────────────────────────────────────────────────


class TestRollingFeatures:
    def test_roll_columns_created(self, small_df: pd.DataFrame) -> None:
        result = add_rolling_features(small_df, windows=_TEST_WINDOWS)
        assert "sales_roll_7_mean" in result.columns
        assert "sales_roll_14_mean" in result.columns

    def test_no_lookahead_with_default_shift(self, small_df: pd.DataFrame) -> None:
        """Rolling mean at row i must not use values from rows i-6…i.

        With shift=MIN_LAG=7 and window=7:
          .shift(7).rolling(7).mean() at row i
          = mean of shifted[i-6..i]
          = mean of original[i-13..i-7]  (all at least 7 periods in the past)
        """
        df = small_df[small_df["id"] == "A"].reset_index(drop=True).copy()
        result = add_rolling_features(df, windows=[7], id_col="id", shift=MIN_LAG)

        # At row 20: original[7..13] inclusive = 7 values
        expected = df.loc[7:13, "sales"].mean()
        actual = result.loc[20, "sales_roll_7_mean"]
        assert actual == pytest.approx(expected, abs=1e-9)

    def test_no_cross_group_leakage(self, small_df: pd.DataFrame) -> None:
        """Rolling features for the first rows of group B must not include group A values."""
        result = add_rolling_features(small_df, windows=[7])
        group_b = result[result["id"] == "B"].reset_index(drop=True)
        # The very first row of B has no history — should be NaN
        assert np.isnan(group_b.loc[0, "sales_roll_7_mean"])


# ── Price features ────────────────────────────────────────────────────────────


class TestPriceFeatures:
    def test_price_columns_created(self, small_df: pd.DataFrame) -> None:
        result = add_price_features(small_df)
        assert "price_change_pct" in result.columns
        assert "is_price_discount" in result.columns

    def test_missing_price_col_returns_unchanged(self, small_df: pd.DataFrame) -> None:
        df = small_df.drop(columns=["sell_price"])
        result = add_price_features(df, price_col="sell_price")
        assert "price_change_pct" not in result.columns

    def test_discount_flag_fires_on_lower_price(self) -> None:
        df = pd.DataFrame({
            "id": ["X"] * 5,
            "sell_price": [10.0, 10.0, 10.0, 10.0, 8.0],  # last row is discounted
        })
        result = add_price_features(df, price_col="sell_price", id_col="id", ref_window=4)
        # At row 4, ref_price ≈ 10, current = 8 → discount
        assert result.loc[4, "is_price_discount"] == 1


# ── Categorical encoding ──────────────────────────────────────────────────────


class TestEncodeCategoricals:
    def test_dtype_becomes_category(self, synthetic_m5_long: pd.DataFrame) -> None:
        result = encode_categoricals(synthetic_m5_long, cat_cols=["item_id", "cat_id"])
        assert result["item_id"].dtype.name == "category"
        assert result["cat_id"].dtype.name == "category"

    def test_missing_cols_skipped_silently(self, small_df: pd.DataFrame) -> None:
        # small_df has no item_id column — should not raise
        result = encode_categoricals(small_df, cat_cols=["item_id", "id"])
        assert "item_id" not in result.columns


# ── build_lgbm_features pipeline ─────────────────────────────────────────────


class TestBuildLGBMFeatures:
    def test_returns_fewer_rows_due_to_lag_nans(self, synthetic_m5_long: pd.DataFrame) -> None:
        # With lag_28 and 120 days data, rows prior to day 28 are dropped
        result = build_lgbm_features(
            synthetic_m5_long, lags=_TEST_LAGS, rolling_windows=_TEST_WINDOWS, drop_na_rows=True
        )
        assert len(result) < len(synthetic_m5_long)
        assert len(result) > 0  # some rows survive

    def test_lag_columns_present(self, synthetic_m5_long: pd.DataFrame) -> None:
        result = build_lgbm_features(
            synthetic_m5_long, lags=_TEST_LAGS, rolling_windows=_TEST_WINDOWS, drop_na_rows=False
        )
        for lag in _TEST_LAGS:
            assert f"sales_lag_{lag}" in result.columns

    def test_rolling_columns_present(self, synthetic_m5_long: pd.DataFrame) -> None:
        result = build_lgbm_features(
            synthetic_m5_long, lags=_TEST_LAGS, rolling_windows=_TEST_WINDOWS, drop_na_rows=False
        )
        for w in _TEST_WINDOWS:
            assert f"sales_roll_{w}_mean" in result.columns

    def test_price_features_added_when_col_present(self, synthetic_m5_long: pd.DataFrame) -> None:
        result = build_lgbm_features(
            synthetic_m5_long, lags=_TEST_LAGS, rolling_windows=_TEST_WINDOWS,
            price_col="sell_price", drop_na_rows=False,
        )
        assert "price_change_pct" in result.columns

    def test_get_feature_columns_excludes_target(self, synthetic_m5_long: pd.DataFrame) -> None:
        feat_df = build_lgbm_features(
            synthetic_m5_long, lags=_TEST_LAGS, rolling_windows=_TEST_WINDOWS, drop_na_rows=False
        )
        feat_cols = get_feature_columns(feat_df)
        assert "sales" not in feat_cols
        assert "date" not in feat_cols


# ── LGBMForecaster ─────────────────────────────────────────────────────────────

lgbm = pytest.importorskip("lightgbm", reason="lightgbm not installed")


_FAST_PARAMS = {"n_estimators": 10, "verbose": -1, "random_state": 42}
"""Minimal hyperparameters for fast unit tests."""

_TEST_MODEL_KWARGS: dict = {
    "lgbm_params": _FAST_PARAMS,
    "lags": _TEST_LAGS,
    "rolling_windows": _TEST_WINDOWS,
}
"""Common kwargs for LGBMForecaster in tests to avoid the lag_364 issue on 120-day data."""


class TestLGBMForecaster:
    def test_fit_predict_runs(self, lgbm_train_test: tuple) -> None:
        from forecasting.lgbm_model import LGBMForecaster
        train, test = lgbm_train_test
        model = LGBMForecaster(**_TEST_MODEL_KWARGS)
        model.fit(train)
        preds = model.predict(test)
        assert "forecast" in preds.columns

    def test_forecast_length_matches_test(self, lgbm_train_test: tuple) -> None:
        from forecasting.lgbm_model import LGBMForecaster
        train, test = lgbm_train_test
        preds = LGBMForecaster(**_TEST_MODEL_KWARGS).fit(train).predict(test)
        assert len(preds) == len(test)

    def test_no_negative_forecasts(self, lgbm_train_test: tuple) -> None:
        from forecasting.lgbm_model import LGBMForecaster
        train, test = lgbm_train_test
        preds = LGBMForecaster(**_TEST_MODEL_KWARGS).fit(train).predict(test)
        assert (preds["forecast"] >= 0).all()

    def test_interval_ordering(self, lgbm_train_test: tuple) -> None:
        """lower_80 <= forecast <= upper_80 must hold for every row."""
        from forecasting.lgbm_model import LGBMForecaster
        train, test = lgbm_train_test
        model = LGBMForecaster(**_TEST_MODEL_KWARGS)
        model.fit(train)
        preds = model.predict_with_intervals(test)
        assert "lower_80" in preds.columns
        assert "upper_80" in preds.columns
        assert (preds["lower_80"] <= preds["forecast"] + 1e-9).all()
        assert (preds["upper_80"] >= preds["forecast"] - 1e-9).all()

    def test_predict_without_fit_raises(self, lgbm_train_test: tuple) -> None:
        from forecasting.lgbm_model import LGBMForecaster
        _, test = lgbm_train_test
        with pytest.raises(RuntimeError, match="fit()"):
            LGBMForecaster(**_TEST_MODEL_KWARGS).predict(test)

    def test_feature_importance_returns_dataframe(self, lgbm_train_test: tuple) -> None:
        from forecasting.lgbm_model import LGBMForecaster
        train, _ = lgbm_train_test
        model = LGBMForecaster(**_TEST_MODEL_KWARGS)
        model.fit(train)
        fi = model.get_feature_importance(top_n=5)
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) <= 5

    def test_save_and_load(self, lgbm_train_test: tuple, tmp_path: Path) -> None:
        from forecasting.lgbm_model import LGBMForecaster
        train, test = lgbm_train_test
        model = LGBMForecaster(**_TEST_MODEL_KWARGS)
        model.fit(train)

        save_path = tmp_path / "model.pkl"
        model.save(save_path)
        loaded = LGBMForecaster.load(save_path)

        preds_original = model.predict(test)
        preds_loaded = loaded.predict(test)
        pd.testing.assert_series_equal(preds_original["forecast"], preds_loaded["forecast"])
