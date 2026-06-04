"""Tests for forecasting/models.py — baseline forecasters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from forecasting.models import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    evaluate_baselines,
)


class TestNaiveForecaster:
    def test_forecast_equals_last_training_value(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        model = NaiveForecaster()
        model.fit(train)
        preds = model.predict(test)

        # For each SKU, the forecast should equal the last training sales value
        for sku_id, _group in test.groupby("id"):
            expected = float(train[train["id"] == sku_id].sort_values("date")["sales"].iloc[-1])
            forecasts = preds[preds["id"] == sku_id]["forecast"]
            assert np.allclose(forecasts.values, expected), (
                f"SKU {sku_id}: expected {expected}, got {forecasts.values}"
            )

    def test_forecast_column_added(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        preds = NaiveForecaster().fit(train).predict(test)
        assert "forecast" in preds.columns

    def test_no_negative_forecasts(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        preds = NaiveForecaster().fit(train).predict(test)
        assert (preds["forecast"] >= 0).all()

    def test_predict_without_fit_raises(self, small_train_test: tuple) -> None:
        _, test = small_train_test
        with pytest.raises(RuntimeError, match="fit()"):
            NaiveForecaster().predict(test)

    def test_output_length_matches_test(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        preds = NaiveForecaster().fit(train).predict(test)
        assert len(preds) == len(test)


class TestSeasonalNaiveForecaster:
    def test_no_negative_forecasts(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        preds = SeasonalNaiveForecaster(7).fit(train).predict(test)
        # NaN is OK (series too short), but non-NaN values must be >= 0
        non_nan = preds["forecast"].dropna()
        assert (non_nan >= 0).all()

    def test_output_length_matches_test(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        preds = SeasonalNaiveForecaster(7).fit(train).predict(test)
        assert len(preds) == len(test)

    def test_forecast_uses_7_day_lookback(self, small_train_test: tuple) -> None:
        """Verify the seasonal naive uses exactly m=7 days lookback."""
        train, test = small_train_test
        model = SeasonalNaiveForecaster(seasonality=7)
        model.fit(train)
        preds = model.predict(test)

        # For each (id, test_date), the forecast should match train sales
        # exactly 7 days prior
        import pandas as pd

        train_lookup = train.set_index(["id", "date"])["sales"]

        for _, row in preds.dropna(subset=["forecast"]).head(20).iterrows():
            target_date = pd.to_datetime(row["date"]) - pd.Timedelta(days=7)
            key = (row["id"], target_date)
            if key in train_lookup.index:
                expected = float(train_lookup[key])
                assert row["forecast"] == pytest.approx(expected), (
                    f"id={row['id']}, date={row['date']}: "
                    f"expected {expected}, got {row['forecast']}"
                )

    def test_predict_without_fit_raises(self, small_train_test: tuple) -> None:
        _, test = small_train_test
        with pytest.raises(RuntimeError, match="fit()"):
            SeasonalNaiveForecaster(7).predict(test)


class TestMovingAverageForecaster:
    def test_forecast_equals_trailing_mean(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        window = 14
        model = MovingAverageForecaster(window=window)
        model.fit(train)
        preds = model.predict(test)

        # For one SKU, verify the forecast matches the trailing mean
        sku_id = test["id"].iloc[0]
        expected_mean = float(
            train[train["id"] == sku_id].sort_values("date")["sales"].tail(window).mean()
        )
        actual_forecasts = preds[preds["id"] == sku_id]["forecast"].values
        assert np.allclose(actual_forecasts, expected_mean)

    def test_no_negative_forecasts(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        preds = MovingAverageForecaster(28).fit(train).predict(test)
        assert (preds["forecast"] >= 0).all()

    def test_different_windows_give_different_forecasts(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        p7 = MovingAverageForecaster(7).fit(train).predict(test)
        p28 = MovingAverageForecaster(28).fit(train).predict(test)
        # With seasonal data, 7-day and 28-day windows should differ
        assert not (p7["forecast"].values == p28["forecast"].values).all()

    def test_predict_without_fit_raises(self, small_train_test: tuple) -> None:
        _, test = small_train_test
        with pytest.raises(RuntimeError, match="fit()"):
            MovingAverageForecaster(7).predict(test)


class TestEvaluateBaselines:
    def test_returns_dataframe_with_all_models(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        metrics = evaluate_baselines(train, test)
        assert "model" in metrics.columns
        assert "wmape" in metrics.columns
        model_names = set(metrics["model"].values)
        assert "Naive" in model_names
        assert "SeasonalNaive(m=7)" in model_names
        assert "MovingAverage(28d)" in model_names

    def test_wmape_values_are_positive(self, small_train_test: tuple) -> None:
        train, test = small_train_test
        metrics = evaluate_baselines(train, test)
        assert (metrics["wmape"] >= 0).all()
