"""Tests for commerce_ml.evaluation.forecast_metrics.

Tests are written against known values so they catch regressions in the
metric implementations, not just "the function ran without error."
"""

import numpy as np
import pytest

from commerce_ml.evaluation.forecast_metrics import (
    mase,
    pinball_loss,
    rmsse,
    summarise_forecast_metrics,
    wmape,
)


class TestWmape:
    def test_perfect_forecast(self) -> None:
        actual = np.array([10.0, 20.0, 30.0])
        assert wmape(actual, actual) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7
        # sum(actual) = 60
        # wmape = 7/60
        actual = np.array([10.0, 20.0, 30.0])
        forecast = np.array([12.0, 18.0, 33.0])
        assert wmape(actual, forecast) == pytest.approx(7 / 60)

    def test_zero_actual_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            wmape(np.array([0.0, 0.0]), np.array([1.0, 2.0]))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            wmape(np.array([1.0, 2.0]), np.array([1.0]))

    def test_accepts_pandas_series(self) -> None:
        import pandas as pd

        actual = pd.Series([10.0, 20.0, 30.0])
        forecast = pd.Series([10.0, 20.0, 30.0])
        assert wmape(actual, forecast) == pytest.approx(0.0)

    def test_handles_zeros_in_actual(self) -> None:
        # Zero-sale days are common in retail — WMAPE should handle them
        actual = np.array([0.0, 10.0, 20.0])
        forecast = np.array([5.0, 10.0, 20.0])
        # |0-5| + |10-10| + |20-20| = 5 / 30
        assert wmape(actual, forecast) == pytest.approx(5 / 30)


class TestMase:
    def test_better_than_naive_is_below_one(self) -> None:
        actual = np.array([10.0, 11.0, 12.0, 13.0])
        perfect = np.array([10.0, 11.0, 12.0, 13.0])
        naive = np.array([9.0, 10.0, 11.0, 12.0])  # shift by 1
        assert mase(actual, perfect, naive) == pytest.approx(0.0)

    def test_naive_reference_equals_one(self) -> None:
        actual = np.array([10.0, 11.0, 12.0, 13.0])
        naive = np.array([9.0, 10.0, 11.0, 12.0])
        # If forecast == naive, MASE should be 1.0
        assert mase(actual, naive, naive) == pytest.approx(1.0)

    def test_zero_naive_mae_and_zero_model_mae(self) -> None:
        actual = np.array([5.0, 5.0, 5.0])
        naive = np.array([5.0, 5.0, 5.0])  # naive is perfect
        assert mase(actual, actual, naive) == pytest.approx(0.0)


class TestPinballLoss:
    def test_perfect_quantile_forecast(self) -> None:
        # If forecast equals actual, loss should be 0 regardless of quantile
        actual = np.array([1.0, 2.0, 3.0])
        assert pinball_loss(actual, actual, quantile=0.9) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # actual=[1,2,3], forecast=[1.5,1.5,2.5], q=0.9
        # errors = [-0.5, 0.5, 0.5]
        # loss = (1-0.9)*0.5 + 0.9*0.5 + 0.9*0.5 = 0.05 + 0.45 + 0.45 = 0.95 / 3
        actual = np.array([1.0, 2.0, 3.0])
        forecast = np.array([1.5, 1.5, 2.5])
        expected = (0.1 * 0.5 + 0.9 * 0.5 + 0.9 * 0.5) / 3
        assert pinball_loss(actual, forecast, quantile=0.9) == pytest.approx(expected)

    def test_invalid_quantile_raises(self) -> None:
        with pytest.raises(ValueError, match="quantile"):
            pinball_loss(np.array([1.0]), np.array([1.0]), quantile=0.0)

    def test_over_forecast_penalised_asymmetrically(self) -> None:
        # At q=0.9, over-forecasting (forecast > actual) is penalised more
        # lightly than under-forecasting
        actual = np.array([10.0])
        over_forecast = np.array([15.0])
        under_forecast = np.array([5.0])
        loss_over = pinball_loss(actual, over_forecast, quantile=0.9)
        loss_under = pinball_loss(actual, under_forecast, quantile=0.9)
        assert loss_under > loss_over  # under-forecast is more costly at q=0.9


class TestRmsse:
    def test_perfect_forecast_is_zero(self) -> None:
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([6.0, 7.0, 8.0])
        assert rmsse(actual, actual, train, seasonality=1) == pytest.approx(0.0)

    def test_returns_positive_float(self) -> None:
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0])
        actual = np.array([5.0, 6.0, 4.0])
        forecast = np.array([4.5, 5.5, 4.5])
        result = rmsse(actual, forecast, train, seasonality=1)
        assert result > 0.0
        assert isinstance(result, float)


class TestSummariseMetrics:
    def test_returns_dataframe_with_expected_columns(self) -> None:
        actual = np.array([10.0, 20.0, 30.0])
        forecast = np.array([11.0, 19.0, 31.0])
        train = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        naive = np.array([9.0, 10.0, 11.0])
        df = summarise_forecast_metrics(actual, forecast, train, naive, label="test_model")
        assert list(df.columns) == ["model", "wmape", "mase", "rmsse"]
        assert df["model"].iloc[0] == "test_model"
        assert len(df) == 1
