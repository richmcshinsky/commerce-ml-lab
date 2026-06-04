"""Forecasting models: baselines, ETS, and LightGBM global model.

Model progression (each builds on the last):

1. ``NaiveForecaster``         -- last observed value (floor baseline)
2. ``SeasonalNaiveForecaster`` -- same day from prior week (strong retail baseline)
3. ``MovingAverageForecaster`` -- rolling mean of last N days

All models share the same interface::

    model = SeasonalNaiveForecaster(seasonality=7)
    model.fit(train_df, id_col="id", date_col="date", sales_col="sales")
    preds = model.predict(test_df, id_col="id", date_col="date")
    # preds is a DataFrame with the same index as test_df, plus a "forecast" column

Design notes
------------
- All models operate on **long-format** DataFrames (one row per SKU-date).
- Forecasts are clipped to >= 0 (sales cannot be negative).
- Models are vectorised across all SKUs simultaneously -- no Python loops
  over individual series at inference time.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Base class ────────────────────────────────────────────────────────────────


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models.

    Subclasses must implement ``fit`` and ``predict``.
    The interface is intentionally minimal: long-format DataFrames in and out.
    """

    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
        sales_col: str = "sales",
    ) -> BaseForecaster:
        """Fit the model on training data.

        Parameters
        ----------
        train : pd.DataFrame
            Long-format training data. Must be sorted by (id, date).
        id_col : str
            Column identifying each time series.
        date_col : str
            Date column.
        sales_col : str
            Target column.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(
        self,
        test: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Generate point forecasts for the test period.

        Parameters
        ----------
        test : pd.DataFrame
            Long-format test data. Must contain id_col and date_col.
        id_col : str
        date_col : str

        Returns
        -------
        pd.DataFrame
            Same rows as ``test``, with an additional ``"forecast"`` column
            (float64, clipped to >= 0).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for tables and plots."""


# ── Naive baseline ────────────────────────────────────────────────────────────


class NaiveForecaster(BaseForecaster):
    """Naive forecaster: repeat the last observed value for all future dates.

    This is the absolute floor baseline. Any model worth deploying must
    beat it on WMAPE. It performs surprisingly well on stable, low-noise
    series but fails badly on seasonal data.

    Attributes
    ----------
    last_values_ : pd.Series
        Index = id, values = last observed sales from training data.
    """

    def __init__(self) -> None:
        self.last_values_: pd.Series | None = None

    @property
    def name(self) -> str:
        return "Naive"

    def fit(
        self,
        train: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
        sales_col: str = "sales",
    ) -> NaiveForecaster:
        """Store the last observed sales value for each series.

        Parameters
        ----------
        train : pd.DataFrame
        id_col, date_col, sales_col : str

        Returns
        -------
        self
        """
        # Sort to ensure we pick the truly last observation
        last_obs = (
            train.sort_values([id_col, date_col])
            .groupby(id_col)[sales_col]
            .last()
        )
        self.last_values_ = last_obs
        logger.info("NaiveForecaster fitted on %d series", len(last_obs))
        return self

    def predict(
        self,
        test: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Predict: broadcast last training value to all test rows.

        Parameters
        ----------
        test : pd.DataFrame
        id_col, date_col : str

        Returns
        -------
        pd.DataFrame
            ``test`` with column ``"forecast"`` appended.
        """
        if self.last_values_ is None:
            raise RuntimeError("Call fit() before predict().")

        result = test.copy()
        result["forecast"] = (
            result[id_col]
            .map(self.last_values_)
            .clip(lower=0)
            .astype(float)
        )
        return result


# ── Seasonal naive ────────────────────────────────────────────────────────────


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal naive forecaster: use the same day from k periods ago.

    For daily retail data with weekly seasonality (``seasonality=7``),
    this forecasts Monday's sales using last Monday's sales, etc.
    This is typically a much stronger baseline than the plain naive
    forecaster because it captures the within-week pattern.

    Parameters
    ----------
    seasonality : int
        Number of periods in one seasonal cycle. Default 7 (weekly).

    Attributes
    ----------
    history_ : pd.DataFrame
        Tail of training data needed for look-back (``seasonality`` rows
        per series).
    """

    def __init__(self, seasonality: int = 7) -> None:
        self.seasonality = seasonality
        self.history_: pd.DataFrame | None = None
        self._id_col: str = "id"
        self._date_col: str = "date"
        self._sales_col: str = "sales"

    @property
    def name(self) -> str:
        return f"SeasonalNaive(m={self.seasonality})"

    def fit(
        self,
        train: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
        sales_col: str = "sales",
    ) -> SeasonalNaiveForecaster:
        """Store the tail of training data (last seasonality * max_horizon rows).

        We store the full training tail rather than just the last
        ``seasonality`` rows, because the predict horizon may exceed one
        seasonal cycle.

        Parameters
        ----------
        train : pd.DataFrame
        id_col, date_col, sales_col : str

        Returns
        -------
        self
        """
        self._id_col = id_col
        self._date_col = date_col
        self._sales_col = sales_col

        # Keep enough history to cover a forecast horizon (up to 365 days)
        max_lookback = self.seasonality * 53  # ~1 year
        self.history_ = (
            train.sort_values([id_col, date_col])
            .groupby(id_col, group_keys=False)
            .tail(max_lookback)
            .copy()
        )
        logger.info(
            "SeasonalNaiveForecaster(m=%d) fitted on %d series",
            self.seasonality,
            train[id_col].nunique(),
        )
        return self

    def predict(
        self,
        test: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Generate forecasts using the seasonal look-back pattern.

        For each test date, looks back ``ceil(gap / seasonality) * seasonality``
        days into training history, where ``gap = (test_date - last_train_date).days``.
        Rounding up to the nearest multiple of seasonality ensures the look-back
        always lands on a training date regardless of the forecast horizon.

        Example (seasonality=7, 28-day horizon):
            days 1–7  → look back 7d  (1 cycle)
            days 8–14 → look back 14d (2 cycles)
            days 15–21 → look back 21d (3 cycles)
            days 22–28 → look back 28d (4 cycles)

        Parameters
        ----------
        test : pd.DataFrame
        id_col, date_col : str

        Returns
        -------
        pd.DataFrame
            ``test`` with column ``"forecast"`` appended. NaN where no
            look-back value exists (series too short).
        """
        if self.history_ is None:
            raise RuntimeError("Call fit() before predict().")

        test = test.copy()

        # Build history lookup table: (id, date) -> sales
        history = self.history_[[id_col, self._date_col, self._sales_col]].copy()
        history = history.rename(columns={self._sales_col: "forecast"})
        history[self._date_col] = pd.to_datetime(history[self._date_col])

        # Deduplicate: if multiple training rows share (id, date), keep the last
        history = history.drop_duplicates(subset=[id_col, self._date_col], keep="last")

        # Last training date per series — used to compute how far ahead we forecast
        last_train_date: pd.Series = history.groupby(id_col)[self._date_col].max()

        # Vectorised look-back: ceil(gap / seasonality) * seasonality days back.
        # This always lands on a training date for any horizon up to history length.
        test_dates = pd.to_datetime(test[date_col])
        last_dates = pd.to_datetime(test[id_col].map(last_train_date))
        gap_days = (test_dates - last_dates).dt.days.clip(lower=1)
        lookback = (
            np.ceil(gap_days.values / self.seasonality).astype(int) * self.seasonality
        )
        test["_lookup_date"] = test_dates - pd.to_timedelta(lookback, unit="D")

        result = test.merge(
            history.rename(columns={self._date_col: "_lookup_date"}),
            on=[id_col, "_lookup_date"],
            how="left",
        )
        result = result.drop(columns=["_lookup_date"])
        result["forecast"] = np.clip(result["forecast"], 0, None)
        return result


# ── Moving average ─────────────────────────────────────────────────────────────


class MovingAverageForecaster(BaseForecaster):
    """Moving average forecaster: mean of the last ``window`` training days.

    Smoother than the naive baseline; less reactive than seasonal naive.
    Works well when there is no strong seasonality but helps reduce noise.

    Parameters
    ----------
    window : int
        Number of trailing days to average. Common values: 7, 14, 28.
    """

    def __init__(self, window: int = 28) -> None:
        self.window = window
        self._means: pd.Series | None = None

    @property
    def name(self) -> str:
        return f"MovingAverage({self.window}d)"

    def fit(
        self,
        train: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
        sales_col: str = "sales",
    ) -> MovingAverageForecaster:
        """Compute the trailing mean for each series.

        Parameters
        ----------
        train : pd.DataFrame
        id_col, date_col, sales_col : str

        Returns
        -------
        self
        """
        means = (
            train.sort_values([id_col, date_col])
            .groupby(id_col, group_keys=False)
            .tail(self.window)
            .groupby(id_col)[sales_col]
            .mean()
        )
        self._means = means
        logger.info(
            "MovingAverageForecaster(w=%d) fitted on %d series",
            self.window,
            len(means),
        )
        return self

    def predict(
        self,
        test: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Predict: broadcast trailing mean to all test rows.

        Parameters
        ----------
        test : pd.DataFrame
        id_col, date_col : str

        Returns
        -------
        pd.DataFrame
            ``test`` with column ``"forecast"`` appended.
        """
        if self._means is None:
            raise RuntimeError("Call fit() before predict().")

        result = test.copy()
        result["forecast"] = (
            result[id_col]
            .map(self._means)
            .clip(lower=0)
            .astype(float)
        )
        return result


# ── Convenience: evaluate all baselines ──────────────────────────────────────


def evaluate_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
) -> pd.DataFrame:
    """Fit and evaluate all three baselines, returning a metrics summary.

    Parameters
    ----------
    train, test : pd.DataFrame
        Long-format train/test DataFrames.
    id_col, date_col, sales_col : str

    Returns
    -------
    pd.DataFrame
        One row per model with columns: ``model``, ``wmape``, ``mase``, ``rmsse``.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[5] / "src"))
    from commerce_ml.evaluation.forecast_metrics import summarise_forecast_metrics

    baselines: list[BaseForecaster] = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(seasonality=7),
        MovingAverageForecaster(window=28),
    ]

    # Aggregate to store level for scalar metric computation
    actual_agg = test.groupby(date_col)[sales_col].sum().values
    train_agg = train.groupby(date_col)[sales_col].sum().values

    # Seasonal naive forecast at the aggregated level (for MASE denominator)
    sn = SeasonalNaiveForecaster(seasonality=7)
    sn.fit(train, id_col, date_col, sales_col)
    sn_preds = sn.predict(test, id_col, date_col)
    sn_agg = sn_preds.groupby(date_col)["forecast"].sum().values

    rows = []
    for model in baselines:
        model.fit(train, id_col, date_col, sales_col)
        preds = model.predict(test, id_col, date_col)
        forecast_agg = preds.groupby(date_col)["forecast"].sum().values

        summary = summarise_forecast_metrics(
            actual=actual_agg,
            forecast=forecast_agg,
            train_actual=train_agg,
            naive_forecast=sn_agg,
            label=model.name,
        )
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)
