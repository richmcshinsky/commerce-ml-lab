"""Time-series feature engineering utilities.

Provides calendar features, lag features, and rolling statistics used by the
demand forecasting and returns forecasting models.

All functions operate on a pandas DataFrame with a ``date`` or datetime index
column and return a new DataFrame with additional feature columns appended.

Notes
-----
**Lag leakage warning:** Lag and rolling features must be computed within the
training split only, then applied to the test split using the same window
boundaries. Functions here do *not* enforce this — callers are responsible for
correct train/test handling. See ``commerce_ml.evaluation.backtest`` for a
walk-forward splitter that prevents leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
    *,
    include_fourier: bool = True,
    fourier_order: int = 3,
) -> pd.DataFrame:
    """Add calendar-based features to a time-series DataFrame.

    Adds: day of week, day of month, day of year, week of year, month, quarter,
    year, is_weekend, and optionally Fourier terms for weekly and annual cycles.

    Parameters
    ----------
    df:
        Input DataFrame. Must contain a datetime column named ``date_col``.
    date_col:
        Name of the datetime column.
    include_fourier:
        If ``True``, add sin/cos Fourier terms for weekly (period=7) and
        annual (period=365.25) seasonality. Useful for tree models that cannot
        natively capture seasonality.
    fourier_order:
        Number of Fourier harmonics to include per cycle.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional feature columns appended. The
        original ``date_col`` is preserved.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    df["day_of_week"] = dates.dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"] = dates.dt.day
    df["day_of_year"] = dates.dt.dayofyear
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["month"] = dates.dt.month
    df["quarter"] = dates.dt.quarter
    df["year"] = dates.dt.year
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

    if include_fourier:
        t = dates.dt.dayofyear.to_numpy(dtype=float)
        for k in range(1, fourier_order + 1):
            df[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * t / 7)
            df[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * t / 7)
            df[f"fourier_annual_sin_{k}"] = np.sin(2 * np.pi * k * t / 365.25)
            df[f"fourier_annual_cos_{k}"] = np.cos(2 * np.pi * k * t / 365.25)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add lag features for a target column.

    Parameters
    ----------
    df:
        Input DataFrame, sorted by date within each group.
    target_col:
        Column to lag.
    lags:
        List of lag periods, e.g. ``[1, 7, 14, 28]``.
    group_col:
        If provided, lags are computed within each group (e.g. SKU or store).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with columns ``{target_col}_lag_{k}`` appended.
    """
    df = df.copy()

    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        if group_col is not None:
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        else:
            df[col_name] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
    group_col: str | None = None,
    agg_funcs: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling window aggregation features.

    Parameters
    ----------
    df:
        Input DataFrame, sorted by date within each group.
    target_col:
        Column to aggregate.
    windows:
        List of rolling window sizes, e.g. ``[7, 14, 28]``.
    group_col:
        If provided, rolling stats are computed within each group.
    agg_funcs:
        Aggregation functions to apply. Defaults to ``["mean", "std"]``.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with rolling feature columns appended.
        Column names follow the pattern ``{target_col}_roll_{window}_{func}``.

    Notes
    -----
    Rolling windows use ``min_periods=1`` to handle the start of each series
    without introducing NaN-filled rows. This slightly inflates early estimates.
    """
    if agg_funcs is None:
        agg_funcs = ["mean", "std"]

    df = df.copy()

    for window in windows:
        for func in agg_funcs:
            col_name = f"{target_col}_roll_{window}_{func}"
            if group_col is not None:
                rolled = (
                    df.groupby(group_col)[target_col]
                    .transform(lambda s, w=window, f=func: s.shift(1).rolling(w, min_periods=1).agg(f))
                )
            else:
                rolled = df[target_col].shift(1).rolling(window, min_periods=1).agg(func)

            df[col_name] = rolled

    return df
