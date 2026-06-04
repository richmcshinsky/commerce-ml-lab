"""Walk-forward backtesting utilities for time-series models.

Walk-forward (also called expanding-window or rolling-origin) evaluation
is the correct way to evaluate forecasting models. It simulates what would
happen in production: you train on everything up to time T, forecast the
next H steps, record errors, then advance T by one step and repeat.

This is *not* k-fold cross-validation. Shuffling time-series data causes
leakage because future information leaks into the training set.

References
----------
- Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*,
  Chapter 5 (Evaluating Forecast Accuracy).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single train/test split from walk-forward evaluation.

    Attributes
    ----------
    fold:
        Zero-indexed fold number.
    train_end:
        Last date (inclusive) in the training window.
    test_start:
        First date of the test window.
    test_end:
        Last date (inclusive) of the test window.
    train_idx:
        Integer indices of training rows in the original DataFrame.
    test_idx:
        Integer indices of test rows in the original DataFrame.
    """

    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: list[int]
    test_idx: list[int]


def walk_forward_splits(
    df: pd.DataFrame,
    date_col: str,
    horizon: int,
    min_train_periods: int,
    step: int = 1,
    freq: str = "D",
) -> Iterator[WalkForwardSplit]:
    """Generate walk-forward train/test splits for a time-series DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame. Must contain a datetime column ``date_col``.
    date_col:
        Name of the date column.
    horizon:
        Forecast horizon in periods (e.g. 28 for a 4-week forecast).
    min_train_periods:
        Minimum number of periods required in the training window before
        generating the first split.
    step:
        How many periods to advance the training window between folds
        (default 1). Larger steps = fewer folds = faster evaluation.
    freq:
        Pandas frequency string for the date grid (default ``"D"`` for daily).

    Yields
    ------
    WalkForwardSplit
        One split per fold, in chronological order.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range("2020-01-01", periods=100, freq="D")
    >>> df = pd.DataFrame({"date": dates, "sales": range(100)})
    >>> splits = list(walk_forward_splits(df, "date", horizon=7, min_train_periods=30))
    >>> len(splits)
    10
    """
    dates = pd.to_datetime(df[date_col])
    all_dates = pd.date_range(dates.min(), dates.max(), freq=freq)
    date_to_idx: dict[pd.Timestamp, list[int]] = {}
    for i, d in enumerate(dates):
        date_to_idx.setdefault(d, []).append(i)

    fold = 0
    train_end_pos = min_train_periods - 1  # zero-indexed

    while train_end_pos + horizon < len(all_dates):
        train_end = all_dates[train_end_pos]
        test_start = all_dates[train_end_pos + 1]
        test_end = all_dates[min(train_end_pos + horizon, len(all_dates) - 1)]

        train_idx = [i for d in all_dates[: train_end_pos + 1] for i in date_to_idx.get(d, [])]
        test_idx = [
            i
            for d in all_dates[train_end_pos + 1 : train_end_pos + 1 + horizon]
            for i in date_to_idx.get(d, [])
        ]

        if train_idx and test_idx:
            yield WalkForwardSplit(
                fold=fold,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_idx=train_idx,
                test_idx=test_idx,
            )

        fold += 1
        train_end_pos += step


def run_backtest(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    horizon: int,
    min_train_periods: int,
    step: int = 7,
) -> pd.DataFrame:
    """Run a full walk-forward backtest and collect per-fold predictions.

    Parameters
    ----------
    df:
        Input DataFrame.
    date_col:
        Name of the date column.
    target_col:
        Name of the target column.
    fit_predict_fn:
        Callable with signature ``(train_df, test_df) -> pd.Series``.
        Returns a Series of forecast values aligned with ``test_df``'s index.
    horizon:
        Forecast horizon.
    min_train_periods:
        Minimum training periods before first fold.
    step:
        Fold step size (default 7 = weekly re-evaluation).

    Returns
    -------
    pd.DataFrame
        Columns: ``fold``, ``date``, ``actual``, ``forecast``.
    """
    records = []

    for split in walk_forward_splits(df, date_col, horizon, min_train_periods, step):
        train = df.iloc[split.train_idx].copy()
        test = df.iloc[split.test_idx].copy()

        forecasts = fit_predict_fn(train, test)

        for idx, (_, row) in zip(split.test_idx, test.iterrows(), strict=False):
            records.append(
                {
                    "fold": split.fold,
                    "date": row[date_col],
                    "actual": row[target_col],
                    "forecast": forecasts.loc[idx] if idx in forecasts.index else float("nan"),
                }
            )

    return pd.DataFrame(records)
