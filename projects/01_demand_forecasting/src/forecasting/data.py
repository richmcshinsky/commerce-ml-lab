"""M5 data loading and preprocessing for the demand forecasting project.

The M5 dataset is stored in wide format: one row per SKU, one column per day
(``d_1`` through ``d_1941``). This module transforms it into long format
(one row per SKU-date), joins the calendar and price tables, and produces
clean train/test splits.

Typical usage
-------------
::

    from forecasting.data import load_m5_long, train_test_split

    df    = load_m5_long(store_id="CA_1")
    train, test = train_test_split(df, test_days=28)

The test window (last 28 days) matches the M5 competition horizon and is a
realistic planning horizon for weekly reorder decisions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running standalone or from a notebook: add repo src/ to path
sys.path.insert(0, str(Path(__file__).parents[5] / "src"))

from commerce_ml.data.loaders import (
    DATA_DIR,
    load_m5_calendar,
    load_m5_prices,
    load_m5_sales,
)

logger = logging.getLogger(__name__)

M5_DIR = DATA_DIR / "m5"
DEFAULT_TEST_DAYS: int = 28


# ── Main loader ───────────────────────────────────────────────────────────────


def load_m5_long(
    store_id: str = "CA_1",
    data_dir: Path = M5_DIR,
) -> pd.DataFrame:
    """Load M5 data and return a single long-format DataFrame.

    Performs three joins:

    1. Melt wide sales -> (id, d, sales)
    2. Merge calendar on ``d`` -> adds date, weekday, events, SNAP flags
    3. Merge prices on (store_id, item_id, wm_yr_wk) -> adds sell_price

    Parameters
    ----------
    store_id:
        Store to load. Default ``"CA_1"`` (~3,049 SKUs, ~5.8M rows).
        Pass ``None`` to load all 10 stores (~30,490 SKUs, ~58M rows).
    data_dir:
        Directory containing the extracted M5 CSV files.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:

        - ``id`` -- unique series key (item_id + store_id)
        - ``item_id``, ``dept_id``, ``cat_id``, ``store_id``, ``state_id``
        - ``d`` -- M5 day string ("d_1" ... "d_1941")
        - ``date`` -- calendar date (datetime64[ns])
        - ``sales`` -- units sold (int32, >= 0)
        - ``sell_price`` -- weekly sell price (float, NaN before listing)
        - ``event_name_1``, ``event_type_1`` -- primary calendar event
        - ``snap_CA``, ``snap_TX``, ``snap_WI`` -- SNAP benefit flags (0/1)
        - ``wm_yr_wk`` -- Walmart fiscal year-week

    Raises
    ------
    FileNotFoundError
        If any M5 file is missing. Run ``make data-m5`` first.
    """
    logger.info("Loading M5 store=%s from %s", store_id, data_dir)

    sales_wide = load_m5_sales(data_dir=data_dir, subset_store=store_id)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    # Melt: wide -> long
    df = sales_wide.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    df["sales"] = df["sales"].astype(np.int32)

    # Join calendar
    calendar = load_m5_calendar(data_dir=data_dir)
    cal_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "event_name_1",
        "event_type_1",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    df = df.merge(calendar[cal_cols], on="d", how="left")
    df["date"] = pd.to_datetime(df["date"])

    # Join prices (left: new items have no price history)
    prices = load_m5_prices(data_dir=data_dir)
    if store_id is not None:
        prices = prices[prices["store_id"] == store_id]
    df = df.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    logger.info(
        "M5 long: %d rows, %d series, %s to %s",
        len(df),
        df["id"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def train_test_split(
    df: pd.DataFrame,
    test_days: int = DEFAULT_TEST_DAYS,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train/test split on a long-format time-series DataFrame.

    The last ``test_days`` calendar days form the test set. All SKUs share
    the same split boundary — the correct approach for a global model.

    Parameters
    ----------
    df:
        Long-format DataFrame with a date column.
    test_days:
        Number of trailing days to hold out. Default 28 (M5 horizon).
    date_col:
        Name of the date column.

    Returns
    -------
    train : pd.DataFrame
    test : pd.DataFrame

    Notes
    -----
    Split is on calendar days, not row count. SKUs with missing dates will
    have fewer test rows than expected — this is fine for evaluation.
    """
    dates = pd.to_datetime(df[date_col])
    split_date = dates.max() - pd.Timedelta(days=test_days - 1)

    train = df[dates < split_date].copy()
    test = df[dates >= split_date].copy()

    logger.info(
        "Train: %s -> %s (%d rows)  |  Test: %s -> %s (%d rows)",
        train[date_col].min().date(),
        train[date_col].max().date(),
        len(train),
        test[date_col].min().date(),
        test[date_col].max().date(),
        len(test),
    )
    return train, test


# ── Aggregate helpers ─────────────────────────────────────────────────────────


def zero_rate(df: pd.DataFrame, sales_col: str = "sales") -> float:
    """Fraction of (SKU, day) observations with zero sales.

    Parameters
    ----------
    df : pd.DataFrame
    sales_col : str

    Returns
    -------
    float
        In M5 CA_1, this is ~62%. Useful for understanding intermittent
        demand before choosing a model.
    """
    return float((df[sales_col] == 0).mean())


def aggregate_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    sales_col: str = "sales",
    group_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate daily sales to a store total or by category/department.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    sales_col : str
    group_col : str or None
        If provided (e.g. ``"cat_id"``), returns one series per group value.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, (group_col if given), ``sales``.
    """
    keys = [date_col] if group_col is None else [date_col, group_col]
    return df.groupby(keys)[sales_col].sum().reset_index().sort_values(date_col)


def make_series_pivot(
    df: pd.DataFrame,
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
) -> pd.DataFrame:
    """Pivot long format to wide: rows=dates, columns=SKU ids.

    Useful for classical per-series models (ETS, seasonal decomposition)
    and for computing hierarchical aggregates.

    Parameters
    ----------
    df : pd.DataFrame
    id_col, date_col, sales_col : str

    Returns
    -------
    pd.DataFrame
        Shape (n_dates, n_series) with DatetimeIndex. Missing values filled 0.
    """
    wide = df.pivot_table(
        index=date_col,
        columns=id_col,
        values=sales_col,
        aggfunc="sum",
        fill_value=0,
    )
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def get_active_items(
    df: pd.DataFrame,
    min_nonzero_days: int = 30,
    sales_col: str = "sales",
    id_col: str = "id",
) -> list[str]:
    """Return IDs of items with enough sales history for reliable modelling.

    Very sparse series (e.g. newly listed or discontinued items) produce
    unstable forecasts. This filter identifies items that have sold on at
    least ``min_nonzero_days`` distinct days.

    Parameters
    ----------
    df : pd.DataFrame
    min_nonzero_days : int
    sales_col, id_col : str

    Returns
    -------
    list[str]
        Item IDs meeting the activity threshold.
    """
    counts = df[df[sales_col] > 0].groupby(id_col)[sales_col].count()
    return list(counts[counts >= min_nonzero_days].index)
