"""Feature engineering for the global LightGBM demand forecasting model.

Design rationale
----------------
Lag selection (7, 14, 28, 91, 364)
    All lags are multiples of 7 to preserve day-of-week alignment.
    "lag_7 on a Monday" is always "last Monday's sales" — the single strongest
    predictor for weekly-seasonal retail data.

Minimum lag = 7 (``MIN_LAG``)
    For a 28-day forecast horizon, any lag < 7 would require look-ahead on
    test days 8-28 (lag_1 on day 8 references test day 7, which we haven't
    predicted). Using lags >= 7 ensures test features always reference training
    history only.

Rolling windows shifted by MIN_LAG
    Rolling statistics use ``.shift(MIN_LAG).rolling(w).mean()`` rather than
    ``.shift(1).rolling(w).mean()``. The rolling mean at time t is therefore the
    mean over [t - MIN_LAG - w, …, t - MIN_LAG - 1], always safe for a
    MIN_LAG-day horizon.

Global model
    Features are built identically for every SKU. The model learns shared
    seasonality patterns once and uses categorical features (item_id, dept_id,
    cat_id) to specialise predictions per series.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration constants ───────────────────────────────────────────────────

LAG_DAYS: list[int] = [7, 14, 28, 91, 364]
"""Lag offsets in days. All multiples of 7 preserve day-of-week alignment."""

ROLLING_WINDOWS: list[int] = [7, 14, 28, 91]
"""Rolling window sizes in days."""

MIN_LAG: int = 7
"""Shift applied before rolling stats to prevent look-ahead for ≤7-day horizons."""

FOURIER_ORDER: int = 3
"""Number of sin/cos pairs per seasonal period (weekly + annual)."""

CAT_COLS_DEFAULT: list[str] = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
"""Columns to encode as LightGBM categorical dtype (integer-coded)."""

# Raw M5 string columns that must never reach the model feature matrix
_M5_RAW_STRING_COLS: frozenset[str] = frozenset([
    "event_name_1", "event_name_2", "event_type_2",  # replaced by has_event flag
    "state_id",  # encoded as category; raw string version dropped
])


# ── Calendar / Fourier features ───────────────────────────────────────────────


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
    include_fourier: bool = True,
    fourier_order: int = FOURIER_ORDER,
) -> pd.DataFrame:
    """Add calendar and Fourier seasonality features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame; a copy is returned.
    date_col : str
        Name of the datetime column.
    include_fourier : bool
        If True, add Fourier sin/cos pairs for weekly (period=7) and
        annual (period=365.25) seasonality.
    fourier_order : int
        Number of Fourier term pairs per seasonal period.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with additional calendar columns.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    df["day_of_week"] = dates.dt.dayofweek.astype("int16")       # 0=Mon, 6=Sun
    df["day_of_month"] = dates.dt.day.astype("int16")
    df["day_of_year"] = dates.dt.dayofyear.astype("int16")
    df["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    df["month"] = dates.dt.month.astype("int16")
    df["quarter"] = dates.dt.quarter.astype("int16")
    df["year"] = dates.dt.year.astype("int16")
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype("int8")

    if include_fourier:
        dow = dates.dt.dayofweek.values.astype(float)
        doy = dates.dt.dayofyear.values.astype(float)
        for k in range(1, fourier_order + 1):
            df[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * dow / 7.0)
            df[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * dow / 7.0)
        for k in range(1, fourier_order + 1):
            df[f"fourier_annual_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            df[f"fourier_annual_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

    return df


# ── Lag features ──────────────────────────────────────────────────────────────


def add_lag_features(
    df: pd.DataFrame,
    sales_col: str = "sales",
    lags: list[int] = LAG_DAYS,
    id_col: str = "id",
) -> pd.DataFrame:
    """Add grouped lag features, respecting series boundaries.

    Assumes ``df`` is sorted by (id_col, date_col). Lag values at the start
    of each series are NaN (correct — no look-back data available).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame sorted by (id_col, date_col).
    sales_col : str
        Column to lag.
    lags : list[int]
        Lag offsets in periods (rows — assumes 1 row per SKU per day).
    id_col : str
        Column identifying each time series.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``{sales_col}_lag_{k}`` columns added.
    """
    df = df.copy()
    for lag in lags:
        df[f"{sales_col}_lag_{lag}"] = (
            df.groupby(id_col, group_keys=False)[sales_col].shift(lag)
        )
    return df


# ── Rolling features ──────────────────────────────────────────────────────────


def add_rolling_features(
    df: pd.DataFrame,
    sales_col: str = "sales",
    windows: list[int] = ROLLING_WINDOWS,
    id_col: str = "id",
    shift: int = MIN_LAG,
    agg_funcs: list[str] | None = None,
) -> pd.DataFrame:
    """Add grouped rolling statistics, shifted to prevent look-ahead.

    The rolling statistic at time t uses values from
    [t - shift - w, …, t - shift - 1], never the current value.
    With ``shift=MIN_LAG=7``, statistics are always safe for a 7-day horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame sorted by (id_col, date_col).
    sales_col : str
        Column to aggregate.
    windows : list[int]
        Rolling window sizes in periods.
    id_col : str
        Column identifying each time series.
    shift : int
        Periods to shift before rolling. Default: MIN_LAG.
    agg_funcs : list[str] or None
        Aggregation functions to apply. Default: ``["mean", "std"]``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with rolling feature columns added.
    """
    if agg_funcs is None:
        agg_funcs = ["mean", "std"]

    df = df.copy()

    for w in windows:
        min_periods = max(1, w // 2)

        for func in agg_funcs:
            col_name = f"{sales_col}_roll_{w}_{func}"

            if func == "mean":
                df[col_name] = (
                    df.groupby(id_col)[sales_col].transform(
                        lambda x, _w=w, _mp=min_periods, _s=shift: (  # noqa: B023
                            x.shift(_s).rolling(_w, min_periods=_mp).mean()
                        )
                    )
                )
            elif func == "std":
                df[col_name] = (
                    df.groupby(id_col)[sales_col].transform(
                        lambda x, _w=w, _mp=min_periods, _s=shift: (  # noqa: B023
                            x.shift(_s).rolling(_w, min_periods=_mp).std()
                        )
                    )
                )
            elif func == "max":
                df[col_name] = (
                    df.groupby(id_col)[sales_col].transform(
                        lambda x, _w=w, _mp=min_periods, _s=shift: (  # noqa: B023
                            x.shift(_s).rolling(_w, min_periods=_mp).max()
                        )
                    )
                )
            else:
                raise ValueError(f"Unsupported agg_func: {func!r}")

    return df


# ── Price features ────────────────────────────────────────────────────────────


def add_price_features(
    df: pd.DataFrame,
    price_col: str = "sell_price",
    id_col: str = "id",
    ref_window: int = 28,
) -> pd.DataFrame:
    """Add price-change and discount indicator features.

    Computes a rolling reference price (trailing mean) and expresses the
    current price as a percentage change from it. A discount flag fires when
    the current price is more than 1% below the reference price.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``price_col``.
    price_col : str
        Column containing the sell price.
    id_col : str
        Column identifying each series (prices vary by item).
    ref_window : int
        Rolling window for reference price computation (default 28 days).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``price_change_pct`` and ``is_price_discount`` columns.
    """
    df = df.copy()
    if price_col not in df.columns:
        return df

    ref_price = df.groupby(id_col)[price_col].transform(
        lambda x: x.shift(1).rolling(ref_window, min_periods=1).mean()
    )
    # Avoid division by zero with a small epsilon
    df["price_change_pct"] = (df[price_col] - ref_price) / (ref_price.clip(lower=1e-6))
    df["is_price_discount"] = (df[price_col] < ref_price * 0.99).astype("int8")
    return df


# ── Categorical encoding ──────────────────────────────────────────────────────


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str] = CAT_COLS_DEFAULT,
) -> pd.DataFrame:
    """Encode columns as ``pd.CategoricalDtype`` for LightGBM native handling.

    LightGBM handles categoricals more efficiently than one-hot encoding for
    high-cardinality features (e.g., item_id has 3,049 unique values in M5 CA_1).
    Columns absent from ``df`` are silently skipped.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list[str]
        Column names to encode.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with specified columns re-typed as ``category``.
    """
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ── Main feature pipeline ─────────────────────────────────────────────────────


def build_lgbm_features(
    df: pd.DataFrame,
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
    price_col: Optional[str] = "sell_price",
    event_col: Optional[str] = "event_type_1",
    snap_col: Optional[str] = None,
    lags: list[int] = LAG_DAYS,
    rolling_windows: list[int] = ROLLING_WINDOWS,
    cat_cols: list[str] = CAT_COLS_DEFAULT,
    drop_na_rows: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix for the LightGBM global model.

    Applies all feature transformations in a single pass. The caller must sort
    ``df`` by (id_col, date_col) before calling, or pass unsorted data and rely
    on the internal sort.

    Lag and rolling features create NaN values at the start of each series
    (because there is no look-back history). If ``drop_na_rows=True``, these
    rows are dropped before returning — appropriate for training. For test
    data (where history is prepended from training), set ``drop_na_rows=False``
    and slice test rows by date after calling.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with at minimum (id_col, date_col, sales_col).
    id_col, date_col, sales_col : str
        Column names.
    price_col : str or None
        Sell price column. If present, adds price-change features.
    event_col : str or None
        Event type column. If present, adds binary has_event flag.
    snap_col : str or None
        SNAP (food-stamp day) flag column.
    lags : list[int]
        Lag offsets in days.
    rolling_windows : list[int]
        Rolling window sizes in days.
    cat_cols : list[str]
        Columns to encode as LightGBM categoricals.
    drop_na_rows : bool
        If True, drop rows where any lag feature is NaN.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame. Shape: (n_non_nan_rows, n_features + original_cols).
    """
    df = df.sort_values([id_col, date_col]).copy()
    n_input = len(df)

    # Calendar + Fourier
    df = add_calendar_features(df, date_col=date_col, include_fourier=True)

    # Lag features
    df = add_lag_features(df, sales_col=sales_col, lags=lags, id_col=id_col)

    # Rolling features (mean + std)
    df = add_rolling_features(df, sales_col=sales_col, windows=rolling_windows, id_col=id_col)

    # Price features
    if price_col is not None and price_col in df.columns:
        df = add_price_features(df, price_col=price_col, id_col=id_col)

    # Binary event flag
    if event_col is not None and event_col in df.columns:
        df["has_event"] = (df[event_col].notna() & (df[event_col] != "")).astype("int8")

    # SNAP day flag
    if snap_col is not None and snap_col in df.columns:
        df["snap_day"] = df[snap_col].astype("int8")

    # Categorical encoding — include id_col itself
    all_cat_cols = list(cat_cols)
    if id_col not in all_cat_cols:
        all_cat_cols.append(id_col)
    df = encode_categoricals(df, cat_cols=all_cat_cols)

    if drop_na_rows:
        lag_cols = [f"{sales_col}_lag_{k}" for k in lags]
        before = len(df)
        df = df.dropna(subset=lag_cols).reset_index(drop=True)
        logger.debug("Dropped %d NaN rows (lag boundary), %d remaining.", before - len(df), len(df))

    logger.info(
        "build_lgbm_features: %d input rows → %d output rows, %d columns.",
        n_input, len(df), df.shape[1],
    )
    return df


# ── Feature column selection ──────────────────────────────────────────────────


def get_feature_columns(
    df: pd.DataFrame,
    sales_col: str = "sales",
    date_col: str = "date",
    id_col: str = "id",
    extra_exclude: Optional[list[str]] = None,
) -> list[str]:
    """Return the model feature column names from a feature-enriched DataFrame.

    Only includes columns with numeric (int / float / bool) or LightGBM-compatible
    ``category`` dtype.  Raw string / object columns are silently excluded — this
    prevents M5's ``event_name_1``, ``state_id``, etc. from reaching the model.

    Parameters
    ----------
    df : pd.DataFrame
    sales_col : str
        Target column to exclude.
    date_col : str
        Date column to exclude.
    id_col : str
        Raw id column to exclude (the category-encoded version is kept).
    extra_exclude : list[str] or None
        Additional columns to exclude.

    Returns
    -------
    list[str]
        Ordered list of feature column names safe to pass to LightGBM.
    """
    always_exclude = {
        sales_col, date_col, id_col,
        "d",          # M5 day index string ("d_1" … "d_1941")
        "wm_yr_wk",   # M5 week key used only for price joins
        "_is_test",   # internal tag used during predict()
    }
    if extra_exclude:
        always_exclude.update(extra_exclude)

    # LightGBM accepts: int*, uint*, float*, bool, and pandas category.
    # Everything else (object / str) raises ValueError at fit-time.
    _valid_kinds = frozenset("iufbU")  # int, uint, float, bool — numpy kind codes
    # pandas CategoricalDtype has kind "O" but name "category"

    result = []
    for c in df.columns:
        if c in always_exclude:
            continue
        dtype = df[c].dtype
        if dtype.name == "category" or dtype.kind in _valid_kinds:
            result.append(c)
        else:
            logger.debug("get_feature_columns: skipping '%s' (dtype=%s)", c, dtype)

    return result
