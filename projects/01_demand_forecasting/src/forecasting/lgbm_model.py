"""Global LightGBM demand forecasting model with 80% prediction intervals.

Architecture
------------
Three LightGBM models are trained:

1. **Point forecast** (``objective="regression"``)
   Minimises MSE; used as the primary forecast for inventory decisions.

2. **Lower bound** (``objective="quantile"``, ``alpha=0.1``)
   10th-percentile forecast — lower bound of the 80% prediction interval.

3. **Upper bound** (``objective="quantile"``, ``alpha=0.9``)
   90th-percentile forecast — upper bound of the 80% prediction interval.

All three share the same feature matrix, so the marginal cost of the quantile
models is small. Quantile regression makes no distributional assumptions and
handles the zero-inflated, right-skewed distribution of retail sales naturally.

Global model rationale
-----------------------
A single model is trained across all SKUs simultaneously. It learns shared
seasonal patterns (day-of-week, holiday uplift) once, while categorical features
(item_id, dept_id, cat_id) let the model specialise predictions per series.
This is more data-efficient than 3,049 separate models and consistently matches
or outperforms them on M5-style retail data.

Test-time feature construction
-------------------------------
Lag and rolling features for the test window are computed using training history.
In ``fit()``, the tail of training data (up to ``max_history_rows`` rows per SKU)
is stored in ``self.history_tail_``. In ``predict()``, this tail is prepended to
the test DataFrame before feature construction, then sliced away. This avoids any
look-ahead while keeping the predict interface clean.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from forecasting.features import (
    CAT_COLS_DEFAULT,
    LAG_DAYS,
    ROLLING_WINDOWS,
    build_lgbm_features,
    get_feature_columns,
)
from forecasting.models import BaseForecaster

logger = logging.getLogger(__name__)

# ── Default hyperparameters ───────────────────────────────────────────────────

LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
"""Sensible defaults tuned for M5 CA_1 (~3k SKUs × ~5 years).

Increase ``n_estimators`` (e.g. to 2000) with ``early_stopping_rounds=50``
for production-quality training. The defaults here train in ~2-3 minutes on
a modern laptop.
"""


# ── LightGBM forecaster ───────────────────────────────────────────────────────


class LGBMForecaster(BaseForecaster):
    """Global LightGBM demand forecasting model.

    Trains one point-forecast model and two quantile models for 80% prediction
    intervals. All three share the same feature matrix.

    Parameters
    ----------
    lgbm_params : dict or None
        LightGBM hyperparameters. If None, ``LGBM_PARAMS`` is used.
    lags : list[int]
        Lag offsets in days for feature engineering.
    rolling_windows : list[int]
        Rolling window sizes in days.
    cat_cols : list[str]
        Columns to encode as LightGBM categoricals (integer-coded).
    price_col : str or None
        Sell price column name, if present in the data.
    event_col : str or None
        Event type column name, if present.
    max_history_rows : int
        Max rows per SKU stored from training for test-time feature construction.
        Must exceed ``max(lags) + max(rolling_windows) + forecast_horizon``.

    Attributes
    ----------
    model_ : lightgbm.LGBMRegressor
        Trained point forecast model.
    model_q10_ : lightgbm.LGBMRegressor
        Trained lower bound (10th-percentile) model.
    model_q90_ : lightgbm.LGBMRegressor
        Trained upper bound (90th-percentile) model.
    feature_cols_ : list[str]
        Feature column names used at training time.
    history_tail_ : pd.DataFrame
        Stored tail of training data for test-time feature construction.
    """

    def __init__(
        self,
        lgbm_params: Optional[dict[str, Any]] = None,
        lags: list[int] = LAG_DAYS,
        rolling_windows: list[int] = ROLLING_WINDOWS,
        cat_cols: list[str] = CAT_COLS_DEFAULT,
        price_col: Optional[str] = "sell_price",
        event_col: Optional[str] = "event_type_1",
        max_history_rows: int = 500,
    ) -> None:
        self.lgbm_params = lgbm_params if lgbm_params is not None else LGBM_PARAMS.copy()
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.cat_cols = cat_cols
        self.price_col = price_col
        self.event_col = event_col
        self.max_history_rows = max_history_rows

        self.model_: Any = None
        self.model_q10_: Any = None
        self.model_q90_: Any = None
        self.feature_cols_: list[str] = []
        self.history_tail_: Optional[pd.DataFrame] = None
        self._id_col: str = "id"
        self._date_col: str = "date"
        self._sales_col: str = "sales"

    @property
    def name(self) -> str:
        return "LightGBM"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_features(
        self,
        df: pd.DataFrame,
        drop_na_rows: bool,
    ) -> pd.DataFrame:
        """Build feature matrix on ``df``, respecting stored column names."""
        return build_lgbm_features(
            df,
            id_col=self._id_col,
            date_col=self._date_col,
            sales_col=self._sales_col,
            price_col=self.price_col,
            event_col=self.event_col,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            cat_cols=self.cat_cols,
            drop_na_rows=drop_na_rows,
        )

    def _make_lgbm(self, objective: str, alpha: float = 0.5) -> Any:
        """Instantiate a LightGBM regressor for the given objective."""
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is required: pip install lightgbm") from exc

        params = {**self.lgbm_params, "objective": objective}
        if objective == "quantile":
            params["alpha"] = alpha
        return LGBMRegressor(**params)

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(
        self,
        train: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
        sales_col: str = "sales",
    ) -> "LGBMForecaster":
        """Train point-forecast and quantile LightGBM models.

        Parameters
        ----------
        train : pd.DataFrame
            Long-format training data sorted by (id_col, date_col).
        id_col, date_col, sales_col : str

        Returns
        -------
        self
        """
        self._id_col = id_col
        self._date_col = date_col
        self._sales_col = sales_col

        # Store training tail for test-time feature construction
        self.history_tail_ = (
            train.sort_values([id_col, date_col])
            .groupby(id_col, group_keys=False)
            .tail(self.max_history_rows)
            .copy()
        )

        # Build training features
        logger.info("Building training features (%d rows, %d SKUs)…", len(train), train[id_col].nunique())
        feat_df = self._build_features(train, drop_na_rows=True)

        self.feature_cols_ = get_feature_columns(
            feat_df,
            sales_col=sales_col,
            date_col=date_col,
            id_col=id_col,
        )

        X = feat_df[self.feature_cols_]
        y = feat_df[sales_col].values
        cat_feature_names = [c for c in self.feature_cols_ if X[c].dtype.name == "category"]

        logger.info(
            "Training on %d rows × %d features (%d categorical)…",
            len(X), len(self.feature_cols_), len(cat_feature_names),
        )

        fit_kwargs: dict[str, Any] = {}
        if cat_feature_names:
            fit_kwargs["categorical_feature"] = cat_feature_names

        # Point forecast model (MSE)
        self.model_ = self._make_lgbm("regression")
        self.model_.fit(X, y, **fit_kwargs)
        logger.info("Point forecast model trained.")

        # Quantile q10 (lower bound)
        self.model_q10_ = self._make_lgbm("quantile", alpha=0.1)
        self.model_q10_.fit(X, y, **fit_kwargs)
        logger.info("Quantile q10 model trained.")

        # Quantile q90 (upper bound)
        self.model_q90_ = self._make_lgbm("quantile", alpha=0.9)
        self.model_q90_.fit(X, y, **fit_kwargs)
        logger.info("Quantile q90 model trained.")

        return self

    def _predict_recursive(
        self,
        test: pd.DataFrame,
        id_col: str,
        date_col: str,
        with_intervals: bool = False,
        batch_size: int = 7,
    ) -> pd.DataFrame:
        """Batched recursive multi-step prediction.

        **Why recursive?**
        Setting all test sales to NaN before building features causes lag_7 at
        test day 8 to be NaN (it looks back to test day 1, which has no sales).
        The model then predicts near-zero because NaN is treated as missing.

        **How it works:**
        Predictions are made in chunks of ``batch_size`` days (default 7 = min lag).
        Before each chunk, the working history is updated with the *previous* chunk's
        predictions. This means:

          - Days 1-7:   lag_7 references actual training data ✓
          - Days 8-14:  lag_7 references days 1-7 *predictions* (in history) ✓
          - Days 15-21: lag_7 references days 8-14 predictions ✓
          - Days 22-28: lag_7 references days 15-21 predictions ✓

        Setting batch_size = min(lags) = 7 ensures no intra-batch recursion is
        needed: within any single batch, lag_7 always points to the *previous* batch
        (already in history), never to the current batch.

        Parameters
        ----------
        test : pd.DataFrame
        id_col, date_col : str
        with_intervals : bool
            If True, also predict q10/q90 and return lower_80/upper_80 columns.
        batch_size : int
            Days per prediction batch. Must be ≤ min(self.lags).

        Returns
        -------
        pd.DataFrame
            ``test`` rows with forecast (+ lower_80, upper_80 if with_intervals).
        """
        if self.history_tail_ is None:
            raise RuntimeError("Call fit() before predict().")

        working_history = self.history_tail_.copy()
        results: list[pd.DataFrame] = []

        test_sorted = test.sort_values([id_col, date_col]).reset_index(drop=True)
        all_dates = sorted(pd.to_datetime(test_sorted[date_col]).unique())

        n_batches = -(-len(all_dates) // batch_size)  # ceiling division
        logger.info("Recursive prediction: %d dates in %d batches of %d.", len(all_dates), n_batches, batch_size)

        for batch_idx in range(0, len(all_dates), batch_size):
            batch_dates = all_dates[batch_idx: batch_idx + batch_size]
            date_mask = pd.to_datetime(test_sorted[date_col]).isin(batch_dates)
            batch_rows = test_sorted[date_mask].copy()

            # Feature construction: training history + this batch (sales=NaN for the batch)
            combined = pd.concat(
                [
                    working_history.assign(_is_test=False),
                    batch_rows.assign(**{self._sales_col: np.nan, "_is_test": True}),
                ],
                ignore_index=True,
            ).sort_values([id_col, date_col])

            feat_df = self._build_features(combined, drop_na_rows=False)
            batch_feat = feat_df[feat_df["_is_test"].astype(bool)].reset_index(drop=True)
            X_batch = batch_feat[self.feature_cols_]

            point = np.clip(self.model_.predict(X_batch), 0, None)
            batch_rows = batch_rows.reset_index(drop=True)
            batch_rows["forecast"] = point.astype("float64")

            if with_intervals:
                lower = np.clip(self.model_q10_.predict(X_batch), 0, None)
                upper = np.clip(self.model_q90_.predict(X_batch), 0, None)
                # Enforce ordering
                batch_rows["lower_80"] = np.minimum(lower, point).astype("float64")
                batch_rows["upper_80"] = np.maximum(upper, point).astype("float64")

            results.append(batch_rows)

            # Update working history: append this batch with point forecasts as sales.
            # Future batches' lag features will reference these predictions instead of NaN.
            history_update = batch_rows.drop(
                columns=[c for c in ["forecast", "lower_80", "upper_80", "_is_test"]
                         if c in batch_rows.columns],
                errors="ignore",
            ).copy()
            history_update[self._sales_col] = point
            working_history = pd.concat([working_history, history_update], ignore_index=True)

        return pd.concat(results, ignore_index=True)

    def predict(
        self,
        test: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Generate point forecasts for the test period.

        Uses batched recursive prediction so that lag features for days beyond
        the minimum lag correctly reference prior predicted values, not NaN.

        Parameters
        ----------
        test : pd.DataFrame
            Long-format test data. Must contain id_col and date_col.
        id_col, date_col : str

        Returns
        -------
        pd.DataFrame
            ``test`` with a ``"forecast"`` column (float64, clipped to >= 0).
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._predict_recursive(test, id_col, date_col, with_intervals=False)

    def predict_with_intervals(
        self,
        test: pd.DataFrame,
        id_col: str = "id",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Generate point forecasts and 80% prediction intervals.

        Uses batched recursive prediction. Both point and quantile models
        receive identical feature matrices for each batch.

        Parameters
        ----------
        test : pd.DataFrame
        id_col, date_col : str

        Returns
        -------
        pd.DataFrame
            ``test`` with columns ``forecast``, ``lower_80``, ``upper_80``.
            lower_80 <= forecast <= upper_80 is enforced by clipping.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._predict_recursive(test, id_col, date_col, with_intervals=True)

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 30,
    ) -> pd.DataFrame:
        """Return a feature importance table from the point forecast model.

        Parameters
        ----------
        importance_type : str
            ``"gain"`` (default) or ``"split"``. Gain is more interpretable
            for tree models — it measures reduction in loss weighted by usage.
        top_n : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``, sorted descending.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before get_feature_importance().")

        importances = self.model_.booster_.feature_importance(importance_type=importance_type)
        df = pd.DataFrame(
            {"feature": self.feature_cols_, "importance": importances}
        ).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        return df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Serialise the forecaster to disk.

        Parameters
        ----------
        path : Path or str
            Destination file path (e.g. ``results/lgbm_model.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("LGBMForecaster saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "LGBMForecaster":
        """Load a serialised forecaster from disk.

        Parameters
        ----------
        path : Path or str
            Path to a ``.pkl`` file written by :meth:`save`.

        Returns
        -------
        LGBMForecaster
        """
        path = Path(path)
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LGBMForecaster, got {type(obj)}")
        logger.info("LGBMForecaster loaded from %s", path)
        return obj
