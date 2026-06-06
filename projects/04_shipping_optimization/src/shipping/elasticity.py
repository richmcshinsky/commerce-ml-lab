"""Conversion elasticity model — P(convert | session_features, shipping_price).

Because shipping price was randomly assigned in the training data (A/B test),
the price coefficient has a causal interpretation: it estimates the average
change in conversion probability per dollar of shipping cost.

Architecture
------------
1. LightGBM classifier on session features + shipping_price.
2. Isotonic calibration so scores are well-calibrated probabilities.
3. ``predict_at_price`` evaluates the counterfactual: *given this session,
   what would P(convert) be at each price point?*
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

SESSION_FEATURES: list[str] = [
    "cart_value",
    "n_items",
    "is_returning",
    "session_depth",
    "time_on_checkout",
    "device_mobile",
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "shipping_price",   # the key causal feature
]

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": 42,
}


class ConversionElasticityModel:
    """LightGBM model for P(convert | session_features, shipping_price).

    Trained on A/B test data where shipping price was randomly assigned,
    so the price coefficient estimates a causal effect.

    Parameters
    ----------
    lgbm_params:
        LightGBM hyperparameters.  Defaults to ``DEFAULT_LGBM_PARAMS``.
    """

    def __init__(self, lgbm_params: dict[str, Any] | None = None) -> None:
        self.lgbm_params = lgbm_params or DEFAULT_LGBM_PARAMS.copy()
        self._model: CalibratedClassifierCV | None = None
        self.feature_cols_: list[str] = SESSION_FEATURES
        self.pr_auc_: float | None = None

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> ConversionElasticityModel:
        """Fit the elasticity model on checkout session data.

        Parameters
        ----------
        df:
            Training DataFrame.  Must contain all columns in ``SESSION_FEATURES``
            plus ``converted``.

        Returns
        -------
        ConversionElasticityModel
            Self (for chaining).
        """
        from lightgbm import LGBMClassifier

        X = self._prepare_x(df)
        y = df["converted"].astype(int).values

        base = LGBMClassifier(**self.lgbm_params)
        self._model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        self._model.fit(X, y)
        logger.info("ConversionElasticityModel fitted on %d sessions.", len(df))
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(convert) for each row at its assigned shipping_price.

        Parameters
        ----------
        df:
            DataFrame with all ``SESSION_FEATURES`` columns.

        Returns
        -------
        np.ndarray, shape (n,)
            Estimated conversion probability for each session.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._prepare_x(df)
        return self._model.predict_proba(X)[:, 1]

    def predict_at_price(
        self, df: pd.DataFrame, price: float
    ) -> np.ndarray:
        """Return counterfactual P(convert) if shipping_price were set to ``price``.

        Parameters
        ----------
        df:
            DataFrame with session features (shipping_price column ignored).
        price:
            Hypothetical shipping price to evaluate.

        Returns
        -------
        np.ndarray, shape (n,)
            Counterfactual conversion probability at the specified price.
        """
        df_cf = df.copy()
        df_cf["shipping_price"] = price
        return self.predict_proba(df_cf)

    def price_curve(
        self, session: pd.Series, prices: list[float]
    ) -> pd.DataFrame:
        """Return P(convert) and expected margin at each price for one session.

        Parameters
        ----------
        session:
            Single checkout session (one row of the sessions DataFrame).
        prices:
            List of prices to evaluate.

        Returns
        -------
        pd.DataFrame
            Columns: ``price``, ``p_convert``, ``expected_margin``.
        """
        from .synthetic import PRODUCT_MARGIN_RATE, SHIPPING_COST_TO_MERCHANT

        session_df = pd.DataFrame([session] * len(prices)).reset_index(drop=True)
        session_df["shipping_price"] = prices
        p_vals = self.predict_proba(session_df)
        margins = p_vals * (session["cart_value"] * PRODUCT_MARGIN_RATE + np.array(prices) - SHIPPING_COST_TO_MERCHANT)
        return pd.DataFrame({"price": prices, "p_convert": p_vals, "expected_margin": margins})

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Pickle the fitted model to ``path``."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ConversionElasticityModel saved to %s.", path)

    @classmethod
    def load(cls, path: Path) -> ConversionElasticityModel:
        """Load a pickled model from ``path``."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("ConversionElasticityModel loaded from %s.", path)
        return obj  # type: ignore[return-value]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _prepare_x(self, df: pd.DataFrame) -> np.ndarray:  # noqa: N802 (X is ML convention)
        """Select and type-cast feature columns."""
        X = df[self.feature_cols_].copy()
        X["is_returning"] = X["is_returning"].astype(float)
        X["device_mobile"] = X["device_mobile"].astype(float)
        return X.values.astype(float)
