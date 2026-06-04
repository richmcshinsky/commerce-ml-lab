"""Return likelihood model: P(return | order features).

Predicts the probability that an order will be returned. Scored at
order-fulfillment time — before any return event occurs.

Used for:
- Risk-based fulfilment routing (flag high-risk orders for inspection)
- Aggregate return forecasting (expected returns per category per week)
- Feature input to the fraud model (customer-level return rate)

Architecture
------------
LightGBM classifier with isotonic probability calibration.  Calibration
ensures that ``predict_proba(x) = 0.30`` actually means ~30% of similar
orders are returned, which matters for downstream cost calculations.

Features
--------
All features are available at order fulfillment (no look-ahead):
- item_price, category (categorical), channel (categorical)
- quantity, account_age_days
- customer_lifetime_return_rate, customer_total_orders
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LIKELIHOOD_FEATURES = [
    "item_price",
    "quantity",
    "account_age_days",
    "customer_lifetime_return_rate",
    "customer_total_orders",
    "category",
    "channel",
]

RISK_THRESHOLDS = {"low": 0.10, "medium": 0.25}
"""Orders with P(return) < 0.10 are low risk; > 0.25 are high risk."""


class ReturnLikelihoodModel:
    """LightGBM-based return probability model with isotonic calibration.

    Parameters
    ----------
    lgbm_params:
        LightGBM hyperparameters. Defaults tuned for ~70K order dataset.
    calibrate:
        If True, apply isotonic regression calibration after training.

    Attributes
    ----------
    model_:
        Trained LGBMClassifier.
    calibrator_:
        Trained IsotonicRegression (None if calibrate=False).
    feature_cols_:
        Feature column names used at training time.
    threshold_:
        Classification threshold (default 0.5, adjustable post-hoc).
    """

    def __init__(
        self,
        lgbm_params: dict[str, Any] | None = None,
        calibrate: bool = True,
    ) -> None:
        self.lgbm_params = lgbm_params or {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.calibrate = calibrate
        self.model_: Any = None
        self.calibrator_: Any = None
        self.feature_cols_: list[str] = []
        self.threshold_: float = 0.5

    @property
    def name(self) -> str:
        return "ReturnLikelihood"

    def _build_features(self, orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        """Join order + customer tables and encode categoricals."""
        df = orders.merge(
            customers[["customer_id", "account_age_days",
                        "lifetime_return_rate", "total_orders"]],
            on="customer_id", how="left",
        )
        df = df.rename(columns={
            "lifetime_return_rate": "customer_lifetime_return_rate",
            "total_orders": "customer_total_orders",
        })
        for col in ["category", "channel"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
        df["account_age_days"] = df["account_age_days"].fillna(365).astype(float)
        df["customer_lifetime_return_rate"] = df["customer_lifetime_return_rate"].fillna(0.12)
        df["customer_total_orders"] = df["customer_total_orders"].fillna(1).astype(float)
        return df

    def fit(
        self,
        orders: pd.DataFrame,
        customers: pd.DataFrame,
        target_col: str = "was_returned",
        val_fraction: float = 0.2,
    ) -> ReturnLikelihoodModel:
        """Train the return likelihood model.

        Parameters
        ----------
        orders:
            Order-level DataFrame including ``was_returned`` label.
        customers:
            Customer-level DataFrame with history features.
        target_col:
            Name of the binary return label column.
        val_fraction:
            Fraction of training data held out for calibration.

        Returns
        -------
        self
        """
        try:
            from lightgbm import LGBMClassifier
            from sklearn.isotonic import IsotonicRegression
        except ImportError as e:
            raise ImportError("lightgbm and scikit-learn required") from e

        feat_df = self._build_features(orders, customers)
        avail_features = [f for f in LIKELIHOOD_FEATURES if f in feat_df.columns]
        self.feature_cols_ = avail_features

        X = feat_df[self.feature_cols_]
        y = feat_df[target_col].astype(int).values

        cat_cols = [c for c in self.feature_cols_ if X[c].dtype.name == "category"]

        # Train/val split for calibration
        n_val = max(1, int(len(X) * val_fraction))
        idx = np.random.default_rng(42).permutation(len(X))
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        fit_kwargs: dict[str, Any] = {}
        if cat_cols:
            fit_kwargs["categorical_feature"] = cat_cols

        self.model_ = LGBMClassifier(**self.lgbm_params)
        self.model_.fit(X.iloc[train_idx], y[train_idx], **fit_kwargs)

        if self.calibrate:
            val_scores = self.model_.predict_proba(X.iloc[val_idx])[:, 1]
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(val_scores, y[val_idx])

        logger.info(
            "ReturnLikelihoodModel trained on %d orders, %d features.",
            len(X), len(self.feature_cols_),
        )
        return self

    def predict_proba(self, orders: pd.DataFrame, customers: pd.DataFrame) -> np.ndarray:
        """Return P(return) for each order.

        Parameters
        ----------
        orders:
            Order-level DataFrame (no label needed).
        customers:
            Customer features.

        Returns
        -------
        np.ndarray
            1-D array of return probabilities in [0, 1].
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        feat_df = self._build_features(orders, customers)
        X = feat_df[self.feature_cols_]
        raw = self.model_.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw

    def predict_with_tier(
        self, orders: pd.DataFrame, customers: pd.DataFrame
    ) -> pd.DataFrame:
        """Return probabilities and risk tiers.

        Returns
        -------
        pd.DataFrame
            Columns: order_id (if present), return_probability, risk_tier.
        """
        proba = self.predict_proba(orders, customers)
        tiers = np.where(
            proba < RISK_THRESHOLDS["low"], "low",
            np.where(proba < RISK_THRESHOLDS["medium"], "medium", "high"),
        )
        result = pd.DataFrame({
            "return_probability": proba.round(4),
            "risk_tier": tiers,
        })
        if "order_id" in orders.columns:
            result.insert(0, "order_id", orders["order_id"].values)
        return result

    def save(self, path: Path | str) -> None:
        """Serialise to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("ReturnLikelihoodModel saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> ReturnLikelihoodModel:
        """Load from disk."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ReturnLikelihoodModel, got {type(obj)}")
        return obj
