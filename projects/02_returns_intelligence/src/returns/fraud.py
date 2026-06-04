"""Returns fraud / abuse detection model.

Detects three fraud archetypes:
- **Wardrober**: buys expensive apparel, returns worn items.
- **Velocity returner**: high order velocity + high return rate.
- **Address-sharing ring**: accounts sharing address/payment hub.

Architecture
------------
1. **Graph features** (networkx): for each return, derive features from a
   bipartite customer-address / customer-payment graph.  Ring members
   show elevated degree and large connected components.
2. **LightGBM classifier** on behavioural + graph features.
3. **Isotonic calibration** so scores are interpretable probabilities.
4. **Cost-aware threshold**: minimises expected cost given FP and FN costs
   rather than using the default 0.5 cut.
5. **SHAP reason codes**: top-2 feature contributions returned per flagged
   return for explainability.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FRAUD_FEATURES = [
    "item_price",
    "days_to_return",
    "customer_lifetime_return_rate",
    "customer_total_orders",
    "customer_total_returns",
    "account_age_days",
    "is_used_condition",
    "is_damaged_condition",
    "is_expensive_category",
    "shared_address_count",
    "shared_payment_count",
    "component_size",
    "category",
    "reason_code",
]

DEFAULT_FP_COST = 5.0
"""Cost of a false positive (analyst review time, customer friction)."""

DEFAULT_FN_COST = 50.0
"""Cost of a false negative (fraud loss, chargeback)."""


# ── Graph feature computation ─────────────────────────────────────────────────


def build_graph_features(
    returns: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Compute graph-based features for every return event.

    Builds a bipartite customer↔address graph and a customer↔payment graph.
    For each customer involved in a return, extracts:
    - ``shared_address_count``: how many other customers share the same address
    - ``shared_payment_count``: how many other customers share the same payment hash
    - ``component_size``: size of the connected component (address + payment graph combined)

    Parameters
    ----------
    returns:
        Return-level DataFrame with ``customer_id``.
    customers:
        Customer-level DataFrame with ``address_id`` and ``payment_hash``.

    Returns
    -------
    pd.DataFrame
        One row per return with graph feature columns added.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx required: pip install networkx") from e

    # Shared-address counts
    addr_counts = (
        customers.groupby("address_id")["customer_id"]
        .transform("count")
        .rename("shared_address_count")
        - 1  # exclude self
    )
    pay_counts = (
        customers.groupby("payment_hash")["customer_id"]
        .transform("count")
        .rename("shared_payment_count")
        - 1
    )
    cust_graph = customers[["customer_id", "address_id", "payment_hash"]].copy()
    cust_graph["shared_address_count"] = addr_counts.values
    cust_graph["shared_payment_count"] = pay_counts.values

    # Build undirected graph: customers connected if they share address OR payment
    G: Any = nx.Graph()
    G.add_nodes_from(customers["customer_id"])

    for _addr, grp in customers.groupby("address_id"):
        cids = grp["customer_id"].tolist()
        if len(cids) > 1:
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    G.add_edge(cids[i], cids[j])

    for _pay, grp in customers.groupby("payment_hash"):
        cids = grp["customer_id"].tolist()
        if len(cids) > 1:
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    G.add_edge(cids[i], cids[j])

    # Component size per customer
    comp_size = {}
    for comp in nx.connected_components(G):
        sz = len(comp)
        for node in comp:
            comp_size[node] = sz

    cust_graph["component_size"] = cust_graph["customer_id"].map(comp_size).fillna(1).astype(int)

    # Merge onto returns
    result = returns.merge(
        cust_graph[
            ["customer_id", "shared_address_count", "shared_payment_count", "component_size"]
        ],
        on="customer_id",
        how="left",
    )
    result[["shared_address_count", "shared_payment_count", "component_size"]] = (
        result[["shared_address_count", "shared_payment_count", "component_size"]]
        .fillna(0)
        .astype(int)
    )
    return result


def _build_fraud_features(
    returns_with_graph: pd.DataFrame,
    orders: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the full fraud feature matrix."""
    # Join order features
    df = returns_with_graph.merge(
        orders[["order_id", "category", "item_price"]],
        on="order_id",
        how="left",
    )
    # Join customer features
    df = df.merge(
        customers[
            [
                "customer_id",
                "account_age_days",
                "lifetime_return_rate",
                "total_orders",
                "total_returns",
            ]
        ],
        on="customer_id",
        how="left",
    )
    df = df.rename(
        columns={
            "lifetime_return_rate": "customer_lifetime_return_rate",
            "total_orders": "customer_total_orders",
            "total_returns": "customer_total_returns",
        }
    )

    # Binary / derived features
    df["is_used_condition"] = (df["condition"] == "used").astype("int8")
    df["is_damaged_condition"] = (df["condition"] == "damaged").astype("int8")
    df["is_expensive_category"] = (
        df["category"].isin(["apparel", "footwear", "electronics"]).astype("int8")
    )

    for col in ["category", "reason_code"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in [
        "customer_lifetime_return_rate",
        "customer_total_orders",
        "customer_total_returns",
        "account_age_days",
        "item_price",
    ]:
        df[col] = df[col].fillna(df[col].median() if len(df) > 0 else 0)

    return df


# ── Model ─────────────────────────────────────────────────────────────────────


class FraudDetectionModel:
    """LightGBM fraud detection with graph features, calibration, and SHAP reasons.

    Parameters
    ----------
    lgbm_params:
        LightGBM hyperparameters.
    fp_cost, fn_cost:
        Per-unit costs for false positives and false negatives.  Used to
        select the operating threshold that minimises expected cost.
    calibrate:
        Apply isotonic calibration after training.

    Attributes
    ----------
    model_:
        Trained LGBMClassifier.
    calibrator_:
        Fitted IsotonicRegression (None if calibrate=False).
    threshold_:
        Operating decision threshold (set by select_threshold() or default 0.5).
    feature_cols_:
        Feature columns used at training time.
    """

    def __init__(
        self,
        lgbm_params: dict[str, Any] | None = None,
        fp_cost: float = DEFAULT_FP_COST,
        fn_cost: float = DEFAULT_FN_COST,
        calibrate: bool = True,
    ) -> None:
        self.lgbm_params = lgbm_params or {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": 10,  # class imbalance correction
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.calibrate = calibrate
        self.model_: Any = None
        self.calibrator_: Any = None
        self.threshold_: float = 0.5
        self.feature_cols_: list[str] = []

    @property
    def name(self) -> str:
        return "FraudDetection"

    def fit(
        self,
        returns: pd.DataFrame,
        orders: pd.DataFrame,
        customers: pd.DataFrame,
        target_col: str = "is_fraud",
        val_fraction: float = 0.2,
        auto_threshold: bool = True,
    ) -> FraudDetectionModel:
        """Train the fraud model.

        Parameters
        ----------
        returns:
            Return-level table with ``is_fraud`` labels.
        orders, customers:
            Supporting tables for feature construction.
        target_col:
            Binary fraud label column.
        val_fraction:
            Fraction held out for calibration and threshold selection.
        auto_threshold:
            If True, call ``select_threshold()`` on the validation set after
            training to choose a cost-optimal operating point.

        Returns
        -------
        self
        """
        try:
            from lightgbm import LGBMClassifier
            from sklearn.isotonic import IsotonicRegression
        except ImportError as e:
            raise ImportError("lightgbm and scikit-learn required") from e

        returns_g = build_graph_features(returns, customers)
        feat_df = _build_fraud_features(returns_g, orders, customers)
        avail = [f for f in FRAUD_FEATURES if f in feat_df.columns]
        self.feature_cols_ = avail

        X = feat_df[self.feature_cols_]
        y = feat_df[target_col].astype(int).values
        cat_cols = [c for c in self.feature_cols_ if X[c].dtype.name == "category"]

        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * val_fraction))
        tr_idx, val_idx = idx[n_val:], idx[:n_val]

        fit_kw: dict[str, Any] = {}
        if cat_cols:
            fit_kw["categorical_feature"] = cat_cols

        self.model_ = LGBMClassifier(**self.lgbm_params)
        self.model_.fit(X.iloc[tr_idx], y[tr_idx], **fit_kw)

        if self.calibrate:
            val_scores = self.model_.predict_proba(X.iloc[val_idx])[:, 1]
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(val_scores, y[val_idx])

        if auto_threshold:
            val_scores_cal = self.predict_proba_raw(X.iloc[val_idx])
            self.threshold_ = self._cost_optimal_threshold(val_scores_cal, y[val_idx])

        logger.info(
            "FraudDetectionModel trained: %d returns, %d features, threshold=%.3f",
            len(X),
            len(self.feature_cols_),
            self.threshold_,
        )
        return self

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw probability predictions (internal)."""
        raw = self.model_.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw

    def _cost_optimal_threshold(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Select threshold minimising E[cost] = FP_cost * FP_rate + FN_cost * FN_rate."""
        thresholds = np.linspace(0.01, 0.99, 200)
        best_cost = float("inf")
        best_t = 0.5
        prevalence = labels.mean()
        for t in thresholds:
            preds = (scores >= t).astype(int)
            fp = ((preds == 1) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
            fn = ((preds == 0) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
            cost = self.fp_cost * fp * (1 - prevalence) + self.fn_cost * fn * prevalence
            if cost < best_cost:
                best_cost = cost
                best_t = float(t)
        return best_t

    def predict(
        self,
        returns: pd.DataFrame,
        orders: pd.DataFrame,
        customers: pd.DataFrame,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Score returns for fraud with SHAP reason codes.

        Parameters
        ----------
        returns, orders, customers:
            Input tables.
        threshold:
            Decision threshold. Uses ``self.threshold_`` if None.

        Returns
        -------
        pd.DataFrame
            Columns: return_id (if present), fraud_probability,
            is_flagged, top_reason_1, top_reason_2.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        t = threshold if threshold is not None else self.threshold_
        returns_g = build_graph_features(returns, customers)
        feat_df = _build_fraud_features(returns_g, orders, customers)
        X = feat_df[self.feature_cols_]
        scores = self.predict_proba_raw(X)

        # SHAP reason codes (fast: use gain importance as a proxy if shap unavailable)
        top_reasons = self._shap_reason_codes(X)

        result = pd.DataFrame(
            {
                "fraud_probability": scores.round(4),
                "is_flagged": scores >= t,
                "top_reason_1": top_reasons[:, 0],
                "top_reason_2": top_reasons[:, 1],
            }
        )
        if "return_id" in returns.columns:
            result.insert(0, "return_id", returns["return_id"].values)
        return result

    def _shap_reason_codes(self, X: pd.DataFrame) -> np.ndarray:
        """Return top-2 feature names per row by absolute SHAP or gain importance."""
        try:
            import shap

            explainer = shap.TreeExplainer(self.model_)
            sv = explainer.shap_values(X)
            shap_vals = sv[1] if isinstance(sv, list) else sv
            top2 = np.argsort(np.abs(shap_vals), axis=1)[:, -2:][:, ::-1]
            cols = np.array(self.feature_cols_)
            return cols[top2]
        except Exception:
            # Fallback: use model gain importance for all rows
            imps = self.model_.booster_.feature_importance("gain")
            top2_idx = np.argsort(imps)[-2:][::-1]
            cols = np.array(self.feature_cols_)
            top2_names = cols[top2_idx]
            return np.tile(top2_names, (len(X), 1))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        logger.info("FraudDetectionModel saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> FraudDetectionModel:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected FraudDetectionModel, got {type(obj)}")
        return obj
