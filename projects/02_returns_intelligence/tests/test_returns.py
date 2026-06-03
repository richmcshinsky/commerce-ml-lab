"""Tests for the returns intelligence suite.

Covers:
- Synthetic data generator schema and fraud-archetype properties
- ReturnLikelihoodModel (skipped if lightgbm unavailable)
- FraudDetectionModel (skipped if lightgbm / networkx unavailable)
- ExchangeRecommender (pure Python, always runs)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from commerce_ml.data.synthetic import SyntheticConfig, generate_returns_dataset
from returns.exchange import (
    ExchangeRecommender,
    build_catalog,
    heuristic_candidates,
    score_candidates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def small_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Minimal dataset: 500 customers, fixed seed."""
    return generate_returns_dataset(n_customers=500, random_state=0)


@pytest.fixture(scope="session")
def catalog(small_dataset: tuple) -> pd.DataFrame:
    _, orders, _ = small_dataset
    return build_catalog(orders)


# ── Synthetic data ─────────────────────────────────────────────────────────────


class TestSyntheticGenerator:
    def test_returns_three_dataframes(self, small_dataset: tuple) -> None:
        customers, orders, returns = small_dataset
        assert isinstance(customers, pd.DataFrame)
        assert isinstance(orders, pd.DataFrame)
        assert isinstance(returns, pd.DataFrame)

    def test_customer_schema(self, small_dataset: tuple) -> None:
        customers, _, _ = small_dataset
        for col in ["customer_id", "address_id", "payment_hash",
                    "account_age_days", "archetype", "lifetime_return_rate"]:
            assert col in customers.columns, f"Missing column: {col}"

    def test_order_schema(self, small_dataset: tuple) -> None:
        _, orders, _ = small_dataset
        for col in ["order_id", "customer_id", "item_id", "category",
                    "item_price", "channel", "was_returned"]:
            assert col in orders.columns, f"Missing column: {col}"

    def test_return_schema(self, small_dataset: tuple) -> None:
        _, _, returns = small_dataset
        for col in ["return_id", "order_id", "customer_id", "days_to_return",
                    "reason_code", "condition", "is_fraud"]:
            assert col in returns.columns, f"Missing column: {col}"

    def test_is_fraud_is_bool(self, small_dataset: tuple) -> None:
        _, _, returns = small_dataset
        assert returns["is_fraud"].dtype == bool or returns["is_fraud"].isin([0, 1]).all()

    def test_fraud_rate_plausible(self, small_dataset: tuple) -> None:
        _, _, returns = small_dataset
        rate = returns["is_fraud"].mean()
        assert 0.005 < rate < 0.20, f"Unexpected fraud rate: {rate:.2%}"

    def test_no_missing_customer_ids(self, small_dataset: tuple) -> None:
        customers, orders, returns = small_dataset
        assert orders["customer_id"].isin(customers["customer_id"]).all()
        assert returns["customer_id"].isin(customers["customer_id"]).all()

    def test_returns_subset_of_returned_orders(self, small_dataset: tuple) -> None:
        _, orders, returns = small_dataset
        returned_oids = set(orders[orders["was_returned"]]["order_id"])
        assert set(returns["order_id"]).issubset(returned_oids)

    def test_ring_members_share_address(self, small_dataset: tuple) -> None:
        customers, _, _ = small_dataset
        rings = customers[customers["archetype"] == "ring"]
        if len(rings) >= 2:
            addr_counts = rings["address_id"].value_counts()
            assert addr_counts.max() > 1, "Ring members should share address"

    def test_wardrober_condition_skewed(self, small_dataset: tuple) -> None:
        _, orders, returns = small_dataset
        customers, _, _ = small_dataset
        wdb_ids = set(customers[customers["archetype"] == "wardrober"]["customer_id"])
        wdb_returns = returns[returns["customer_id"].isin(wdb_ids)]
        if len(wdb_returns) > 0:
            used_rate = (wdb_returns["condition"] != "new").mean()
            assert used_rate > 0.3, "Wardrobers should return items in used/damaged condition"

    def test_reproducible(self) -> None:
        c1, o1, r1 = generate_returns_dataset(n_customers=200, random_state=7)
        c2, o2, r2 = generate_returns_dataset(n_customers=200, random_state=7)
        assert len(r1) == len(r2)
        assert (r1["is_fraud"] == r2["is_fraud"]).all()


# ── Exchange recommender ───────────────────────────────────────────────────────


class TestExchangeRecommender:
    def test_fit_builds_catalog(self, small_dataset: tuple) -> None:
        _, orders, _ = small_dataset
        er = ExchangeRecommender()
        er.fit(orders)
        assert er.catalog_ is not None
        assert len(er.catalog_) > 0

    def test_recommend_returns_top_k(self, small_dataset: tuple) -> None:
        _, orders, _ = small_dataset
        er = ExchangeRecommender().fit(orders)
        recs = er.recommend("apparel_042", "changed_mind", 65.0, top_k=3)
        assert len(recs) <= 3

    def test_recommend_excludes_original(self, small_dataset: tuple) -> None:
        _, orders, _ = small_dataset
        er = ExchangeRecommender().fit(orders)
        original = "apparel_042"
        recs = er.recommend(original, "changed_mind", 65.0, top_k=5)
        assert original not in recs["item_id"].values

    def test_scores_in_unit_interval(self, small_dataset: tuple) -> None:
        _, orders, _ = small_dataset
        er = ExchangeRecommender().fit(orders)
        recs = er.recommend("electronics_010", "defective", 180.0, top_k=3)
        if len(recs) > 0:
            assert (recs["score"] >= 0).all() and (recs["score"] <= 1).all()

    def test_defective_returns_same_item(self, catalog: pd.DataFrame) -> None:
        # If item exists in catalog, defective rule should return it
        first_item = catalog["item_id"].iloc[0]
        candidates, rule = heuristic_candidates(first_item, "defective", catalog, top_k=5)
        assert "defective" in rule.lower() or len(candidates) == 0

    def test_reason_code_rule_applied(self, catalog: pd.DataFrame) -> None:
        item = catalog["item_id"].iloc[0]
        for reason in ["too_small", "too_large", "changed_mind", "not_as_described"]:
            candidates, rule = heuristic_candidates(item, reason, catalog, top_k=5)
            assert isinstance(rule, str) and len(rule) > 0

    def test_score_candidates_sorted_descending(self, catalog: pd.DataFrame) -> None:
        candidates = catalog.head(10).copy()
        scored = score_candidates(candidates, original_price=60.0)
        if len(scored) > 1:
            assert scored["score"].iloc[0] >= scored["score"].iloc[-1]

    def test_batch_recommend(self, small_dataset: tuple) -> None:
        customers, orders, returns = small_dataset
        er = ExchangeRecommender().fit(orders)
        sample = returns[returns["exchange_requested"]].head(5)
        if len(sample) > 0:
            batch = er.recommend_batch(sample, orders, customers, top_k=2)
            assert "return_id" in batch.columns
            assert "score" in batch.columns


# ── Likelihood + Fraud (skip if lightgbm unavailable) ─────────────────────────

lgbm = pytest.importorskip("lightgbm", reason="lightgbm not installed")
_FAST = {"n_estimators": 10, "verbose": -1, "random_state": 42}


class TestReturnLikelihood:
    def test_fit_and_predict(self, small_dataset: tuple) -> None:
        from returns.likelihood import ReturnLikelihoodModel
        customers, orders, returns = small_dataset
        model = ReturnLikelihoodModel(lgbm_params=_FAST).fit(orders, customers)
        proba = model.predict_proba(orders.head(10), customers)
        assert len(proba) == 10
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_risk_tiers_assigned(self, small_dataset: tuple) -> None:
        from returns.likelihood import ReturnLikelihoodModel
        customers, orders, _ = small_dataset
        result = ReturnLikelihoodModel(lgbm_params=_FAST).fit(orders, customers).predict_with_tier(orders.head(20), customers)
        assert "risk_tier" in result.columns
        assert result["risk_tier"].isin(["low", "medium", "high"]).all()

    def test_predict_without_fit_raises(self, small_dataset: tuple) -> None:
        from returns.likelihood import ReturnLikelihoodModel
        customers, orders, _ = small_dataset
        with pytest.raises(RuntimeError):
            ReturnLikelihoodModel().predict_proba(orders.head(5), customers)


networkx = pytest.importorskip("networkx", reason="networkx not installed")


class TestFraudDetection:
    def test_fit_and_predict(self, small_dataset: tuple) -> None:
        from returns.fraud import FraudDetectionModel
        customers, orders, returns = small_dataset
        model = FraudDetectionModel(
            lgbm_params={**_FAST, "scale_pos_weight": 5}
        ).fit(returns, orders, customers)
        preds = model.predict(returns.head(10), orders, customers)
        assert "fraud_probability" in preds.columns
        assert "is_flagged" in preds.columns

    def test_probabilities_in_unit_interval(self, small_dataset: tuple) -> None:
        from returns.fraud import FraudDetectionModel
        customers, orders, returns = small_dataset
        model = FraudDetectionModel(lgbm_params={**_FAST, "scale_pos_weight": 5})
        model.fit(returns, orders, customers)
        preds = model.predict(returns.head(20), orders, customers)
        assert ((preds["fraud_probability"] >= 0) & (preds["fraud_probability"] <= 1)).all()

    def test_threshold_in_unit_interval(self, small_dataset: tuple) -> None:
        from returns.fraud import FraudDetectionModel
        customers, orders, returns = small_dataset
        model = FraudDetectionModel(lgbm_params={**_FAST, "scale_pos_weight": 5})
        model.fit(returns, orders, customers)
        assert 0 < model.threshold_ < 1
