"""Tests for forecasting/inventory.py.

Tests cover the core inventory-math identities (newsvendor critical ratio,
safety-stock z-score scaling), edge cases (zero demand, degenerate intervals),
and the simulation (no-stockout trajectory, ordering cadence, cost accounting).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from forecasting.inventory import (
    Z_80,
    compute_policy_costs,
    compute_sku_policies,
    newsvendor_quantity,
    order_up_to_level,
    reorder_point,
    safety_stock,
    service_level_frontier,
    simulate_inventory,
    std_from_interval,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_preds_interval() -> pd.DataFrame:
    """Minimal predictions-with-intervals DataFrame: 3 SKUs × 28 days."""
    rng = np.random.default_rng(42)
    skus = ["A", "B", "C"]
    n_days = 28
    rows = []
    for sku in skus:
        mean_d = rng.uniform(1, 5)
        for _ in range(n_days):
            f = max(mean_d + rng.normal(0, 0.5), 0)
            hw = rng.uniform(0.5, 2.0)
            rows.append({"id": sku, "forecast": f, "lower_80": max(f - hw, 0), "upper_80": f + hw})
    return pd.DataFrame(rows)


@pytest.fixture()
def constant_demand() -> np.ndarray:
    """Deterministic demand of 5 units/day for 90 days."""
    return np.full(90, 5.0)


# ── std_from_interval ─────────────────────────────────────────────────────────


class TestStdFromInterval:
    def test_symmetric_interval(self) -> None:
        """Half-width / Z_80 should equal the returned std."""
        lower, upper = 2.0, 8.0  # half-width = 3.0
        result = std_from_interval(lower, upper)
        assert result == pytest.approx(3.0 / Z_80, rel=1e-6)

    def test_zero_width_interval_returns_small_positive(self) -> None:
        result = std_from_interval(5.0, 5.0)
        assert result > 0

    def test_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="upper_80"):
            std_from_interval(8.0, 2.0)  # upper < lower


# ── newsvendor_quantity ───────────────────────────────────────────────────────


class TestNewsvendorQuantity:
    def test_symmetric_cost_equals_mean(self) -> None:
        """With equal costs, the optimal quantity is the mean demand."""
        q = newsvendor_quantity(mean_demand=100, std_demand=20, cost_overstock=1.0, cost_understock=1.0)
        assert q == pytest.approx(100.0, abs=1e-6)

    def test_high_understock_cost_above_mean(self) -> None:
        """High stockout penalty pushes Q above the mean."""
        q = newsvendor_quantity(mean_demand=100, std_demand=20, cost_overstock=1.0, cost_understock=9.0)
        assert q > 100.0

    def test_high_overstock_cost_below_mean(self) -> None:
        """High holding cost pushes Q below the mean."""
        q = newsvendor_quantity(mean_demand=100, std_demand=20, cost_overstock=9.0, cost_understock=1.0)
        assert q < 100.0

    def test_zero_std_returns_mean(self) -> None:
        """With no uncertainty, the optimal Q is the mean (clipped to ≥ 0)."""
        q = newsvendor_quantity(mean_demand=50, std_demand=0, cost_overstock=1.0, cost_understock=4.0)
        assert q == pytest.approx(50.0, abs=1e-4)

    def test_non_negative_result(self) -> None:
        """Q* should never be negative even with extreme parameters."""
        q = newsvendor_quantity(mean_demand=0, std_demand=0, cost_overstock=100, cost_understock=1.0)
        assert q >= 0.0

    def test_invalid_cost_raises(self) -> None:
        with pytest.raises(ValueError):
            newsvendor_quantity(100, 10, cost_overstock=-1, cost_understock=5)


# ── safety_stock ──────────────────────────────────────────────────────────────


class TestSafetyStock:
    def test_scales_with_sqrt_lead_time(self) -> None:
        """SS should scale with the square root of lead time."""
        ss1 = safety_stock(std_demand_per_period=10, lead_time_periods=1)
        ss4 = safety_stock(std_demand_per_period=10, lead_time_periods=4)
        assert ss4 == pytest.approx(2 * ss1, rel=1e-6)

    def test_higher_service_level_more_stock(self) -> None:
        ss90 = safety_stock(std_demand_per_period=10, lead_time_periods=7, service_level=0.90)
        ss99 = safety_stock(std_demand_per_period=10, lead_time_periods=7, service_level=0.99)
        assert ss99 > ss90

    def test_zero_std_gives_zero_ss(self) -> None:
        assert safety_stock(0.0, 7) == 0.0

    def test_invalid_service_level_raises(self) -> None:
        with pytest.raises(ValueError):
            safety_stock(5.0, 7, service_level=1.5)


# ── reorder_point ─────────────────────────────────────────────────────────────


class TestReorderPoint:
    def test_equals_lead_time_demand_plus_ss(self) -> None:
        mu, sigma, lt = 5.0, 2.0, 7
        rop = reorder_point(mu, lt, sigma, service_level=0.95)
        ss = safety_stock(sigma, lt, 0.95)
        expected = mu * lt + ss
        assert rop == pytest.approx(expected, rel=1e-6)

    def test_zero_demand_equals_safety_stock(self) -> None:
        rop = reorder_point(0.0, 7, 2.0, service_level=0.95)
        ss = safety_stock(2.0, 7, service_level=0.95)
        assert rop == pytest.approx(ss, rel=1e-6)

    def test_non_negative(self) -> None:
        rop = reorder_point(0.1, 1, 0.01, service_level=0.5)
        assert rop >= 0.0


# ── order_up_to_level ─────────────────────────────────────────────────────────


class TestOrderUpToLevel:
    def test_S_greater_than_s(self) -> None:
        s = reorder_point(5.0, 7, 2.0)
        S = order_up_to_level(s, mean_demand_per_period=5.0, review_period=7)
        assert S > s

    def test_review_period_zero_gives_s(self) -> None:
        s = 20.0
        S = order_up_to_level(s, mean_demand_per_period=5.0, review_period=0)
        assert S == pytest.approx(s, abs=1e-9)


# ── compute_sku_policies ──────────────────────────────────────────────────────


class TestComputeSkuPolicies:
    def test_returns_one_row_per_sku(self, mock_preds_interval: pd.DataFrame) -> None:
        policies = compute_sku_policies(mock_preds_interval, lead_time_days=7)
        assert len(policies) == mock_preds_interval["id"].nunique()

    def test_required_columns_present(self, mock_preds_interval: pd.DataFrame) -> None:
        policies = compute_sku_policies(mock_preds_interval)
        for col in ["reorder_point", "order_up_to", "safety_stock", "newsvendor_q"]:
            assert col in policies.columns

    def test_order_up_to_exceeds_reorder_point(self, mock_preds_interval: pd.DataFrame) -> None:
        policies = compute_sku_policies(mock_preds_interval)
        assert (policies["order_up_to"] > policies["reorder_point"]).all()

    def test_reorder_point_exceeds_safety_stock(self, mock_preds_interval: pd.DataFrame) -> None:
        """ROP includes expected lead-time demand on top of safety stock."""
        policies = compute_sku_policies(mock_preds_interval)
        assert (policies["reorder_point"] >= policies["safety_stock"]).all()

    def test_missing_column_raises(self, mock_preds_interval: pd.DataFrame) -> None:
        df = mock_preds_interval.drop(columns=["lower_80"])
        with pytest.raises(ValueError, match="Missing columns"):
            compute_sku_policies(df)


# ── simulate_inventory ────────────────────────────────────────────────────────


class TestSimulateInventory:
    def test_returns_correct_length(self, constant_demand: np.ndarray) -> None:
        result = simulate_inventory(constant_demand, policy_s=20, policy_S=60, lead_time_days=3)
        assert len(result) == len(constant_demand)

    def test_no_stockout_when_policy_is_generous(self) -> None:
        """With a very high reorder point and large S, no stockouts should occur."""
        demand = np.full(90, 5.0)
        result = simulate_inventory(demand, policy_s=200, policy_S=400, lead_time_days=3)
        assert result["stockout_units"].sum() == 0.0

    def test_on_hand_never_negative(self, constant_demand: np.ndarray) -> None:
        result = simulate_inventory(constant_demand, policy_s=30, policy_S=80, lead_time_days=5)
        assert (result["on_hand_after"] >= -1e-9).all()

    def test_orders_placed_at_correct_rate(self, constant_demand: np.ndarray) -> None:
        """With constant demand and a well-tuned policy, orders should be placed regularly."""
        result = simulate_inventory(constant_demand, policy_s=20, policy_S=60, lead_time_days=3)
        n_orders = (result["order_placed"] > 0).sum()
        assert n_orders > 0

    def test_invalid_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="policy_S"):
            simulate_inventory(np.ones(10), policy_s=50, policy_S=30, lead_time_days=3)


# ── compute_policy_costs ──────────────────────────────────────────────────────


class TestComputePolicyCosts:
    def test_fill_rate_one_when_no_stockouts(self) -> None:
        demand = np.full(30, 5.0)
        sim = simulate_inventory(demand, policy_s=200, policy_S=400, lead_time_days=1)
        costs = compute_policy_costs(sim)
        assert costs["fill_rate"] == pytest.approx(1.0)

    def test_total_cost_equals_sum_of_components(self, constant_demand: np.ndarray) -> None:
        sim = simulate_inventory(constant_demand, policy_s=30, policy_S=80, lead_time_days=3)
        costs = compute_policy_costs(sim)
        expected = costs["total_holding_cost"] + costs["total_stockout_cost"] + costs["total_ordering_cost"]
        assert costs["total_cost"] == pytest.approx(expected, abs=0.01)

    def test_cycle_service_level_in_unit_interval(self, constant_demand: np.ndarray) -> None:
        sim = simulate_inventory(constant_demand, policy_s=30, policy_S=80, lead_time_days=3)
        costs = compute_policy_costs(sim)
        assert 0.0 <= costs["cycle_service_level"] <= 1.0


# ── service_level_frontier ────────────────────────────────────────────────────


class TestServiceLevelFrontier:
    def test_safety_stock_monotone_in_service_level(self) -> None:
        df = service_level_frontier(mean_demand_per_period=5.0, std_demand_per_period=2.0,
                                    lead_time_periods=7)
        assert df["safety_stock"].is_monotonic_increasing

    def test_custom_service_levels(self) -> None:
        sls = [0.80, 0.90, 0.95, 0.99]
        df = service_level_frontier(5.0, 2.0, 7, service_levels=sls)
        assert list(df["service_level"]) == sls
