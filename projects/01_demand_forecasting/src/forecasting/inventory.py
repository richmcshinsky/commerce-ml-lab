"""Inventory optimisation layer: from forecasts to reorder decisions.

Translates LightGBM point forecasts and 80% prediction intervals into
concrete inventory policy parameters using classical operations research models.

Models implemented
------------------
1. **Newsvendor model**
   Optimal single-period order quantity given asymmetric overage / underage costs
   and normally distributed demand.  The critical ratio ``Cu / (Cu + Co)`` maps
   directly to the optimal service level.

2. **Safety stock**
   Buffer stock computed from forecast-error standard deviation and a target
   cycle-service level.  Derived from the 80% quantile interval via
   ``σ = (upper_80 - lower_80) / (2 × 1.282)``.

3. **(s, S) continuous-review policy**
   - Reorder point *s*: place an order whenever on-hand inventory ≤ s.
   - Order-up-to level *S*: each order raises the inventory position to S.
   Parameters are set so that the probability of a stockout during the lead
   time equals ``1 - service_level``.

4. **Inventory simulation**
   Discrete-event simulation under an (s, S) policy with stochastic demand.
   Used to measure empirical service level, average on-hand inventory, and
   total cost.

Design notes
------------
All per-period statistics are in *days*.  Pass ``lead_time_days`` as an
integer number of calendar days.  Demand standard deviation is derived from
the LightGBM 80% prediction interval unless explicitly supplied.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats

    def _norm_ppf(p: float) -> float:
        return float(_scipy_stats.norm.ppf(p))

except ImportError:  # pragma: no cover — scipy always available in production

    def _norm_ppf(p: float) -> float:  # type: ignore[misc]
        """Normal quantile function via stdlib (scipy not available)."""
        from statistics import NormalDist

        return NormalDist().inv_cdf(p)


logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

Z_80: float = 1.282
"""Z-score for an 80% prediction interval (one-tailed p = 0.90)."""

DEFAULT_SERVICE_LEVEL: float = 0.95
"""Default cycle-service level target."""

DEFAULT_HOLDING_COST_PCT: float = 0.25
"""Annual holding cost as a fraction of unit cost (industry default: 25%)."""


# ── Core inventory math ───────────────────────────────────────────────────────


def std_from_interval(
    lower_80: float,
    upper_80: float,
) -> float:
    """Estimate demand standard deviation from an 80% prediction interval.

    Assumes the interval is symmetric around the point forecast and that
    demand is approximately normally distributed within the interval.

    Parameters
    ----------
    lower_80 : float
        10th-percentile forecast (lower bound of 80% PI).
    upper_80 : float
        90th-percentile forecast (upper bound of 80% PI).

    Returns
    -------
    float
        Estimated standard deviation (≥ 0).
    """
    if upper_80 < lower_80:
        raise ValueError(f"upper_80 ({upper_80}) must be ≥ lower_80 ({lower_80})")
    half_width = (upper_80 - lower_80) / 2.0
    return max(half_width / Z_80, 1e-9)


def newsvendor_quantity(
    mean_demand: float,
    std_demand: float,
    cost_overstock: float,
    cost_understock: float,
) -> float:
    """Optimal order quantity from the single-period newsvendor model.

    Minimises expected total cost (overage + underage) given normally
    distributed demand.  The critical ratio is ``Cu / (Cu + Co)``.

    Parameters
    ----------
    mean_demand : float
        Expected demand over the period (e.g. one review cycle or lead time).
    std_demand : float
        Standard deviation of demand over the same period.
    cost_overstock : float
        Per-unit cost of ordering one too many (holding + shrinkage/disposal).
    cost_understock : float
        Per-unit cost of unmet demand (lost margin + goodwill cost).

    Returns
    -------
    float
        Optimal order quantity Q* ≥ 0.

    Notes
    -----
    Q* = μ + z(CR) × σ,  where CR = Cu / (Cu + Co).
    For symmetric costs (Co = Cu), Q* = μ (order the mean).
    For high stockout cost (Cu ≫ Co), Q* > μ (order more than the mean).
    """
    if cost_overstock <= 0:
        raise ValueError("cost_overstock must be positive")
    if cost_understock <= 0:
        raise ValueError("cost_understock must be positive")
    if std_demand < 0:
        raise ValueError("std_demand must be non-negative")

    critical_ratio = cost_understock / (cost_understock + cost_overstock)
    z = _norm_ppf(critical_ratio)
    q_star = mean_demand + z * max(std_demand, 1e-9)
    return max(q_star, 0.0)


def safety_stock(
    std_demand_per_period: float,
    lead_time_periods: int,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> float:
    """Safety stock required to achieve a given cycle-service level.

    Accounts for demand variability accumulated over the lead time under the
    assumption that per-period demands are i.i.d. normally distributed.

    Parameters
    ----------
    std_demand_per_period : float
        Standard deviation of demand per period (e.g. daily).
    lead_time_periods : int
        Replenishment lead time in periods.
    service_level : float
        Target probability of not stocking out during the lead time.
        E.g. 0.95 = 95% cycle service level.

    Returns
    -------
    float
        Safety stock quantity (non-negative).

    Notes
    -----
    SS = z(SL) × σ_day × √L,  where L is lead time in periods.
    """
    if not 0 < service_level < 1:
        raise ValueError(f"service_level must be in (0, 1), got {service_level}")
    if lead_time_periods <= 0:
        raise ValueError(f"lead_time_periods must be ≥ 1, got {lead_time_periods}")

    z = _norm_ppf(service_level)
    ss = z * std_demand_per_period * np.sqrt(lead_time_periods)
    return max(float(ss), 0.0)


def reorder_point(
    mean_demand_per_period: float,
    lead_time_periods: int,
    std_demand_per_period: float,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> float:
    """Reorder point s: the inventory level that triggers a replenishment order.

    Defined as expected demand during the lead time plus safety stock.

    Parameters
    ----------
    mean_demand_per_period : float
        Expected daily demand.
    lead_time_periods : int
        Replenishment lead time in periods.
    std_demand_per_period : float
        Standard deviation of daily demand.
    service_level : float
        Target cycle-service level (default 0.95).

    Returns
    -------
    float
        Reorder point s (non-negative).

    Notes
    -----
    s = μ × L + SS(SL, σ, L) = μL + z(SL) × σ × √L
    """
    if mean_demand_per_period < 0:
        raise ValueError("mean_demand_per_period must be non-negative")

    expected_lead_time_demand = mean_demand_per_period * lead_time_periods
    ss = safety_stock(std_demand_per_period, lead_time_periods, service_level)
    return float(expected_lead_time_demand + ss)


def order_up_to_level(
    s: float,
    mean_demand_per_period: float,
    review_period: int = 7,
) -> float:
    """Order-up-to level S in an (s, S) policy.

    S is set so that after reordering the inventory position covers expected
    demand over the review period beyond the reorder point.

    Parameters
    ----------
    s : float
        Reorder point.
    mean_demand_per_period : float
        Expected daily demand.
    review_period : int
        Target inventory coverage in periods beyond the reorder point.
        Default 7 (one week of stock above the reorder point).

    Returns
    -------
    float
        Order-up-to level S (≥ s).
    """
    return float(s + mean_demand_per_period * review_period)


# ── Batch policy computation ──────────────────────────────────────────────────


def compute_sku_policies(
    preds_interval: pd.DataFrame,
    lead_time_days: int = 7,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    cost_overstock: float = 1.0,
    cost_understock: float = 5.0,
    review_period: int = 7,
    id_col: str = "id",
    forecast_col: str = "forecast",
    lower_col: str = "lower_80",
    upper_col: str = "upper_80",
) -> pd.DataFrame:
    """Compute (s, S) policy parameters for every SKU in a forecast DataFrame.

    Derives demand distribution parameters (μ, σ) from the LightGBM 80%
    prediction interval, then applies the newsvendor and safety-stock formulas
    to produce actionable reorder parameters.

    Parameters
    ----------
    preds_interval : pd.DataFrame
        Output of ``LGBMForecaster.predict_with_intervals()``.
        Expected columns: id_col, forecast_col, lower_col, upper_col.
    lead_time_days : int
        Replenishment lead time in calendar days.
    service_level : float
        Target cycle-service level (e.g. 0.95).
    cost_overstock : float
        Per-unit overage cost (holding + disposal).  Used for newsvendor Q.
    cost_understock : float
        Per-unit underage cost (lost margin + goodwill).  Used for newsvendor Q.
    review_period : int
        Days of coverage to stock above the reorder point (sets S - s).
    id_col, forecast_col, lower_col, upper_col : str
        Column name overrides.

    Returns
    -------
    pd.DataFrame
        One row per SKU with columns:
        id, mean_daily_demand, std_daily_demand,
        safety_stock, reorder_point, order_up_to, newsvendor_q,
        service_level, lead_time_days.
    """
    required = {id_col, forecast_col, lower_col, upper_col}
    missing = required - set(preds_interval.columns)
    if missing:
        raise ValueError(f"Missing columns in preds_interval: {missing}")

    records = []
    grouped = preds_interval.groupby(id_col)

    for sku_id, grp in grouped:
        mean_d = grp[forecast_col].mean()
        # Per-day σ from the mean interval half-width
        mean_lower = grp[lower_col].mean()
        mean_upper = grp[upper_col].mean()
        std_d = std_from_interval(mean_lower, mean_upper)

        ss = safety_stock(std_d, lead_time_days, service_level)
        rop = reorder_point(mean_d, lead_time_days, std_d, service_level)
        oul = order_up_to_level(rop, mean_d, review_period)
        nv_q = newsvendor_quantity(
            mean_demand=mean_d * lead_time_days,
            std_demand=std_d * np.sqrt(lead_time_days),
            cost_overstock=cost_overstock,
            cost_understock=cost_understock,
        )

        records.append(
            {
                id_col: sku_id,
                "mean_daily_demand": round(mean_d, 4),
                "std_daily_demand": round(std_d, 4),
                "safety_stock": round(ss, 2),
                "reorder_point": round(rop, 2),
                "order_up_to": round(oul, 2),
                "newsvendor_q": round(nv_q, 2),
                "service_level": service_level,
                "lead_time_days": lead_time_days,
            }
        )

    df = (
        pd.DataFrame(records)
        .sort_values("mean_daily_demand", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("compute_sku_policies: %d SKUs processed.", len(df))
    return df


# ── Inventory simulation ──────────────────────────────────────────────────────


def simulate_inventory(
    demand_series: np.ndarray,
    policy_s: float,
    policy_S: float,
    lead_time_days: int,
    initial_inventory: float | None = None,
    unit_cost: float = 10.0,
    cost_overstock: float = 1.0,
    cost_understock: float = 5.0,
    ordering_cost: float = 20.0,
) -> pd.DataFrame:
    """Simulate inventory under an (s, S) continuous-review policy.

    Each period (day):
    1. Receive any order placed ``lead_time_days`` ago.
    2. Satisfy demand; any demand beyond on-hand inventory is a stockout.
    3. If inventory position ≤ s, place an order for (S - inventory_position).

    Parameters
    ----------
    demand_series : np.ndarray
        Sequence of daily demand values (e.g. actuals or draws from a
        fitted distribution).
    policy_s : float
        Reorder point.
    policy_S : float
        Order-up-to level.
    lead_time_days : int
        Days from order placement to receipt.
    initial_inventory : float or None
        Starting on-hand inventory.  Defaults to ``policy_S / 2``.
    unit_cost : float
        Unit purchase price (for holding cost computation).
    cost_overstock : float
        Per-unit daily holding cost fraction (e.g. 0.25/365 for 25% p.a.).
    cost_understock : float
        Per-unit penalty for each unit of unmet demand.
    ordering_cost : float
        Fixed cost per replenishment order placed.

    Returns
    -------
    pd.DataFrame
        Day-level simulation results with columns:
        day, demand, on_hand_before, received, on_hand_after,
        stockout_units, order_placed, holding_cost, stockout_cost,
        ordering_cost_daily.
    """
    if policy_S <= policy_s:
        raise ValueError(f"policy_S ({policy_S:.2f}) must be > policy_s ({policy_s:.2f})")

    n = len(demand_series)
    if initial_inventory is None:
        initial_inventory = policy_S / 2.0

    # Pipeline: orders in transit indexed by arrival day
    pipeline: dict[int, float] = {}
    inventory = float(initial_inventory)
    # Inventory position = on-hand + on-order
    on_order = 0.0

    rows = []
    for day in range(n):
        # 1. Receive arrivals scheduled for today
        received = pipeline.pop(day, 0.0)
        inventory += received
        on_order -= received

        # 2. Satisfy demand
        d = float(demand_series[day])
        sold = min(inventory, d)
        stockout = max(d - inventory, 0.0)
        inventory_after = inventory - sold

        # 3. Check reorder condition (use inventory position = on-hand + on-order)
        inv_position = inventory_after + on_order
        order_qty = 0.0
        if inv_position <= policy_s:
            order_qty = policy_S - inv_position
            arrival_day = day + lead_time_days
            pipeline[arrival_day] = pipeline.get(arrival_day, 0.0) + order_qty
            on_order += order_qty

        # 4. Costs
        hold_cost = inventory_after * cost_overstock * unit_cost
        stock_cost = stockout * cost_understock
        ord_cost_daily = ordering_cost if order_qty > 0 else 0.0

        rows.append(
            {
                "day": day,
                "demand": d,
                "on_hand_before": inventory,
                "received": received,
                "on_hand_after": inventory_after,
                "stockout_units": stockout,
                "order_placed": order_qty,
                "holding_cost": hold_cost,
                "stockout_cost": stock_cost,
                "ordering_cost_daily": ord_cost_daily,
            }
        )

        inventory = inventory_after

    return pd.DataFrame(rows)


def compute_policy_costs(
    simulation: pd.DataFrame,
) -> dict[str, float]:
    """Summarise costs and service metrics from a simulation run.

    Parameters
    ----------
    simulation : pd.DataFrame
        Output of :func:`simulate_inventory`.

    Returns
    -------
    dict
        Keys: total_holding_cost, total_stockout_cost, total_ordering_cost,
              total_cost, fill_rate, cycle_service_level, avg_on_hand,
              n_orders, n_stockout_days.
    """
    total_demand = simulation["demand"].sum()
    total_stockout_units = simulation["stockout_units"].sum()
    total_holding = simulation["holding_cost"].sum()
    total_stockout = simulation["stockout_cost"].sum()
    total_ordering = simulation["ordering_cost_daily"].sum()

    fill_rate = 1.0 - (total_stockout_units / total_demand) if total_demand > 0 else 1.0
    cycle_sl = (simulation["stockout_units"] == 0).mean()

    return {
        "total_holding_cost": round(total_holding, 2),
        "total_stockout_cost": round(total_stockout, 2),
        "total_ordering_cost": round(total_ordering, 2),
        "total_cost": round(total_holding + total_stockout + total_ordering, 2),
        "fill_rate": round(fill_rate, 4),
        "cycle_service_level": round(cycle_sl, 4),
        "avg_on_hand": round(simulation["on_hand_after"].mean(), 2),
        "n_orders": int((simulation["order_placed"] > 0).sum()),
        "n_stockout_days": int((simulation["stockout_units"] > 0).sum()),
    }


# ── Service-level trade-off ───────────────────────────────────────────────────


def service_level_frontier(
    mean_demand_per_period: float,
    std_demand_per_period: float,
    lead_time_periods: int,
    service_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Compute the safety-stock vs. service-level trade-off curve.

    Useful for the "efficient frontier" chart showing the cost of higher
    availability commitments.

    Parameters
    ----------
    mean_demand_per_period : float
    std_demand_per_period : float
    lead_time_periods : int
    service_levels : list[float] or None
        Service levels to evaluate.  Default: 0.50 to 0.99 in steps of 0.01.

    Returns
    -------
    pd.DataFrame
        Columns: service_level, safety_stock, reorder_point.
    """
    if service_levels is None:
        service_levels = [round(x, 2) for x in np.arange(0.50, 0.999, 0.01)]

    rows = [
        {
            "service_level": sl,
            "safety_stock": safety_stock(std_demand_per_period, lead_time_periods, sl),
            "reorder_point": reorder_point(
                mean_demand_per_period, lead_time_periods, std_demand_per_period, sl
            ),
        }
        for sl in service_levels
    ]
    return pd.DataFrame(rows)
