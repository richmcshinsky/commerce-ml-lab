"""Inventory optimisation layer: from forecasts to reorder decisions.

Translates point forecasts and prediction intervals into inventory policy
parameters using classical operations research models:

- Newsvendor model: optimal order quantity given demand uncertainty and
  asymmetric costs (overage cost vs. underage/stockout cost).
- Safety stock: buffer stock computed from forecast error standard deviation
  and a target service level.
- (s, S) reorder policy: reorder to S when inventory drops below s.

This layer is what separates "I can forecast demand" from "I can actually
help you run inventory operations."
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def newsvendor_quantity(
    mean_demand: float,
    std_demand: float,
    cost_overstock: float,
    cost_understock: float,
) -> float:
    """Optimal order quantity from the newsvendor model.

    Minimises expected cost given normally distributed demand and asymmetric
    overage/underage costs. The critical ratio is Cu / (Cu + Co).

    Parameters
    ----------
    mean_demand : float
        Expected demand over the lead time.
    std_demand : float
        Standard deviation of demand over the lead time.
    cost_overstock : float
        Cost per unit ordered but not sold (holding + disposal).
    cost_understock : float
        Cost per unit of unmet demand (lost margin + goodwill).

    Returns
    -------
    float
        Optimal order quantity.
    """
    # Placeholder — full implementation in project notebooks
    critical_ratio = cost_understock / (cost_understock + cost_overstock)
    return float(stats.norm.ppf(critical_ratio, loc=mean_demand, scale=std_demand))


def safety_stock(
    std_demand_per_period: float,
    lead_time_periods: int,
    service_level: float = 0.95,
) -> float:
    """Safety stock for a given service level.

    Parameters
    ----------
    std_demand_per_period : float
        Standard deviation of demand per period.
    lead_time_periods : int
        Lead time in periods (e.g. days).
    service_level : float
        Target in-stock probability, e.g. 0.95 = 95%.

    Returns
    -------
    float
        Safety stock quantity (rounded up in practice).
    """
    z = stats.norm.ppf(service_level)
    return float(z * std_demand_per_period * np.sqrt(lead_time_periods))
