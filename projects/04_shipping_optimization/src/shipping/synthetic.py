"""Synthetic data generator for shipping price optimisation (Project 04).

Simulates a checkout A/B test where each session is randomly assigned one
shipping price from a fixed menu.  Segment-specific price sensitivity means
the optimal price varies by user type — which the elasticity model learns.

Design
------
- Four behavioural segments mirror Project 03's CATE decomposition.
- Each session gets ONE randomly assigned price (A/B test design), so the
  price coefficient in the elasticity model has a causal interpretation.
- Named business features (cart_value, session_depth …) plus six anonymous
  features (f0–f5) for cross-project consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

SHIPPING_COST_TO_MERCHANT: float = 4.50
"""Merchant's fulfilment cost per shipment (USD)."""

PRODUCT_MARGIN_RATE: float = 0.35
"""Gross product margin as a fraction of cart value."""

PRICE_OPTIONS: list[float] = [0.00, 2.99, 4.99, 7.99, 9.99, 12.99]
"""Candidate shipping prices shown in A/B test (USD)."""

SEGMENT_PARAMS: dict[str, dict] = {
    "sure_thing": {
        "weight": 0.25,
        "base_logit": 2.3,
        "price_sensitivity": 0.06,   # barely affected by price
        "cart_value_mean": 105.0,
        "cart_value_std": 55.0,
        "session_depth_mean": 12.0,
        "returning_prob": 0.65,
    },
    "persuadable": {
        "weight": 0.30,
        "base_logit": 0.35,
        "price_sensitivity": 0.28,   # very sensitive — free shipping converts them
        "cart_value_mean": 72.0,
        "cart_value_std": 40.0,
        "session_depth_mean": 9.0,
        "returning_prob": 0.35,
    },
    "lost_cause": {
        "weight": 0.25,
        "base_logit": -2.7,
        "price_sensitivity": 0.02,   # won't convert regardless
        "cart_value_mean": 58.0,
        "cart_value_std": 35.0,
        "session_depth_mean": 4.0,
        "returning_prob": 0.20,
    },
    "sleeping_dog": {
        "weight": 0.20,
        "base_logit": -0.35,
        "price_sensitivity": -0.07,  # counter-intuitive: higher price → slightly higher P(convert)
        "cart_value_mean": 135.0,
        "cart_value_std": 70.0,
        "session_depth_mean": 7.0,
        "returning_prob": 0.50,
    },
}

SEGMENT_NAMES: list[str] = list(SEGMENT_PARAMS.keys())
SEGMENT_WEIGHTS: list[float] = [SEGMENT_PARAMS[s]["weight"] for s in SEGMENT_NAMES]


@dataclass
class ShippingDataConfig:
    """Configuration for the synthetic shipping dataset.

    Parameters
    ----------
    n_sessions:
        Total number of checkout sessions to generate.
    random_state:
        Seed for reproducibility.
    """

    n_sessions: int = 60_000
    random_state: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def generate_shipping_dataset(
    n_sessions: int = 60_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic checkout A/B-test dataset for shipping price optimisation.

    Each row is one checkout session.  The shipping price was randomly assigned
    from ``PRICE_OPTIONS`` (simulating an A/B price experiment), so the
    price–conversion relationship is identifiable causally.

    Parameters
    ----------
    n_sessions:
        Number of sessions to generate.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:

        ``session_id``, ``segment``, ``cart_value``, ``n_items``,
        ``is_returning``, ``session_depth``, ``time_on_checkout``,
        ``device_mobile``, ``f0`` … ``f5``,
        ``shipping_price``, ``converted``, ``cart_margin``.
    """
    rng = np.random.default_rng(random_state)
    n = n_sessions

    # ── Segment assignment ─────────────────────────────────────────────────────
    segments = rng.choice(SEGMENT_NAMES, size=n, p=SEGMENT_WEIGHTS)

    # ── Business features (segment-correlated) ─────────────────────────────────
    cart_value = np.zeros(n)
    session_depth = np.zeros(n)
    is_returning = np.zeros(n, dtype=bool)

    for seg, params in SEGMENT_PARAMS.items():
        mask = segments == seg
        k = mask.sum()
        cart_value[mask] = np.clip(
            rng.normal(params["cart_value_mean"], params["cart_value_std"], k), 15.0, 500.0
        )
        session_depth[mask] = np.clip(
            rng.poisson(params["session_depth_mean"], k), 1, 30
        ).astype(float)
        is_returning[mask] = rng.random(k) < params["returning_prob"]

    n_items = np.clip(rng.poisson(3.0, n), 1, 10)
    time_on_checkout = np.clip(rng.gamma(shape=2.0, scale=60.0, size=n), 10.0, 720.0)
    device_mobile = rng.random(n) < 0.54

    # ── Anonymous features (f0–f5) ─────────────────────────────────────────────
    # f0: correlates with cart_value (normalised)
    f0 = (cart_value - cart_value.mean()) / (cart_value.std() + 1e-8) + rng.normal(0, 0.3, n)
    # f1: correlates with session_depth
    f1 = (session_depth - session_depth.mean()) / (session_depth.std() + 1e-8) + rng.normal(0, 0.3, n)
    # f2: correlates with is_returning
    f2 = is_returning.astype(float) + rng.normal(0, 0.4, n)
    # f3–f5: noise
    f3 = rng.normal(0, 1.0, n)
    f4 = rng.normal(0, 1.0, n)
    f5 = rng.normal(0, 1.0, n)

    # ── Random price assignment (A/B test) ─────────────────────────────────────
    shipping_price = rng.choice(PRICE_OPTIONS, size=n)

    # ── Conversion outcomes ────────────────────────────────────────────────────
    sensitivity = np.array([SEGMENT_PARAMS[s]["price_sensitivity"] for s in segments])
    base_logit = np.array([SEGMENT_PARAMS[s]["base_logit"] for s in segments])

    # Small feature contributions: returning customers and deep sessions convert more
    feature_contrib = (
        0.25 * is_returning.astype(float)
        + 0.015 * (session_depth - 8.0)
        + 0.002 * (cart_value - 85.0)
    )
    logit = base_logit + feature_contrib - sensitivity * shipping_price
    p_convert = _sigmoid(logit)
    converted = rng.random(n) < p_convert

    # ── Derived fields ─────────────────────────────────────────────────────────
    cart_margin = cart_value * PRODUCT_MARGIN_RATE

    return pd.DataFrame(
        {
            "session_id": [f"S{i:07d}" for i in range(n)],
            "segment": segments,
            "cart_value": cart_value.round(2),
            "n_items": n_items,
            "is_returning": is_returning,
            "session_depth": session_depth.astype(int),
            "time_on_checkout": time_on_checkout.round(1),
            "device_mobile": device_mobile,
            "f0": f0.round(4),
            "f1": f1.round(4),
            "f2": f2.round(4),
            "f3": f3.round(4),
            "f4": f4.round(4),
            "f5": f5.round(4),
            "shipping_price": shipping_price,
            "converted": converted,
            "cart_margin": cart_margin.round(2),
        }
    )
