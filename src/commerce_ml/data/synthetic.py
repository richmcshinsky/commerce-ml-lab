"""Synthetic e-commerce returns data generator.

Generates a realistic multi-table dataset for the returns intelligence project.
The goal is not perfect realism but *controlled* realism: known archetypes planted
at known rates so model performance can be meaningfully evaluated.

Tables
------
customers : pd.DataFrame
    One row per customer.
orders : pd.DataFrame
    One row per order.
returns : pd.DataFrame
    One row per return event with ground-truth ``is_fraud`` label.

Fraud archetypes
----------------
1. **Wardrober** — buys high-price apparel/footwear, returns worn items.
   Signals: used/damaged condition, expensive category, high return rate.
2. **Velocity returner** — places many orders quickly and returns most.
   Signals: many recent orders, short time-to-return, high return rate.
3. **Address-sharing ring** — cluster of accounts sharing a delivery address
   or payment hash.  Each account looks normal individually; fraud lives on
   the graph.  Signals: shared_address_count > 1, connected component size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    """Configuration for the synthetic data generator."""

    n_customers: int = 20_000
    fraud_rate: float = 0.02
    wardrober_frac: float = 0.008
    velocity_frac: float = 0.005
    ring_frac: float = 0.006
    ring_size: int = 8
    avg_orders_per_customer: float = 3.5
    return_rate_normal: float = 0.12
    random_state: int = 42
    categories: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "apparel",     "avg_price": 65,  "base_return_rate": 0.20},
            {"name": "footwear",    "avg_price": 95,  "base_return_rate": 0.18},
            {"name": "electronics", "avg_price": 180, "base_return_rate": 0.10},
            {"name": "home_decor",  "avg_price": 55,  "base_return_rate": 0.08},
            {"name": "beauty",      "avg_price": 35,  "base_return_rate": 0.05},
        ]
    )


def generate_returns_dataset(
    n_customers: int = 20_000,
    random_state: int = 42,
    config: SyntheticConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic returns dataset with planted fraud archetypes.

    Parameters
    ----------
    n_customers:
        Number of unique customers to simulate.
    random_state:
        Random seed for reproducibility.
    config:
        Optional ``SyntheticConfig``. If ``None``, defaults are used with
        ``n_customers`` and ``random_state`` overridden.

    Returns
    -------
    customers, orders, returns : pd.DataFrame
        Three related DataFrames.

        customers columns:
            customer_id, address_id, payment_hash, account_age_days,
            total_orders, total_returns, lifetime_return_rate, archetype

        orders columns:
            order_id, customer_id, item_id, category, item_price, quantity,
            channel, order_date, was_returned

        returns columns:
            return_id, order_id, customer_id, return_date, days_to_return,
            reason_code, condition, exchange_requested, is_fraud
    """
    if config is None:
        config = SyntheticConfig(n_customers=n_customers, random_state=random_state)

    rng = np.random.default_rng(config.random_state)
    n = config.n_customers

    # ── Assign archetypes ─────────────────────────────────────────────────────
    n_wardrobers  = int(n * config.wardrober_frac)
    n_velocity    = int(n * config.velocity_frac)
    n_ring_budget = int(n * config.ring_frac)

    # Randomise ring sizes between 3 and ring_size*2 so the network chart
    # shows a realistic distribution rather than a single spike at ring_size.
    ring_sizes: list[int] = []
    remaining = n_ring_budget
    while remaining >= 3:
        sz = min(int(rng.integers(3, config.ring_size * 2 + 1)), remaining)
        ring_sizes.append(sz)
        remaining -= sz
    n_ring_total = sum(ring_sizes)
    n_normal     = n - n_wardrobers - n_velocity - n_ring_total

    archetypes: list[str] = (
        ["normal"]    * n_normal
        + ["wardrober"] * n_wardrobers
        + ["velocity"]  * n_velocity
        + ["ring"]      * n_ring_total
    )
    rng.shuffle(archetypes)  # type: ignore[arg-type]
    customer_ids = [f"C{i:06d}" for i in range(n)]

    # ── Address / payment assignment ──────────────────────────────────────────
    address_ids  = [f"A{i:06d}" for i in range(n)]
    payment_hash = [f"P{i:06d}" for i in range(n)]
    ring_idx     = [i for i, a in enumerate(archetypes) if a == "ring"]
    cursor = 0
    for ring_num, sz in enumerate(ring_sizes):
        members = ring_idx[cursor: cursor + sz]
        cursor += sz
        shared_addr = f"RING_A{ring_num:04d}"
        shared_pay  = f"RING_P{ring_num:04d}"
        for m in members:
            address_ids[m]  = shared_addr
            payment_hash[m] = shared_pay

    customers = pd.DataFrame({
        "customer_id":      customer_ids,
        "address_id":       address_ids,
        "payment_hash":     payment_hash,
        "account_age_days": rng.integers(30, 3650, size=n).tolist(),
        "archetype":        archetypes,
    })

    # ── Generate orders ───────────────────────────────────────────────────────
    cat_names  = [c["name"] for c in config.categories]
    cat_prices = [c["avg_price"] for c in config.categories]
    cat_rr     = [c["base_return_rate"] for c in config.categories]
    cat_weight = np.array([0.35, 0.20, 0.20, 0.15, 0.10])
    start_date = pd.Timestamp("2023-01-01")
    total_days = 730  # 2 years

    order_rows: list[dict] = []
    for i, cust_id in enumerate(customer_ids):
        arch = archetypes[i]
        n_orders = max(1, int(rng.poisson(
            lam=8.0 if arch == "velocity" else config.avg_orders_per_customer
        )))
        for _ in range(n_orders):
            w = np.array([0.50, 0.35, 0.05, 0.05, 0.05]) if arch == "wardrober" else cat_weight
            cat_idx = int(rng.choice(len(cat_names), p=w / w.sum()))
            cat     = cat_names[cat_idx]
            price   = round(float(np.clip(rng.lognormal(np.log(cat_prices[cat_idx]), 0.3), 5, 999)), 2)

            if arch == "velocity":
                day_offset = int(rng.integers(0, 90))
            else:
                day_offset = int(rng.integers(0, total_days))
            order_date = start_date + pd.Timedelta(days=day_offset)

            channel  = str(rng.choice(["web", "mobile", "marketplace"], p=[0.55, 0.35, 0.10]))
            quantity = int(rng.choice([1, 2, 3], p=[0.80, 0.15, 0.05]))

            base_rr = cat_rr[cat_idx]
            if arch == "wardrober" and cat in ("apparel", "footwear"):
                return_prob = min(0.85, base_rr * 3.0)
            elif arch == "velocity":
                return_prob = min(0.75, base_rr * 2.5)
            elif arch == "ring":
                return_prob = min(0.40, base_rr * 1.5)
            else:
                return_prob = base_rr * float(rng.uniform(0.7, 1.3))

            order_rows.append({
                "customer_id":  cust_id,
                "category":     cat,
                "item_price":   price,
                "quantity":     quantity,
                "channel":      channel,
                "order_date":   order_date,
                "was_returned": bool(rng.random() < return_prob),
                "_arch":        arch,
                "_cat_idx":     cat_idx,
            })

    orders = pd.DataFrame(order_rows)
    orders.insert(0, "order_id", [f"O{i:07d}" for i in range(len(orders))])
    orders["item_id"] = (
        orders["category"]
        + "_"
        + (orders.groupby("category").cumcount() % 300).astype(str).str.zfill(3)
    )

    # Customer-level summaries
    cust_stats = (
        orders.groupby("customer_id")
        .agg(total_orders=("order_id", "count"), total_returns=("was_returned", "sum"))
        .reset_index()
    )
    cust_stats["lifetime_return_rate"] = (
        cust_stats["total_returns"] / cust_stats["total_orders"].clip(lower=1)
    )
    customers = customers.merge(cust_stats, on="customer_id", how="left")
    customers[["total_orders", "total_returns"]] = (
        customers[["total_orders", "total_returns"]].fillna(0).astype(int)
    )
    customers["lifetime_return_rate"] = customers["lifetime_return_rate"].fillna(0.0)

    # ── Generate return events ────────────────────────────────────────────────
    reason_map = {
        "apparel":     ["too_small", "too_large", "wrong_color", "changed_mind", "defective"],
        "footwear":    ["too_small", "too_large", "defective", "changed_mind"],
        "electronics": ["defective", "not_as_described", "changed_mind"],
        "home_decor":  ["changed_mind", "not_as_described", "defective"],
        "beauty":      ["changed_mind", "defective", "not_as_described"],
    }
    returned = orders[orders["was_returned"]].copy()
    return_rows: list[dict] = []

    for _, row in returned.iterrows():
        arch = row["_arch"]
        cat  = row["category"]
        order_date = pd.to_datetime(row["order_date"])
        price      = float(row["item_price"])

        if arch == "wardrober":
            days_to_return = int(rng.integers(15, 45))
        elif arch == "velocity":
            days_to_return = int(rng.integers(1, 8))
        else:
            days_to_return = int(rng.integers(1, 30))

        possible_reasons = reason_map.get(cat, ["changed_mind"])
        reason = str(
            rng.choice(["changed_mind", "defective"]) if arch == "wardrober"
            else rng.choice(possible_reasons)
        )

        if arch == "wardrober":
            condition = str(rng.choice(["used", "damaged"], p=[0.7, 0.3]))
        elif arch == "velocity":
            condition = str(rng.choice(["new", "used"], p=[0.6, 0.4]))
        else:
            condition = str(rng.choice(["new", "used", "damaged"], p=[0.75, 0.20, 0.05]))

        is_fraud = False
        if arch == "wardrober" and cat in ("apparel", "footwear") and price > 60:
            is_fraud = bool(rng.random() < 0.80)
        elif arch == "velocity":
            is_fraud = bool(rng.random() < 0.70)
        elif arch == "ring":
            is_fraud = bool(rng.random() < 0.50)

        return_rows.append({
            "order_id":           row["order_id"],
            "customer_id":        row["customer_id"],
            "return_date":        order_date + pd.Timedelta(days=days_to_return),
            "days_to_return":     days_to_return,
            "reason_code":        reason,
            "condition":          condition,
            "exchange_requested": bool(rng.random() < 0.25),
            "is_fraud":           is_fraud,
        })

    returns_df = pd.DataFrame(return_rows)
    returns_df.insert(0, "return_id", [f"R{i:07d}" for i in range(len(returns_df))])
    orders = orders.drop(columns=["_arch", "_cat_idx"])

    fraud_rate = float(returns_df["is_fraud"].mean()) if len(returns_df) else 0.0
    logger.info(
        "Generated %d customers, %d orders, %d returns (fraud=%.2f%%)",
        len(customers), len(orders), len(returns_df), fraud_rate * 100,
    )
    return customers, orders, returns_df
