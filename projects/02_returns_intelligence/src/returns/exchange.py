"""Exchange recommendation engine.

Given a return event and a stated reason code, recommends the best
exchange candidates to offer the customer.

Architecture
------------
Two-stage pipeline:

1. **Heuristic filter** (reason-code rules, ~70% of cases):
   - ``too_small``        → same item, next size up
   - ``too_large``        → same item, next size down
   - ``wrong_color``      → same item family, all available colors
   - ``defective``        → exact replacement (same item_id)
   - ``changed_mind``     → same category, top-N by popularity
   - ``not_as_described`` → same category, top-N by popularity

2. **Scoring** (LightGBM ranker or feature-weighted score):
   Ranks the heuristic-filtered candidates using item popularity,
   price proximity to the returned item, and category affinity.

This design makes the common cases fast and interpretable (the heuristic
explains itself in plain English) while using the ML ranker to fine-tune
ordering when multiple candidates exist.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Size ordering for size-based rules ────────────────────────────────────────
SIZE_ORDER: list[str] = ["XS", "S", "M", "L", "XL", "XXL", "2XL", "3XL",
                          "5", "6", "7", "8", "9", "10", "11", "12",
                          "28", "30", "32", "34", "36", "38", "40", "42"]

_SIZE_RANK = {s: i for i, s in enumerate(SIZE_ORDER)}


def _next_size(current_size: str, direction: int) -> str | None:
    """Return the adjacent size (+1 or -1). Returns None if at boundary."""
    rank = _SIZE_RANK.get(current_size)
    if rank is None:
        return None
    new_rank = rank + direction
    if 0 <= new_rank < len(SIZE_ORDER):
        return SIZE_ORDER[new_rank]
    return None


# ── Heuristic catalog helpers ─────────────────────────────────────────────────


def _parse_item_id(item_id: str) -> tuple[str, str, str]:
    """Parse item_id into (category, sku_base, variant).

    Convention used by the synthetic generator: ``{category}_{sku_base}_{variant}``
    e.g. ``apparel_042_M_blue`` → category=apparel, base=042, variant=M_blue.
    Falls back gracefully for simpler IDs.
    """
    parts = item_id.split("_", 2)
    category = parts[0] if len(parts) > 0 else "unknown"
    sku_base = parts[1] if len(parts) > 1 else item_id
    variant  = parts[2] if len(parts) > 2 else ""
    return category, sku_base, variant


def build_catalog(orders: pd.DataFrame) -> pd.DataFrame:
    """Build a simple item catalog from orders history.

    Returns one row per (item_id, category) with:
    - ``popularity``: total units sold
    - ``avg_price``: average item price
    """
    catalog = (
        orders.groupby(["item_id", "category"])
        .agg(popularity=("quantity", "sum"), avg_price=("item_price", "mean"))
        .reset_index()
    )
    catalog["popularity_rank"] = catalog.groupby("category")["popularity"].rank(
        ascending=False, method="min"
    )
    return catalog


# ── Heuristic filter ──────────────────────────────────────────────────────────


def heuristic_candidates(
    item_id: str,
    reason_code: str,
    catalog: pd.DataFrame,
    top_k: int = 10,
) -> tuple[pd.DataFrame, str]:
    """Apply reason-code heuristic to generate exchange candidates.

    Parameters
    ----------
    item_id:
        The returned item's ID.
    reason_code:
        Return reason (too_small / too_large / wrong_color / defective /
        changed_mind / not_as_described).
    catalog:
        Item catalog from ``build_catalog()``.
    top_k:
        Maximum candidates to return before scoring.

    Returns
    -------
    candidates : pd.DataFrame
        Subset of catalog rows.
    rule_applied : str
        Human-readable description of the rule used.
    """
    category, sku_base, variant = _parse_item_id(item_id)
    cat_items = catalog[catalog["category"] == category].copy()

    if reason_code == "defective":
        candidates = catalog[catalog["item_id"] == item_id].copy()
        rule = "Exact replacement (same item, defective return)"

    elif reason_code in ("too_small", "too_large"):
        # Try to find same base SKU with adjacent size
        direction = 1 if reason_code == "too_small" else -1
        next_sz = _next_size(variant, direction)
        if next_sz is not None:
            target_id = f"{category}_{sku_base}_{next_sz}"
            candidates = catalog[catalog["item_id"] == target_id].copy()
            rule = f"Same style, {'larger' if direction > 0 else 'smaller'} size"
        else:
            candidates = cat_items.head(top_k).copy()
            rule = "Same category (size boundary reached)"

        if candidates.empty:
            # Fallback: same category top items
            candidates = cat_items.head(top_k).copy()
            rule += " → fallback to category top"

    elif reason_code == "wrong_color":
        same_base = catalog[
            catalog["item_id"].str.startswith(f"{category}_{sku_base}")
            & (catalog["item_id"] != item_id)
        ].copy()
        candidates = same_base if not same_base.empty else cat_items.head(top_k)
        rule = "Same style, available colors"

    else:  # changed_mind, not_as_described, or unknown
        candidates = cat_items.nsmallest(top_k, "popularity_rank").copy()
        rule = f"Same category ({category}), top by popularity"

    # Exclude the original item
    candidates = candidates[candidates["item_id"] != item_id]
    return candidates.head(top_k).reset_index(drop=True), rule


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_candidates(
    candidates: pd.DataFrame,
    original_price: float,
    customer_return_rate: float = 0.12,
    w_popularity: float = 0.5,
    w_price_proximity: float = 0.3,
    w_popularity_rank: float = 0.2,
) -> pd.DataFrame:
    """Score exchange candidates using a weighted feature formula.

    Features:
    - ``popularity``: normalised popularity (higher = better)
    - ``price_proximity``: 1 - |price_delta| / original_price (closer = better)
    - ``popularity_rank``: inverse rank (lower rank number = higher score)

    Parameters
    ----------
    candidates:
        Catalog rows from ``heuristic_candidates()``.
    original_price:
        Price of the returned item.
    customer_return_rate:
        Customer's historical return rate (used to down-weight risky candidates
        — not currently implemented but documented for future use).
    w_popularity, w_price_proximity, w_popularity_rank:
        Scoring weights (must sum to ~1).

    Returns
    -------
    pd.DataFrame
        Candidates sorted by ``score`` descending, with score column added.
    """
    if candidates.empty:
        return candidates

    df = candidates.copy()

    # Normalised popularity (0-1)
    max_pop = df["popularity"].max() if df["popularity"].max() > 0 else 1
    df["pop_norm"] = df["popularity"] / max_pop

    # Price proximity (0-1; 1 = same price)
    df["price_prox"] = 1.0 - (df["avg_price"] - original_price).abs().clip(upper=original_price) / (original_price + 1e-6)

    # Rank-based score (0-1)
    max_rank = df["popularity_rank"].max() if df["popularity_rank"].max() > 0 else 1
    df["rank_score"] = 1.0 - (df["popularity_rank"] - 1) / max_rank

    df["score"] = (
        w_popularity       * df["pop_norm"]
        + w_price_proximity  * df["price_prox"]
        + w_popularity_rank  * df["rank_score"]
    ).round(4)

    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ── Main recommender ──────────────────────────────────────────────────────────


class ExchangeRecommender:
    """Two-stage exchange recommendation: heuristic filter + feature scoring.

    Parameters
    ----------
    w_popularity, w_price_proximity, w_popularity_rank:
        Weights for the scoring formula.

    Attributes
    ----------
    catalog_:
        Item catalog built from training orders.
    """

    def __init__(
        self,
        w_popularity: float = 0.5,
        w_price_proximity: float = 0.3,
        w_popularity_rank: float = 0.2,
    ) -> None:
        self.w_popularity = w_popularity
        self.w_price_proximity = w_price_proximity
        self.w_popularity_rank = w_popularity_rank
        self.catalog_: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "ExchangeRecommender"

    def fit(self, orders: pd.DataFrame) -> ExchangeRecommender:
        """Build the item catalog from training orders.

        Parameters
        ----------
        orders:
            Order-level DataFrame used to compute item popularity.

        Returns
        -------
        self
        """
        self.catalog_ = build_catalog(orders)
        logger.info(
            "ExchangeRecommender catalog built: %d items, %d categories.",
            len(self.catalog_), self.catalog_["category"].nunique(),
        )
        return self

    def recommend(
        self,
        item_id: str,
        reason_code: str,
        original_price: float,
        customer_return_rate: float = 0.12,
        top_k: int = 3,
    ) -> pd.DataFrame:
        """Recommend exchange candidates for a single return.

        Parameters
        ----------
        item_id:
            The returned item's ID.
        reason_code:
            Return reason code.
        original_price:
            Price of the returned item.
        customer_return_rate:
            Customer's historical return rate.
        top_k:
            Number of candidates to return.

        Returns
        -------
        pd.DataFrame
            Top-K candidates with columns: item_id, score, rule_applied,
            avg_price, popularity.
        """
        if self.catalog_ is None:
            raise RuntimeError("Call fit() before recommend().")

        candidates, rule = heuristic_candidates(
            item_id, reason_code, self.catalog_, top_k=top_k * 3
        )
        scored = score_candidates(
            candidates, original_price, customer_return_rate,
            self.w_popularity, self.w_price_proximity, self.w_popularity_rank,
        )
        if scored.empty or "score" not in scored.columns:
            return pd.DataFrame(columns=["item_id", "score", "avg_price", "popularity", "rule_applied"])
        top = scored.head(top_k)[["item_id", "score", "avg_price", "popularity"]].copy()
        top["rule_applied"] = rule
        return top.reset_index(drop=True)

    def recommend_batch(
        self,
        returns: pd.DataFrame,
        orders: pd.DataFrame,
        customers: pd.DataFrame,
        top_k: int = 3,
    ) -> pd.DataFrame:
        """Recommend exchanges for a batch of return events.

        Parameters
        ----------
        returns:
            Return-level DataFrame with ``reason_code``.
        orders:
            Used to look up ``item_price`` for each returned order.
        customers:
            Used to look up customer return rate.
        top_k:
            Candidates per return.

        Returns
        -------
        pd.DataFrame
            Columns: return_id, rank (1-top_k), item_id, score, rule_applied.
        """
        # Join item_price and item_id from orders
        ret = returns.merge(
            orders[["order_id", "item_id", "item_price"]],
            on="order_id", how="left",
        ).merge(
            customers[["customer_id", "lifetime_return_rate"]],
            on="customer_id", how="left",
        )
        ret["lifetime_return_rate"] = ret["lifetime_return_rate"].fillna(0.12)

        rows = []
        for _, r in ret.iterrows():
            recs = self.recommend(
                item_id=str(r.get("item_id", "unknown")),
                reason_code=str(r.get("reason_code", "changed_mind")),
                original_price=float(r.get("item_price", 50.0)),
                customer_return_rate=float(r.get("lifetime_return_rate", 0.12)),
                top_k=top_k,
            )
            for rank_i, rec_row in recs.iterrows():
                rows.append({
                    "return_id":    r.get("return_id", ""),
                    "rank":         rank_i + 1,
                    "item_id":      rec_row["item_id"],
                    "score":        rec_row["score"],
                    "rule_applied": rec_row["rule_applied"],
                })

        return pd.DataFrame(rows)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> ExchangeRecommender:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ExchangeRecommender, got {type(obj)}")
        return obj
