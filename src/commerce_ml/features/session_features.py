"""Session-level feature engineering for the checkout intent model.

Transforms raw clickstream events (page views, add-to-cart actions, etc.)
into a flat feature vector suitable for the intent classifier.

Event types recognised
----------------------
- ``page_view`` — user visited a product or category page
- ``add_to_cart`` — item added to cart
- ``remove_from_cart`` — item removed from cart
- ``size_change`` — user changed size/variant on a PDP
- ``checkout_start`` — user reached checkout
- ``purchase`` — session converted (target label)
"""

from __future__ import annotations

import pandas as pd


def build_session_features(
    events: pd.DataFrame,
    session_col: str = "session_id",
    event_col: str = "event_type",
    ts_col: str = "timestamp",
    price_col: str | None = "price",
) -> pd.DataFrame:
    """Aggregate raw session events into a flat feature DataFrame.

    Parameters
    ----------
    events:
        Raw event log. One row per event.
        Required columns: ``session_id``, ``event_type``, ``timestamp``.
        Optional: ``price`` (item price at time of event).
    session_col:
        Name of the session identifier column.
    event_col:
        Name of the event type column.
    ts_col:
        Name of the timestamp column.
    price_col:
        Name of the item price column, or ``None`` if not present.

    Returns
    -------
    pd.DataFrame
        One row per session with columns:

        - ``n_page_views`` — total page views in session
        - ``n_add_to_cart`` — total add-to-cart events
        - ``n_remove_from_cart`` — total remove-from-cart events
        - ``n_size_changes`` — total size/variant changes (indecision signal)
        - ``cart_size`` — items currently in cart (adds minus removes)
        - ``session_duration_seconds`` — time from first to last event
        - ``price_max`` — max item price viewed (if price_col provided)
        - ``price_mean`` — mean item price viewed (if price_col provided)
        - ``has_checkout_start`` — 1 if session reached checkout, else 0
        - ``converted`` — 1 if session ended in purchase, else 0 (label)
    """
    events = events.copy()
    events[ts_col] = pd.to_datetime(events[ts_col])

    grp = events.groupby(session_col)

    counts = (
        events[
            events[event_col].isin(
                [
                    "page_view",
                    "add_to_cart",
                    "remove_from_cart",
                    "size_change",
                    "checkout_start",
                    "purchase",
                ]
            )
        ]
        .groupby([session_col, event_col])
        .size()
        .unstack(fill_value=0)
        .rename(
            columns={
                "page_view": "n_page_views",
                "add_to_cart": "n_add_to_cart",
                "remove_from_cart": "n_remove_from_cart",
                "size_change": "n_size_changes",
                "checkout_start": "has_checkout_start",
                "purchase": "converted",
            }
        )
        .reindex(
            columns=[
                "n_page_views",
                "n_add_to_cart",
                "n_remove_from_cart",
                "n_size_changes",
                "has_checkout_start",
                "converted",
            ],
            fill_value=0,
        )
    )

    # Cart size = net adds
    counts["cart_size"] = (counts["n_add_to_cart"] - counts["n_remove_from_cart"]).clip(lower=0)

    # Session duration
    duration = (
        (grp[ts_col].max() - grp[ts_col].min())
        .dt.total_seconds()
        .rename("session_duration_seconds")
    )
    counts = counts.join(duration)

    # Price features (optional)
    if price_col is not None and price_col in events.columns:
        price_stats = grp[price_col].agg(price_max="max", price_mean="mean")
        counts = counts.join(price_stats)

    # Binarise checkout_start flag
    counts["has_checkout_start"] = (counts["has_checkout_start"] > 0).astype(int)

    return counts.reset_index()
