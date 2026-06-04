"""Shared helpers for the Commerce ML Lab Streamlit portfolio."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

# numpy 2.0 removed np.trapz → use np.trapezoid
try:
    _trapezoid = np.trapezoid  # type: ignore[attr-defined]  # numpy >= 2.0
except AttributeError:
    _trapezoid = np.trapz  # type: ignore[attr-defined]      # numpy < 2.0


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_inventory_policies() -> pd.DataFrame:
    p = ROOT / "projects/01_demand_forecasting/results/inventory_policies.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_forecast_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for f in [
        ROOT / "projects/01_demand_forecasting/results/baselines_metrics.csv",
        ROOT / "projects/01_demand_forecasting/results/lgbm_metrics.csv",
    ]:
        if f.exists():
            rows.append(pd.read_csv(f))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.cache_data
def load_returns_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = ROOT / "projects/02_returns_intelligence/results"

    def _load(name: str) -> pd.DataFrame:
        p = base / name
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    return _load("customers.parquet"), _load("orders.parquet"), _load("returns.parquet")


@st.cache_data
def load_uplift_data(max_rows: int = 80_000) -> pd.DataFrame:
    p = ROOT / "projects/03_checkout_intent/results/uplift_data.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    return df


# ── Qini curve math ────────────────────────────────────────────────────────────

def qini_curve(
    y: np.ndarray,
    w: np.ndarray,
    score: np.ndarray,
    n_bins: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (fraction_targeted, incremental_conversions) for a Qini curve."""
    order = np.argsort(score)[::-1]
    y_s, w_s = y[order], w[order]
    n = len(y_s)
    bins = np.unique(np.linspace(0, n, n_bins + 1, dtype=int))
    fracs: list[float] = [0.0]
    incr: list[float] = [0.0]
    for k in bins[1:]:
        top_y, top_w = y_s[:k], w_s[:k]
        nt = top_w.sum()
        nc = k - nt
        treated_conv = (top_y * top_w).sum()
        control_conv = (top_y * (1 - top_w)).sum()
        inc = treated_conv - control_conv * (nt / max(nc, 1))
        fracs.append(k / n)
        incr.append(float(inc))
    return np.array(fracs), np.array(incr)


def qini_coeff(fracs: np.ndarray, incr: np.ndarray) -> float:
    """Area between the Qini curve and the random-targeting diagonal."""
    random_line = incr[-1] * fracs
    return float(_trapezoid(incr - random_line, fracs))


# ── Returns heuristics ─────────────────────────────────────────────────────────

EXCHANGE_RULES: dict[str, str] = {
    "too_small": "Next size up — same style, same color.",
    "too_large": "Next size down — same style, same color.",
    "wrong_color": "Any available color variant in this style.",
    "defective": "Exact replacement with expedited shipping.",
    "not_as_described": "Full refund or closest matching item — review listing accuracy.",
    "changed_mind": "Store credit or similar item in same category.",
}


def score_return(
    category: str,
    condition: str,
    reason_code: str,
    days_to_return: int,
    item_price: float,
) -> tuple[float, str, str]:
    """
    Heuristic two-stage fraud risk scorer for a single return.

    Returns
    -------
    risk_score : float  — 0–1 fraud probability estimate
    risk_label : str    — "Low", "Medium", or "High"
    exchange_rec : str  — recommended exchange action
    """
    score = 0.0

    # Condition signal
    if condition == "damaged":
        score += 0.30
    elif condition == "used":
        score += 0.15

    # Reason + condition interaction
    if reason_code == "defective" and condition == "damaged":
        score += 0.25  # velocity pattern: abuse then claim defect
    elif reason_code == "changed_mind" and days_to_return > 25:
        score += 0.20  # wardrober: use, then return late
    elif reason_code == "not_as_described":
        score += 0.10  # ambiguous signal

    # Timing signal
    if days_to_return <= 3 and reason_code == "defective":
        score += 0.15  # suspiciously fast "defect" claim
    elif days_to_return >= 25 and condition in ("used", "damaged"):
        score += 0.15  # late return in poor condition

    # High-value category signal
    if category == "electronics" and item_price > 150:
        score += 0.15
    elif category in ("clothing", "shoes") and days_to_return > 20 and condition == "used":
        score += 0.10  # wardrober

    score = min(score, 1.0)
    label = "Low" if score < 0.25 else "Medium" if score < 0.55 else "High"
    rec = EXCHANGE_RULES.get(reason_code, "Manual review — no matching rule.")
    return score, label, rec


