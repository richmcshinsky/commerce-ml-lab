"""Evaluation suite for propensity and uplift models.

Metrics
-------
- **Qini coefficient**: area between the Qini curve and the random baseline.
  The primary evaluation metric for uplift models.
- **AUUC**: area under the uplift curve (equivalent to Qini but normalised).
- **Uplift@K**: incremental conversion rate when targeting the top-K% by score.
- **Incremental revenue**: cost-adjusted comparison of policies at a fixed budget.
- **Qini curve**: full curve data for plotting.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# qini_coefficient implemented locally below — avoids sklearn import at module level


__all__ = [
    "qini_coefficient",
    "qini_curve",
    "auuc",
    "uplift_at_k",
    "incremental_conversions_at_k",
    "policy_comparison",
]

def qini_coefficient(
    y_true: "np.ndarray | pd.Series",
    treatment: "np.ndarray | pd.Series",
    uplift_score: "np.ndarray | pd.Series",
) -> float:
    """Qini coefficient (area between Qini curve and random baseline)."""
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "w": np.asarray(treatment, dtype=float),
        "score": np.asarray(uplift_score, dtype=float),
    }).sort_values("score", ascending=False).reset_index(drop=True)

    n = len(df)
    n_treated = df["w"].sum()
    n_control = n - n_treated
    frac = np.arange(1, n + 1) / n

    incremental = (
        (df["y"] * df["w"]).cumsum()
        - (df["y"] * (1 - df["w"])).cumsum() * (n_treated / max(n_control, 1))
    )
    random_line = incremental.iloc[-1] * frac
    return float(np.trapz(incremental.values, frac) - np.trapz(random_line, frac))




def qini_curve(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_score: np.ndarray | pd.Series,
    n_bins: int = 100,
) -> pd.DataFrame:
    """Compute the Qini curve data points.

    Sorts individuals by ``uplift_score`` descending and computes the
    cumulative incremental conversions at each targeting fraction.

    Parameters
    ----------
    y_true:
        Binary outcome (1 = converted).
    treatment:
        Binary treatment assignment (1 = treated).
    uplift_score:
        Predicted uplift score (higher = target first).
    n_bins:
        Number of evenly-spaced quantiles along the curve.

    Returns
    -------
    pd.DataFrame
        Columns: fraction_targeted, incremental_conversions, random_baseline.
    """
    y = np.asarray(y_true, dtype=float)
    w = np.asarray(treatment, dtype=float)
    s = np.asarray(uplift_score, dtype=float)

    order = np.argsort(s)[::-1]
    y_s, w_s = y[order], w[order]

    n = len(y_s)
    n_treated = w_s.sum()
    n_control = n - n_treated

    cum_trt = (y_s * w_s).cumsum()
    cum_ctl = (y_s * (1 - w_s)).cumsum()

    scale = n_treated / max(n_control, 1)
    incremental = cum_trt - cum_ctl * scale

    fractions = np.linspace(0, 1, n_bins + 1)
    idx = np.minimum((fractions * n).astype(int), n - 1)

    return pd.DataFrame({
        "fraction_targeted": fractions,
        "incremental_conversions": np.concatenate([[0], incremental[idx[1:]].values
                                                   if hasattr(incremental, "values")
                                                   else incremental[idx[1:]]]),
        "random_baseline": fractions * float(incremental.iloc[-1]
                                             if hasattr(incremental, "iloc")
                                             else incremental[-1]),
    })


def auuc(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_score: np.ndarray | pd.Series,
) -> float:
    """Area under the uplift curve (normalised Qini).

    Returns
    -------
    float
        AUUC value. Positive means better than random targeting.
    """
    curve = qini_curve(y_true, treatment, uplift_score)
    area_model  = float(np.trapz(curve["incremental_conversions"], curve["fraction_targeted"]))
    area_random = float(np.trapz(curve["random_baseline"],          curve["fraction_targeted"]))
    return area_model - area_random


def uplift_at_k(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_score: np.ndarray | pd.Series,
    k: float = 0.20,
) -> float:
    """Incremental conversion rate when targeting the top-k fraction.

    Parameters
    ----------
    k:
        Fraction of population to target (e.g. 0.20 = top 20%).

    Returns
    -------
    float
        Incremental conversions per person targeted (positive = lift).
    """
    y = np.asarray(y_true, dtype=float)
    w = np.asarray(treatment, dtype=float)
    s = np.asarray(uplift_score, dtype=float)

    n_target = max(1, int(len(s) * k))
    top_idx  = np.argsort(s)[::-1][:n_target]
    rest_idx = np.argsort(s)[::-1][n_target:]

    def _ate(idx: np.ndarray) -> float:
        w_sub = w[idx]
        y_sub = y[idx]
        n_t = w_sub.sum()
        n_c = len(w_sub) - n_t
        if n_t == 0 or n_c == 0:
            return 0.0
        return float(y_sub[w_sub == 1].mean() - y_sub[w_sub == 0].mean())

    return _ate(top_idx)


def incremental_conversions_at_k(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    uplift_score: np.ndarray | pd.Series,
    k: float = 0.20,
) -> float:
    """Total incremental conversions when targeting the top-k fraction.

    Returns
    -------
    float
        Estimated additional conversions from the intervention.
    """
    n_target = max(1, int(len(uplift_score) * k))
    return uplift_at_k(y_true, treatment, uplift_score, k) * n_target


def policy_comparison(
    y_true: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    scores: dict[str, np.ndarray],
    k_values: list[float] | None = None,
) -> pd.DataFrame:
    """Compare multiple targeting policies at different budget levels.

    Parameters
    ----------
    y_true:
        Binary outcome.
    treatment:
        Binary treatment assignment.
    scores:
        Dict of {model_name: uplift_score_array}.
    k_values:
        Targeting fractions to evaluate (default: 0.05 to 0.50).

    Returns
    -------
    pd.DataFrame
        Columns: model, k, uplift_at_k, incremental_conversions.
    """
    if k_values is None:
        k_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    rows = []
    for name, score in scores.items():
        for k in k_values:
            uat_k = uplift_at_k(y_true, treatment, score, k)
            inc_c = incremental_conversions_at_k(y_true, treatment, score, k)
            rows.append({
                "model": name,
                "k": k,
                "uplift_at_k": round(uat_k, 6),
                "incremental_conversions": round(inc_c, 1),
            })
    return pd.DataFrame(rows)
