"""Shared plotting utilities for the commerce-ml-lab portfolio.

All plots use a consistent style: clean white background, a muted colour
palette, and a standard figure size. Import ``apply_style()`` once at the
top of any notebook to apply the style globally.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Global style ─────────────────────────────────────────────────────────────

PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]
FIGURE_SIZE = (12, 5)
FIGURE_SIZE_SQUARE = (8, 8)


def apply_style() -> None:
    """Apply the shared matplotlib style.

    Call once at the top of a notebook::

        from commerce_ml.viz.plotting import apply_style
        apply_style()
    """
    plt.rcParams.update({
        "figure.figsize": FIGURE_SIZE,
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler("color", PALETTE),
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titlepad": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "font.family": "sans-serif",
    })


# ── Forecasting plots ─────────────────────────────────────────────────────────


def plot_forecast(
    dates: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    forecasts: dict[str, np.ndarray],
    title: str = "Demand Forecast",
    ax: Any | None = None,
) -> Any:
    """Plot actual vs. one or more model forecasts on a shared time axis.

    Parameters
    ----------
    dates:
        Date values for the x-axis.
    actual:
        Actual sales values.
    forecasts:
        Dict mapping model name -> forecast array. Each array must be the
        same length as ``actual``.
    title:
        Plot title.
    ax:
        Optional existing matplotlib Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(dates, actual, label="Actual", color="black", linewidth=1.5, alpha=0.8)

    for i, (name, values) in enumerate(forecasts.items()):
        ax.plot(dates, values, label=name, color=PALETTE[i % len(PALETTE)],
                linewidth=1.5, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    return ax


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric_col: str = "wmape",
    model_col: str = "model",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Horizontal bar chart comparing model metrics.

    Parameters
    ----------
    metrics_df:
        DataFrame with one row per model. Must contain ``model_col`` and
        ``metric_col``.
    metric_col:
        Name of the metric column to plot.
    model_col:
        Name of the model label column.
    title:
        Plot title. Defaults to the metric name.
    ax:
        Optional existing Axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(3, len(metrics_df) * 0.7)))

    df = metrics_df.sort_values(metric_col, ascending=True)
    colors = [PALETTE[0] if i == 0 else "#94A3B8" for i in range(len(df))]
    ax.barh(df[model_col], df[metric_col], color=colors)
    ax.set_xlabel(metric_col.upper())
    ax.set_title(title or f"{metric_col.upper()} by model")
    ax.grid(axis="x")
    plt.tight_layout()
    return ax


# ── Classification / Fraud plots ──────────────────────────────────────────────


def plot_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    label: str = "Model",
    ax: Any | None = None,
) -> Any:
    """Plot a precision-recall curve.

    Parameters
    ----------
    recall:
        Recall values (from ``sklearn.metrics.precision_recall_curve``).
    precision:
        Precision values.
    label:
        Legend label for this curve.
    ax:
        Optional existing Axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)

    ax.plot(recall, precision, label=label, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid()
    plt.tight_layout()
    return ax


# ── Uplift plots ──────────────────────────────────────────────────────────────


def plot_qini_curve(
    fraction_targeted: np.ndarray,
    incremental_conversions: np.ndarray,
    random_baseline: np.ndarray,
    label: str = "Model",
    ax: Any | None = None,
) -> Any:
    """Plot a Qini curve against the random-targeting baseline.

    Parameters
    ----------
    fraction_targeted:
        Fraction of population targeted (x-axis), from 0 to 1.
    incremental_conversions:
        Cumulative incremental conversions from targeting by uplift score.
    random_baseline:
        Cumulative incremental conversions from random targeting.
    label:
        Model label.
    ax:
        Optional existing Axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)

    ax.plot(fraction_targeted, incremental_conversions, label=label,
            color=PALETTE[0], linewidth=2)
    ax.plot(fraction_targeted, random_baseline, label="Random targeting",
            color="#94A3B8", linewidth=1.5, linestyle="--")
    ax.fill_between(fraction_targeted, random_baseline, incremental_conversions,
                    alpha=0.1, color=PALETTE[0])
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Qini Curve")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    return ax
