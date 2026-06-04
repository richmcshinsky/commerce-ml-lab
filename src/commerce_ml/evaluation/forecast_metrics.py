"""Forecasting evaluation metrics.

Implements WMAPE, MASE, RMSSE, and pinball loss — the standard metrics for
retail demand forecasting. These are intentional choices over plain MAPE:

- **WMAPE** avoids division-by-zero on zero-sales days (common in retail) and
  weights errors proportionally to actual volume.
- **MASE** is scale-free and comparable across series with different volumes.
- **RMSSE** is the official M5 competition metric.
- **Pinball loss** evaluates quantile forecasts; essential for inventory
  optimisation where over- and under-forecasting have asymmetric costs.

References
----------
- Hyndman & Koehler (2006), "Another look at measures of forecast accuracy."
- M5 Accuracy competition evaluation metric documentation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wmape(actual: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Weighted Mean Absolute Percentage Error.

    Unlike MAPE, WMAPE weights each error by the actual volume, which:
    (a) avoids division by zero on zero-sale periods, and
    (b) gives larger errors on high-volume periods proportionally more weight.

    .. math::

        \\text{WMAPE} = \\frac{\\sum |a_i - f_i|}{\\sum |a_i|}

    Parameters
    ----------
    actual:
        Array of actual values.
    forecast:
        Array of forecast values. Same length as ``actual``.

    Returns
    -------
    float
        WMAPE in [0, inf). 0.0 is perfect.

    Raises
    ------
    ValueError
        If ``actual`` and ``forecast`` have different lengths.
    ValueError
        If ``sum(|actual|)`` is zero (all actuals are zero).

    Examples
    --------
    >>> wmape(np.array([10, 20, 30]), np.array([12, 18, 33]))
    0.1
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError(
            f"actual and forecast must have the same length, got {len(actual)} and {len(forecast)}"
        )

    denom = np.sum(np.abs(actual))
    if denom == 0.0:
        raise ValueError("sum(|actual|) is zero — WMAPE is undefined when all actuals are zero")

    return float(np.sum(np.abs(actual - forecast)) / denom)


def mase(
    actual: np.ndarray | pd.Series,
    forecast: np.ndarray | pd.Series,
    naive_forecast: np.ndarray | pd.Series,
) -> float:
    """Mean Absolute Scaled Error.

    Scales the MAE of the forecast by the MAE of a naive (seasonal) baseline,
    making the metric comparable across series of different scales.

    .. math::

        \\text{MASE} = \\frac{\\text{MAE}(\\text{forecast})}{\\text{MAE}(\\text{naive})}

    Parameters
    ----------
    actual:
        Array of actual values (test period).
    forecast:
        Array of forecast values (test period).
    naive_forecast:
        Array of naive baseline forecast values for the same test period.
        Typically the seasonal naive: :math:`\\hat{y}_t = y_{t-m}` where
        :math:`m` is the seasonal period (e.g. 7 for daily data).

    Returns
    -------
    float
        MASE. Values < 1.0 mean the model beats the naive baseline.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    naive = np.asarray(naive_forecast, dtype=float)

    mae_model = np.mean(np.abs(actual - forecast))
    mae_naive = np.mean(np.abs(actual - naive))

    if mae_naive == 0.0:
        return 0.0 if mae_model == 0.0 else float("inf")

    return float(mae_model / mae_naive)


def rmsse(
    actual: np.ndarray | pd.Series,
    forecast: np.ndarray | pd.Series,
    train_actual: np.ndarray | pd.Series,
    seasonality: int = 1,
) -> float:
    """Root Mean Squared Scaled Error (M5 official metric).

    .. math::

        \\text{RMSSE} = \\sqrt{
            \\frac{\\frac{1}{h} \\sum_{t=n+1}^{n+h}(y_t - \\hat{y}_t)^2}
            {\\frac{1}{n-m} \\sum_{t=m+1}^{n}(y_t - y_{t-m})^2}
        }

    Parameters
    ----------
    actual:
        Test-period actual values (length h).
    forecast:
        Test-period forecast values (length h).
    train_actual:
        Training-period actual values (length n).
    seasonality:
        Seasonal period m (default 1 for non-seasonal; use 7 for daily retail).

    Returns
    -------
    float
        RMSSE. Values < 1.0 beat the seasonal naive baseline.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    train = np.asarray(train_actual, dtype=float)

    n = len(train)
    m = seasonality
    # Guard: if the training series is shorter than the seasonal period,
    # fall back to a lag-1 naive denominator rather than raising a broadcast error.
    if n <= m:
        m = max(1, n - 1)

    mse_forecast = np.mean((actual - forecast) ** 2)
    mse_naive_train = np.mean((train[m:] - train[: n - m]) ** 2)

    if mse_naive_train == 0.0:
        return 0.0 if mse_forecast == 0.0 else float("inf")

    return float(np.sqrt(mse_forecast / mse_naive_train))


def pinball_loss(
    actual: np.ndarray | pd.Series,
    quantile_forecast: np.ndarray | pd.Series,
    quantile: float,
) -> float:
    """Pinball (quantile) loss.

    Evaluates a single quantile forecast. Used to assess prediction intervals
    from quantile regression models (e.g. LightGBM with ``objective="quantile"``).

    .. math::

        L_\\tau(y, \\hat{q}) =
        \\begin{cases}
            \\tau \\cdot (y - \\hat{q}) & \\text{if } y \\geq \\hat{q} \\\\
            (1 - \\tau) \\cdot (\\hat{q} - y) & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    actual:
        Array of actual values.
    quantile_forecast:
        Array of predicted quantile values.
    quantile:
        Target quantile in (0, 1). E.g. 0.9 for the 90th percentile forecast.

    Returns
    -------
    float
        Mean pinball loss across all observations.

    Examples
    --------
    >>> pinball_loss(np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 2.5]), 0.9)
    0.2
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    actual = np.asarray(actual, dtype=float)
    qf = np.asarray(quantile_forecast, dtype=float)

    errors = actual - qf
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1.0) * errors)
    return float(np.mean(loss))


def summarise_forecast_metrics(
    actual: np.ndarray | pd.Series,
    forecast: np.ndarray | pd.Series,
    train_actual: np.ndarray | pd.Series,
    naive_forecast: np.ndarray | pd.Series,
    label: str = "model",
) -> pd.DataFrame:
    """Compute all standard forecast metrics and return a summary DataFrame.

    Parameters
    ----------
    actual:
        Test-period actuals.
    forecast:
        Test-period point forecasts.
    train_actual:
        Training-period actuals (used for RMSSE denominator).
    naive_forecast:
        Seasonal naive forecast for the test period (used for MASE).
    label:
        Name for this model in the output table.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns: ``model``, ``wmape``,
        ``mase``, ``rmsse``.
    """
    return pd.DataFrame(
        [
            {
                "model": label,
                "wmape": wmape(actual, forecast),
                "mase": mase(actual, forecast, naive_forecast),
                "rmsse": rmsse(actual, forecast, train_actual, seasonality=7),
            }
        ]
    )
