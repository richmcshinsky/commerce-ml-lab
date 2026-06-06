"""Shipping price optimiser.

Given a fitted ConversionElasticityModel, selects the shipping option that
maximises expected margin per session:

    E[margin | X, p] = P(convert | X, p) × (cart_margin + p - shipping_cost)

A conversion-rate floor can be specified to prevent the optimiser from
recommending a price that hurts conversion too severely.

Shipping options
----------------
Four tiers are available.  The optimiser chooses ONE per session — the option
that maximises expected margin — and returns it alongside a full breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .synthetic import PRODUCT_MARGIN_RATE, SHIPPING_COST_TO_MERCHANT

# ── Shipping option catalogue ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ShippingOption:
    """A single shipping tier available to show at checkout.

    Parameters
    ----------
    name:
        Display label (e.g. "Standard").
    price:
        Price charged to the customer (USD).
    transit_days:
        Estimated delivery window string (e.g. "3–5 days").
    label:
        Short machine-readable key.
    """

    name: str
    price: float
    transit_days: str
    label: str


SHIPPING_OPTIONS: list[ShippingOption] = [
    ShippingOption("Free Shipping", 0.00,  "5–7 days", "free"),
    ShippingOption("Standard",      4.99,  "3–5 days", "standard"),
    ShippingOption("Expedited",     7.99,  "2–3 days", "expedited"),
    ShippingOption("Express",       12.99, "1 day",    "express"),
]

FLAT_RATE_OPTION: ShippingOption = SHIPPING_OPTIONS[1]  # $4.99 Standard


@dataclass
class OptimizationResult:
    """Result of a single-session shipping price optimisation.

    Parameters
    ----------
    recommended:
        The chosen shipping option.
    expected_margin:
        Expected margin at the recommended price.
    p_convert:
        Predicted conversion probability at the recommended price.
    breakdown:
        Full comparison across all options.
    """

    recommended: ShippingOption
    expected_margin: float
    p_convert: float
    breakdown: pd.DataFrame


class ShippingPriceOptimizer:
    """Selects the shipping option that maximises expected margin per session.

    Parameters
    ----------
    elasticity_model:
        Fitted ``ConversionElasticityModel``.
    options:
        Shipping options to evaluate.  Defaults to ``SHIPPING_OPTIONS``.
    min_p_convert:
        Optional floor on predicted conversion probability.  Options below
        this threshold are excluded from consideration.
    """

    def __init__(
        self,
        elasticity_model: object,
        options: list[ShippingOption] | None = None,
        min_p_convert: float | None = None,
    ) -> None:
        self.elasticity_model = elasticity_model
        self.options = options or SHIPPING_OPTIONS
        self.min_p_convert = min_p_convert

    def recommend(self, session: pd.Series) -> OptimizationResult:
        """Return the optimal shipping option for a single checkout session.

        Parameters
        ----------
        session:
            One row from the sessions DataFrame.

        Returns
        -------
        OptimizationResult
            Recommended option with expected margin and full breakdown.
        """
        session_df = pd.DataFrame([session])
        rows: list[dict] = []
        for opt in self.options:
            p = float(self.elasticity_model.predict_at_price(session_df, opt.price)[0])
            em = p * (session["cart_value"] * PRODUCT_MARGIN_RATE + opt.price - SHIPPING_COST_TO_MERCHANT)
            rows.append(
                {
                    "option": opt.name,
                    "price": opt.price,
                    "transit_days": opt.transit_days,
                    "p_convert": round(p, 4),
                    "expected_margin": round(em, 2),
                }
            )

        breakdown = pd.DataFrame(rows)
        eligible = breakdown if self.min_p_convert is None else breakdown[breakdown["p_convert"] >= self.min_p_convert]
        if eligible.empty:
            eligible = breakdown  # relax constraint if all options fail the floor

        best_idx = eligible["expected_margin"].idxmax()
        best_row = breakdown.loc[best_idx]
        best_opt = next(o for o in self.options if o.name == best_row["option"])

        return OptimizationResult(
            recommended=best_opt,
            expected_margin=float(best_row["expected_margin"]),
            p_convert=float(best_row["p_convert"]),
            breakdown=breakdown,
        )

    def compare_policies(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Compare flat-rate vs. optimised policy across a session dataset.

        Parameters
        ----------
        sessions:
            Full sessions DataFrame (output of ``generate_shipping_dataset``).

        Returns
        -------
        pd.DataFrame
            One row per policy with columns:
            ``policy``, ``mean_expected_margin``, ``mean_p_convert``,
            ``margin_per_converted_session``.
        """
        results: list[dict] = []

        # ── Flat rate (always $4.99 Standard) ─────────────────────────────────
        p_flat = self.elasticity_model.predict_at_price(sessions, FLAT_RATE_OPTION.price)
        em_flat = p_flat * (sessions["cart_value"] * PRODUCT_MARGIN_RATE + FLAT_RATE_OPTION.price - SHIPPING_COST_TO_MERCHANT)
        results.append(
            {
                "policy": f"Flat rate (${FLAT_RATE_OPTION.price:.2f})",
                "mean_expected_margin": em_flat.mean(),
                "mean_p_convert": p_flat.mean(),
                "margin_per_converted": em_flat.mean() / max(p_flat.mean(), 1e-6),
            }
        )

        # ── Always free shipping ───────────────────────────────────────────────
        p_free = self.elasticity_model.predict_at_price(sessions, 0.00)
        em_free = p_free * (sessions["cart_value"] * PRODUCT_MARGIN_RATE - SHIPPING_COST_TO_MERCHANT)
        results.append(
            {
                "policy": "Always free shipping",
                "mean_expected_margin": em_free.mean(),
                "mean_p_convert": p_free.mean(),
                "margin_per_converted": em_free.mean() / max(p_free.mean(), 1e-6),
            }
        )

        # ── Always Express ($12.99) ────────────────────────────────────────────
        express = SHIPPING_OPTIONS[-1]
        p_exp = self.elasticity_model.predict_at_price(sessions, express.price)
        em_exp = p_exp * (sessions["cart_value"] * PRODUCT_MARGIN_RATE + express.price - SHIPPING_COST_TO_MERCHANT)
        results.append(
            {
                "policy": f"Always Express (${express.price:.2f})",
                "mean_expected_margin": em_exp.mean(),
                "mean_p_convert": p_exp.mean(),
                "margin_per_converted": em_exp.mean() / max(p_exp.mean(), 1e-6),
            }
        )

        # ── Optimised (vectorised) ─────────────────────────────────────────────
        all_ems: list[np.ndarray] = []
        all_ps: list[np.ndarray] = []
        for opt in self.options:
            p_o = self.elasticity_model.predict_at_price(sessions, opt.price)
            em_o = p_o * (sessions["cart_value"] * PRODUCT_MARGIN_RATE + opt.price - SHIPPING_COST_TO_MERCHANT)
            all_ems.append(em_o)
            all_ps.append(p_o)

        em_matrix = np.stack(all_ems, axis=1)   # (n, n_options)
        p_matrix = np.stack(all_ps, axis=1)
        best_idx = em_matrix.argmax(axis=1)
        em_opt = em_matrix[np.arange(len(sessions)), best_idx]
        p_opt = p_matrix[np.arange(len(sessions)), best_idx]

        results.append(
            {
                "policy": "Optimised (max expected margin)",
                "mean_expected_margin": em_opt.mean(),
                "mean_p_convert": p_opt.mean(),
                "margin_per_converted": em_opt.mean() / max(p_opt.mean(), 1e-6),
            }
        )

        return pd.DataFrame(results).round(4)

    def segment_price_distribution(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Return the distribution of recommended prices by segment.

        Parameters
        ----------
        sessions:
            Full sessions DataFrame including ``segment`` column.

        Returns
        -------
        pd.DataFrame
            Pivot table: segments × price options, values = fraction of sessions.
        """
        all_ems: list[np.ndarray] = []
        for opt in self.options:
            p_o = self.elasticity_model.predict_at_price(sessions, opt.price)
            em_o = p_o * (sessions["cart_value"] * PRODUCT_MARGIN_RATE + opt.price - SHIPPING_COST_TO_MERCHANT)
            all_ems.append(em_o)

        em_matrix = np.stack(all_ems, axis=1)
        best_opt_idx = em_matrix.argmax(axis=1)
        recommended_price = np.array([self.options[i].price for i in best_opt_idx])
        recommended_label = np.array([self.options[i].name for i in best_opt_idx])

        df = sessions[["segment"]].copy()
        df["recommended_price"] = recommended_price
        df["recommended_label"] = recommended_label

        pivot = (
            df.groupby(["segment", "recommended_label"])
            .size()
            .unstack(fill_value=0)
            .apply(lambda row: row / row.sum(), axis=1)
        )
        return pivot.round(3)
