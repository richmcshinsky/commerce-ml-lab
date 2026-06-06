"""Training script for the shipping price optimisation model.

Usage
-----
    python -m shipping.train         # full run, 60k sessions
    python -m shipping.train --quick  # 8k sessions, fast params (CI / smoke)

Outputs (in results/)
---------------------
    sessions.parquet
    elasticity_model.pkl
    shipping_metrics.csv
    shipping_elasticity_curves.png
    shipping_expected_margin_curves.png
    shipping_policy_comparison.png
    shipping_price_distribution.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
# _HERE = src/shipping/  →  parents[0] = src/  →  parents[1] = project root
# _HERE.parents[3] = repo root  →  repo root / src  = shared commerce_ml library
sys.path.insert(0, str(_HERE.parents[3] / "src"))  # commerce_ml shared library
sys.path.insert(0, str(_HERE.parent))              # project src/ — makes 'shipping' importable

RESULTS = _HERE.parent.parent / "results"


def main(quick: bool = False) -> None:  # noqa: C901
    """Train and evaluate the shipping price optimisation system."""
    from shipping.elasticity import DEFAULT_LGBM_PARAMS, ConversionElasticityModel
    from shipping.optimizer import FLAT_RATE_OPTION, SHIPPING_OPTIONS, ShippingPriceOptimizer
    from shipping.synthetic import (
        PRODUCT_MARGIN_RATE,
        SEGMENT_NAMES,
        SHIPPING_COST_TO_MERCHANT,
        generate_shipping_dataset,
    )

    RESULTS.mkdir(exist_ok=True)

    n_sessions = 8_000 if quick else 60_000
    lgbm_params = {**DEFAULT_LGBM_PARAMS, "n_estimators": 50 if quick else 400}

    # ── Generate data ──────────────────────────────────────────────────────────
    logger.info("Generating %d checkout sessions…", n_sessions)
    df = generate_shipping_dataset(n_sessions=n_sessions, random_state=42)
    logger.info(
        "  Overall conversion rate: %.1f%%  |  Segments: %s",
        df["converted"].mean() * 100,
        {s: f"{(df['segment'] == s).mean():.0%}" for s in SEGMENT_NAMES},
    )
    df.to_parquet(RESULTS / "sessions.parquet", index=False)

    # ── Train / test split ─────────────────────────────────────────────────────
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

    # ── Train elasticity model ─────────────────────────────────────────────────
    logger.info("Training ConversionElasticityModel…")
    model = ConversionElasticityModel(lgbm_params=lgbm_params)
    model.fit(train_df)

    # Evaluate on test set
    p_pred = model.predict_proba(test_df)
    y_true = test_df["converted"].astype(int).values
    pr_auc = average_precision_score(y_true, p_pred)
    model.pr_auc_ = pr_auc
    logger.info("  Elasticity model PR-AUC: %.4f  (random baseline = %.4f)", pr_auc, y_true.mean())
    model.save(RESULTS / "elasticity_model.pkl")

    # ── Policy comparison ──────────────────────────────────────────────────────
    logger.info("Comparing pricing policies on test set…")
    optimizer = ShippingPriceOptimizer(model)
    policy_df = optimizer.compare_policies(test_df)
    logger.info("\n%s", policy_df.to_string(index=False))

    # ── Segment price distribution ─────────────────────────────────────────────
    seg_price_df = optimizer.segment_price_distribution(test_df)

    # ── Compute improvement metrics ────────────────────────────────────────────
    flat_em = float(policy_df[policy_df["policy"].str.startswith("Flat")]["mean_expected_margin"].iloc[0])
    opt_em = float(policy_df[policy_df["policy"].str.startswith("Opt")]["mean_expected_margin"].iloc[0])
    improvement_pct = (opt_em - flat_em) / abs(flat_em) * 100

    flat_p = float(policy_df[policy_df["policy"].str.startswith("Flat")]["mean_p_convert"].iloc[0])
    opt_p = float(policy_df[policy_df["policy"].str.startswith("Opt")]["mean_p_convert"].iloc[0])

    logger.info(
        "Optimised vs flat-rate: +%.1f%% expected margin | conversion %+.1f pp",
        improvement_pct,
        (opt_p - flat_p) * 100,
    )

    # ── Save metrics ───────────────────────────────────────────────────────────
    metrics_rows = [
        {"metric": "elasticity_pr_auc", "value": round(pr_auc, 4)},
        {"metric": "random_baseline_pr_auc", "value": round(y_true.mean(), 4)},
        {"metric": "margin_improvement_pct", "value": round(improvement_pct, 2)},
        {"metric": "flat_rate_mean_conversion", "value": round(flat_p, 4)},
        {"metric": "optimised_mean_conversion", "value": round(opt_p, 4)},
        {"metric": "flat_rate_mean_em", "value": round(flat_em, 4)},
        {"metric": "optimised_mean_em", "value": round(opt_em, 4)},
        {"metric": "n_train", "value": len(train_df)},
        {"metric": "n_test", "value": len(test_df)},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS / "shipping_metrics.csv", index=False)

    # ── Charts ─────────────────────────────────────────────────────────────────
    prices = np.linspace(0, 13, 60)
    seg_colors = {
        "sure_thing": "#1565C0",
        "persuadable": "#2E7D32",
        "lost_cause": "#757575",
        "sleeping_dog": "#E65100",
    }

    # Chart 1: Price elasticity curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for seg in SEGMENT_NAMES:
        seg_sessions = test_df[test_df["segment"] == seg].head(200)
        if seg_sessions.empty:
            continue
        p_curves = [model.predict_at_price(seg_sessions, p).mean() for p in prices]
        ax.plot(prices, p_curves, label=seg.replace("_", " ").title(),
                color=seg_colors[seg], linewidth=2.2)

    ax.set_xlabel("Shipping price ($)")
    ax.set_ylabel("Mean P(convert)")
    ax.set_title("Price elasticity by customer segment")
    ax.legend()
    ax.axvline(x=FLAT_RATE_OPTION.price, color="grey", linestyle="--", alpha=0.5,
               label=f"Flat rate (${FLAT_RATE_OPTION.price:.2f})")
    ax.grid(alpha=0.3)

    # Chart 2: Expected margin curves
    ax2 = axes[1]
    for seg in SEGMENT_NAMES:
        seg_sessions = test_df[test_df["segment"] == seg].head(200)
        if seg_sessions.empty:
            continue
        em_curves = [
            (model.predict_at_price(seg_sessions, p) *
             (seg_sessions["cart_value"] * PRODUCT_MARGIN_RATE + p - SHIPPING_COST_TO_MERCHANT)).mean()
            for p in prices
        ]
        ax2.plot(prices, em_curves, label=seg.replace("_", " ").title(),
                 color=seg_colors[seg], linewidth=2.2)

    ax2.set_xlabel("Shipping price ($)")
    ax2.set_ylabel("Mean expected margin ($)")
    ax2.set_title("Expected margin by segment — optimal price varies")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "shipping_elasticity_curves.png", dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("Saved shipping_elasticity_curves.png")

    # Chart 3: Policy comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    policies = policy_df["policy"].tolist()
    x = np.arange(len(policies))
    bar_colors = ["#90A4AE", "#90A4AE", "#90A4AE", "#1565C0"]

    axes[0].bar(x, policy_df["mean_expected_margin"], color=bar_colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policies, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean expected margin per session ($)")
    axes[0].set_title("Expected margin by pricing policy")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(policy_df["mean_expected_margin"]):
        axes[0].text(i, v + 0.05, f"${v:.2f}", ha="center", fontsize=9)

    axes[1].bar(x, policy_df["mean_p_convert"] * 100, color=bar_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policies, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("Mean conversion rate (%)")
    axes[1].set_title("Conversion rate by pricing policy")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(policy_df["mean_p_convert"]):
        axes[1].text(i, v * 100 + 0.3, f"{v:.1%}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS / "shipping_policy_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("Saved shipping_policy_comparison.png")

    # Chart 4: Price distribution by segment
    if not seg_price_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        option_names = [o.name for o in SHIPPING_OPTIONS]
        option_colors = ["#2E7D32", "#1565C0", "#E65100", "#7B1FA2"]
        bottom = np.zeros(len(seg_price_df))

        for opt_name, col in zip(option_names, option_colors, strict=False):
            if opt_name not in seg_price_df.columns:
                continue
            vals = seg_price_df[opt_name].values
            ax.bar(seg_price_df.index, vals, bottom=bottom, label=opt_name, color=col, alpha=0.85)
            bottom += vals

        ax.set_ylabel("Fraction of sessions")
        ax.set_title("Recommended shipping option by segment (optimised policy)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS / "shipping_price_distribution.png", dpi=130, bbox_inches="tight")
        plt.close()
        logger.info("Saved shipping_price_distribution.png")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Shipping Price Optimisation — Training Complete")
    print("=" * 60)
    print(f"  Sessions:              {n_sessions:,}  (train={len(train_df):,} / test={len(test_df):,})")
    print(f"  Elasticity PR-AUC:     {pr_auc:.4f}  (random = {y_true.mean():.4f})")
    print(f"  Margin improvement:    +{improvement_pct:.1f}%  vs flat-rate ${FLAT_RATE_OPTION.price:.2f}")
    print(f"  Conversion delta:      {(opt_p - flat_p)*100:+.1f} pp")
    print(f"  Results saved to:      {RESULTS}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train shipping price optimisation model")
    parser.add_argument("--quick", action="store_true", help="Fast run with 8k sessions")
    args = parser.parse_args()
    main(quick=args.quick)
