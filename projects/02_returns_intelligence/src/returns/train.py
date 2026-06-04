"""Training script for the returns intelligence suite.

Usage:
    python -m returns.train            # train all three models
    python -m returns.train --quick    # 5k customers, fast params (CI / smoke)

Outputs:
    results/customers.parquet
    results/orders.parquet
    results/returns.parquet
    results/likelihood_model.pkl
    results/fraud_model.pkl
    results/exchange_model.pkl
    results/returns_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parents[3] / "src"))  # commerce_ml shared library


def main(quick: bool = False) -> None:  # noqa: C901
    from commerce_ml.data.synthetic import generate_returns_dataset
    from commerce_ml.evaluation.classification_metrics import pr_auc, precision_at_k
    from returns.exchange import ExchangeRecommender
    from returns.fraud import FraudDetectionModel
    from returns.likelihood import ReturnLikelihoodModel

    RESULTS = _HERE.parent.parent / "results"
    RESULTS.mkdir(exist_ok=True)

    n_customers = 2_000 if quick else 20_000
    fast_params = {"n_estimators": 20 if quick else 300, "verbose": -1, "random_state": 42}

    # ── Generate / load data ──────────────────────────────────────────────────
    logger.info("Generating synthetic dataset (%d customers)…", n_customers)
    customers, orders, returns = generate_returns_dataset(
        n_customers=n_customers, random_state=42
    )
    logger.info(
        "  %d customers | %d orders | %d returns (fraud=%.2f%%)",
        len(customers), len(orders), len(returns), returns["is_fraud"].mean() * 100,
    )

    customers.to_parquet(RESULTS / "customers.parquet", index=False)
    orders.to_parquet(RESULTS / "orders.parquet", index=False)
    returns.to_parquet(RESULTS / "returns.parquet", index=False)

    # Temporal splits
    orders_sorted  = orders.sort_values("order_date")
    returns_sorted = returns.sort_values("return_date")
    n_tr_o  = int(len(orders_sorted)  * 0.8)
    n_tr_r  = int(len(returns_sorted) * 0.8)
    train_orders  = orders_sorted.iloc[:n_tr_o]
    test_orders   = orders_sorted.iloc[n_tr_o:]
    train_returns = returns_sorted.iloc[:n_tr_r]
    test_returns  = returns_sorted.iloc[n_tr_r:]

    # ── Model A: Return likelihood ────────────────────────────────────────────
    logger.info("Training ReturnLikelihoodModel…")
    lm = ReturnLikelihoodModel(lgbm_params=fast_params)
    lm.fit(train_orders, customers)

    proba = lm.predict_proba(test_orders, customers)
    y_true_rr = test_orders["was_returned"].astype(int).values
    lm_prauc = pr_auc(y_true_rr, proba)
    logger.info("  Likelihood PR-AUC: %.4f  (random=%.4f)", lm_prauc, y_true_rr.mean())
    lm.save(RESULTS / "likelihood_model.pkl")

    # ── Model B: Fraud detection ──────────────────────────────────────────────
    logger.info("Training FraudDetectionModel…")
    fraud_params = {**fast_params, "scale_pos_weight": 10}
    fm = FraudDetectionModel(lgbm_params=fraud_params)
    fm.fit(train_returns, orders, customers, auto_threshold=True)

    preds_fraud = fm.predict(test_returns, orders, customers)
    y_true_fr = test_returns["is_fraud"].astype(int).values
    fm_prauc  = pr_auc(y_true_fr, preds_fraud["fraud_probability"].values)
    p_at_50   = precision_at_k(y_true_fr, preds_fraud["fraud_probability"].values, k=50)
    logger.info(
        "  Fraud PR-AUC: %.4f  Precision@50: %.1f%%  threshold=%.3f",
        fm_prauc, p_at_50 * 100, fm.threshold_,
    )
    fm.save(RESULTS / "fraud_model.pkl")

    # ── Model C: Exchange recommender ─────────────────────────────────────────
    logger.info("Training ExchangeRecommender…")
    er = ExchangeRecommender()
    er.fit(orders)
    logger.info("  Catalog: %d items", len(er.catalog_))
    er.save(RESULTS / "exchange_model.pkl")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = pd.DataFrame([
        {"model": "ReturnLikelihood",  "metric": "pr_auc",       "value": round(lm_prauc, 4)},
        {"model": "ReturnLikelihood",  "metric": "random_baseline", "value": round(y_true_rr.mean(), 4)},
        {"model": "FraudDetection",    "metric": "pr_auc",       "value": round(fm_prauc, 4)},
        {"model": "FraudDetection",    "metric": "precision_at_50", "value": round(p_at_50, 4)},
        {"model": "FraudDetection",    "metric": "threshold",    "value": round(fm.threshold_, 4)},
        {"model": "ExchangeRecommender", "metric": "catalog_size", "value": len(er.catalog_)},
    ])
    metrics.to_csv(RESULTS / "returns_metrics.csv", index=False)

    print("\n" + "=" * 55)
    print("Returns Intelligence Training Complete")
    print("=" * 55)
    print(f"  Likelihood PR-AUC  : {lm_prauc:.4f}  (random = {y_true_rr.mean():.4f})")
    print(f"  Fraud PR-AUC       : {fm_prauc:.4f}  (random = {y_true_fr.mean():.4f})")
    print(f"  Fraud Precision@50 : {p_at_50:.1%}")
    print(f"  Fraud threshold    : {fm.threshold_:.3f}")
    print(f"  Exchange catalog   : {len(er.catalog_):,} items")
    print("=" * 55)
    print(f"  Models saved to {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train returns intelligence models")
    parser.add_argument("--quick", action="store_true", help="Fast run with 2k customers")
    args = parser.parse_args()
    main(quick=args.quick)
