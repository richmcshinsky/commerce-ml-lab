"""Training script for the checkout intent and uplift models.

Usage:
    python -m intent.train            # train all three models on Criteo (or synthetic)
    python -m intent.train --quick    # 10k rows, fast params (CI / smoke)

Outputs (in results/):
    uplift_data.parquet
    propensity_model.pkl
    t_learner.pkl
    s_learner.pkl
    intent_metrics.csv
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


def main(quick: bool = False) -> None:
    from commerce_ml.data.loaders import CRITEO_DIR, generate_criteo_like, load_criteo
    from intent.evaluate import (
        incremental_conversions_at_k,
        qini_coefficient,
        uplift_at_k,
    )
    from intent.features import temporal_split
    from intent.models import PropensityModel, SLearner, TLearner

    RESULTS = _HERE.parent.parent / "results"
    RESULTS.mkdir(exist_ok=True)

    fast_params = {"n_estimators": 20 if quick else 500, "verbose": -1,
                   "random_state": 42, "n_jobs": -1}
    n_rows = 10_000 if quick else 200_000

    # ── Load data ─────────────────────────────────────────────────────────────
    criteo_path = CRITEO_DIR / "criteo_uplift_v2.1.csv.gz"
    if criteo_path.exists() and not quick:
        logger.info("Loading real Criteo data (10%% sample)...")
        df = load_criteo(sample_frac=0.10)
        df["segment"] = "unknown"
        source = "Criteo"
    else:
        logger.info("Generating synthetic Criteo-like data (%d rows)...", n_rows)
        df = generate_criteo_like(n_rows=n_rows, random_state=42)
        source = "Synthetic"

    logger.info("  %d rows | treatment_rate=%.1f%% | conversion_rate=%.2f%%",
                len(df), df["treatment"].mean()*100, df["conversion"].mean()*100)

    df.to_parquet(RESULTS / "uplift_data.parquet", index=False)

    train, test = temporal_split(df, test_frac=0.20, random_state=42)
    y, w = test["conversion"].values, test["treatment"].values
    ate = y[w == 1].mean() - y[w == 0].mean()
    logger.info("  ATE = %+.4f", ate)

    # ── Propensity model ──────────────────────────────────────────────────────
    logger.info("Training PropensityModel...")
    prop_model = PropensityModel(lgbm_params=fast_params)
    prop_model.fit(train)
    prop_scores = prop_model.predict_proba(test)
    prop_qini = qini_coefficient(y, w, prop_scores)
    logger.info("  Propensity Qini: %.4f", prop_qini)
    prop_model.save(RESULTS / "propensity_model.pkl")

    # ── T-learner ─────────────────────────────────────────────────────────────
    logger.info("Training TLearner...")
    t_model = TLearner(lgbm_params=fast_params)
    t_model.fit(train)
    t_scores = t_model.predict_cate(test)
    t_qini = qini_coefficient(y, w, t_scores)
    t_u20 = uplift_at_k(y, w, t_scores, k=0.20)
    logger.info("  T-learner  Qini=%.4f  Uplift@20%%=%.4f", t_qini, t_u20)
    t_model.save(RESULTS / "t_learner.pkl")

    # ── S-learner ─────────────────────────────────────────────────────────────
    logger.info("Training SLearner...")
    s_model = SLearner(lgbm_params=fast_params)
    s_model.fit(train)
    s_scores = s_model.predict_cate(test)
    s_qini = qini_coefficient(y, w, s_scores)
    s_u20 = uplift_at_k(y, w, s_scores, k=0.20)
    logger.info("  S-learner  Qini=%.4f  Uplift@20%%=%.4f", s_qini, s_u20)
    s_model.save(RESULTS / "s_learner.pkl")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = pd.DataFrame([
        {"model": "Propensity", "qini": round(prop_qini, 4),
         "uplift_at_20pct": round(uplift_at_k(y, w, prop_scores, k=0.20), 4),
         "incremental_conv_at_20pct": round(incremental_conversions_at_k(y, w, prop_scores, 0.20), 1)},
        {"model": "T-learner",  "qini": round(t_qini, 4),
         "uplift_at_20pct": round(t_u20, 4),
         "incremental_conv_at_20pct": round(incremental_conversions_at_k(y, w, t_scores, 0.20), 1)},
        {"model": "S-learner",  "qini": round(s_qini, 4),
         "uplift_at_20pct": round(s_u20, 4),
         "incremental_conv_at_20pct": round(incremental_conversions_at_k(y, w, s_scores, 0.20), 1)},
    ])
    metrics.to_csv(RESULTS / "intent_metrics.csv", index=False)

    print("\n" + "=" * 58)
    print(f"Checkout Intent Training Complete  [{source}]")
    print("=" * 58)
    print(metrics.to_string(index=False))
    print("=" * 58)
    print(f"  Models saved to {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(quick=parser.parse_args().quick)
