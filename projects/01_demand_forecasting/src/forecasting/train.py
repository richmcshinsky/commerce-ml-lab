"""Training script for the LightGBM demand forecasting model.

Usage
-----
From the project root::

    make train-forecast

Or directly::

    cd projects/01_demand_forecasting
    PYTHONPATH=../../src:src python -m forecasting.train

Output
------
Saves the trained model to ``results/lgbm_model.pkl`` and a metrics CSV to
``results/lgbm_metrics.csv``.  Also prints a side-by-side comparison with
the baseline models.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Allow importing commerce_ml from the monorepo src
_REPO_SRC = Path(__file__).parents[5] / "src"
_PROJECT_SRC = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_SRC))
sys.path.insert(0, str(_PROJECT_SRC))

from forecasting.data import load_m5_long, train_test_split
from forecasting.lgbm_model import LGBMForecaster
from forecasting.models import evaluate_baselines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parents[3] / "results"
MODEL_PATH = RESULTS_DIR / "lgbm_model.pkl"
METRICS_PATH = RESULTS_DIR / "lgbm_metrics.csv"

TEST_DAYS: int = 28
STORE_ID: str = "CA_1"


def main() -> None:
    """Train the LightGBM model and evaluate against baselines."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading M5 data (store=%s)…", STORE_ID)
    df = load_m5_long(store_id=STORE_ID)
    logger.info("Loaded %d rows × %d columns.", *df.shape)

    train, test = train_test_split(df, test_days=TEST_DAYS)
    logger.info(
        "Train: %d rows (%d SKUs), Test: %d rows.",
        len(train),
        train["id"].nunique(),
        len(test),
    )

    # ── Baselines ─────────────────────────────────────────────────────────────
    logger.info("Evaluating baselines…")
    baseline_metrics = evaluate_baselines(train, test)
    logger.info("Baseline results:\n%s", baseline_metrics.to_string(index=False))

    # ── LightGBM training ──────────────────────────────────────────────────────
    logger.info("Training LightGBM global model…")
    model = LGBMForecaster()
    model.fit(train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    logger.info("Generating test predictions…")
    preds = model.predict(test)

    # Aggregate to store level for scalar metrics
    import sys

    sys.path.insert(0, str(_REPO_SRC))
    from commerce_ml.evaluation.forecast_metrics import summarise_forecast_metrics

    actual_agg = test.groupby("date")["sales"].sum().values
    forecast_agg = preds.groupby("date")["forecast"].sum().values
    train_agg = train.groupby("date")["sales"].sum().values

    # Use seasonal naive as the MASE reference
    from forecasting.models import SeasonalNaiveForecaster

    sn = SeasonalNaiveForecaster(seasonality=7).fit(train)
    sn_agg = sn.predict(test).groupby("date")["forecast"].sum().values

    lgbm_metrics = summarise_forecast_metrics(
        actual=actual_agg,
        forecast=forecast_agg,
        train_actual=train_agg,
        naive_forecast=sn_agg,
        label="LightGBM",
    )

    # ── Results table ─────────────────────────────────────────────────────────
    all_metrics = pd.concat([baseline_metrics, lgbm_metrics], ignore_index=True)
    all_metrics = all_metrics.sort_values("wmape").reset_index(drop=True)

    print("\n" + "=" * 60)
    print("  Demand Forecasting — Model Comparison")
    print("=" * 60)
    print(all_metrics.to_string(index=False, float_format="%.4f"))
    print("=" * 60)

    all_metrics.to_csv(METRICS_PATH, index=False)
    logger.info("Metrics saved to %s", METRICS_PATH)

    # ── Feature importance ────────────────────────────────────────────────────
    fi = model.get_feature_importance(top_n=15)
    print("\nTop 15 features by gain:")
    print(fi.to_string(index=False))

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Done.")


if __name__ == "__main__":
    main()
