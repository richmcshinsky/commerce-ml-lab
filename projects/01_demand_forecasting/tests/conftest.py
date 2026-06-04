"""Pytest fixtures for the demand forecasting project.

Provides a synthetic M5-like dataset so tests run without downloading
the real M5 data. The synthetic data has the same schema as the real
dataset and follows similar statistical properties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_m5_long() -> pd.DataFrame:
    """Generate a minimal synthetic M5-like long-format DataFrame.

    Properties:
    - 10 SKUs, 3 categories (FOODS, HOBBIES, HOUSEHOLD)
    - 120 days of data (2020-01-01 to 2020-04-29)
    - Weekly seasonality planted (weekends ~30% higher)
    - ~60% zero-sales days (realistic for slow-moving items)
    - Prices vary by category

    Returns
    -------
    pd.DataFrame
        Same schema as load_m5_long() output.
    """
    rng = np.random.default_rng(42)

    n_days = 120
    start_date = pd.Timestamp("2020-01-01")

    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # Define SKUs
    categories = ["FOODS_1", "FOODS_2", "HOBBIES_1", "HOBBIES_2",
                  "HOUSEHOLD_1", "HOUSEHOLD_2", "FOODS_3", "FOODS_4",
                  "HOBBIES_3", "HOUSEHOLD_3"]
    cat_map = {
        "FOODS_1": "FOODS", "FOODS_2": "FOODS", "FOODS_3": "FOODS", "FOODS_4": "FOODS",
        "HOBBIES_1": "HOBBIES", "HOBBIES_2": "HOBBIES", "HOBBIES_3": "HOBBIES",
        "HOUSEHOLD_1": "HOUSEHOLD", "HOUSEHOLD_2": "HOUSEHOLD", "HOUSEHOLD_3": "HOUSEHOLD",
    }
    price_by_cat = {"FOODS": 5.0, "HOBBIES": 15.0, "HOUSEHOLD": 10.0}

    rows = []
    for i, item_id in enumerate(categories):
        cat = cat_map[item_id]
        base_rate = rng.uniform(0.3, 2.0)  # mean daily sales
        price = price_by_cat[cat] * rng.uniform(0.8, 1.2)

        for d, date in enumerate(dates):
            # Weekly seasonality: weekends sell more
            dow_mult = 1.3 if date.dayofweek >= 5 else 1.0
            mu = base_rate * dow_mult

            # Poisson sales, clipped to mimic intermittent demand
            sales = int(rng.poisson(mu))
            if rng.random() < 0.45:  # inject zeros for ~60% overall zero rate
                sales = 0

            rows.append({
                "id": f"{item_id}_CA_1",
                "item_id": item_id,
                "dept_id": item_id,
                "cat_id": cat,
                "store_id": "CA_1",
                "state_id": "CA",
                "d": f"d_{d + 1}",
                "date": date,
                "sales": sales,
                "sell_price": round(price, 2),
                "event_name_1": None,
                "event_type_1": None,
                "snap_CA": int(rng.random() < 0.14),  # SNAP ~14% of days
                "snap_TX": 0,
                "snap_WI": 0,
                "wm_yr_wk": int(date.strftime("%Y%V")),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def small_train_test(synthetic_m5_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test split of the synthetic data (28-day test window)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
    from forecasting.data import train_test_split
    return train_test_split(synthetic_m5_long, test_days=28)
