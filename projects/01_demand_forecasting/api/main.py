"""Demand Forecasting API.

Endpoints
---------
GET  /health          — liveness check
POST /forecast        — point forecasts + prediction intervals for a SKU
POST /reorder         — reorder recommendation given current inventory + lead time

Run locally:
    make serve-forecast
    # then open http://localhost:8001/docs

Implementation notes
--------------------
On startup the API loads the trained LightGBM model and the M5 CA_1 dataset.
The last ``max_history_rows`` rows of training data are kept in memory as the
"current state" of each SKU; the model uses this history to construct lag and
rolling features for the forecast horizon.

For a production deployment, the history would be fetched from a real-time
database (DynamoDB, BigQuery, etc.) rather than the M5 flat file.
"""
from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import (
    DayForecast,
    ForecastRequest,
    ForecastResponse,
    ReorderRequest,
    ReorderResponse,
)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_RESULTS = _HERE.parent / "results"
_MODEL_PATH = _RESULTS / "lgbm_model.pkl"


# ── App-level state ────────────────────────────────────────────────────────────

class _AppState:
    model: Any = None        # LGBMForecaster
    history: pd.DataFrame | None = None   # training tail for all SKUs


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load model and data at startup; release at shutdown."""
    import sys
    src = _HERE.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if not _MODEL_PATH.exists():
        logger.warning(
            "Model not found at %s — run `make train-forecast` first. "
            "Endpoints will return 503 until the model is available.",
            _MODEL_PATH,
        )
    else:
        try:
            from forecasting.lgbm_model import LGBMForecaster
            from forecasting.data import load_m5_long, train_test_split

            _state.model = LGBMForecaster.load(_MODEL_PATH)
            df = load_m5_long()
            train, _ = train_test_split(df, test_days=28)
            # Keep enough history per SKU to construct all lag features
            _state.history = (
                train.sort_values(["id", "date"])
                .groupby("id", group_keys=False)
                .tail(_state.model.max_history_rows)
                .copy()
            )
            logger.info(
                "Loaded LGBMForecaster and history (%d rows, %d SKUs).",
                len(_state.history),
                _state.history["id"].nunique(),
            )
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)

    yield  # app runs here

    _state.model = None
    _state.history = None


app = FastAPI(
    title="Commerce ML — Demand Forecasting",
    description="Demand forecasts and inventory reorder recommendations for M5 CA_1.",
    version="0.2.0",
    lifespan=lifespan,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_model() -> Any:
    if _state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `make train-forecast` then restart the server.",
        )
    return _state.model


def _build_future_df(
    item_id: str,
    store_id: str,
    horizon_days: int,
) -> pd.DataFrame:
    """Construct a future DataFrame for the given SKU and horizon.

    Uses the last known sell price and sets event columns to None (no event).
    In production this would pull live prices / calendar data from a database.
    """
    assert _state.history is not None

    # M5 IDs include a dataset suffix, e.g. "FOODS_1_001_CA_1_evaluation".
    # Accept the user-facing form "FOODS_1_001" + "CA_1" and resolve the
    # actual stored ID by prefix matching.
    base_id = f"{item_id}_{store_id}"
    all_ids = _state.history["id"].unique()

    # 1. Exact match (works if data has no suffix)
    if base_id in all_ids:
        sku_id = base_id
    else:
        # 2. Prefix match — picks up _evaluation / _validation suffix variants
        matches = [i for i in all_ids if i.startswith(base_id)]
        if not matches:
            # Show a few valid item_id examples by stripping the store+suffix
            sample_items = []
            for raw_id in all_ids[:5]:
                # strip known suffixes: _evaluation, _validation
                clean = str(raw_id)
                for suffix in ("_evaluation", "_validation", f"_{store_id}"):
                    if clean.endswith(suffix):
                        clean = clean[: -len(suffix)]
                sample_items.append(clean)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"SKU '{base_id}' not found in training data for store '{store_id}'. "
                    f"Pass only the item portion of the ID, e.g. 'FOODS_1_001'. "
                    f"Sample valid item_ids: {', '.join(sample_items)}"
                ),
            )
        sku_id = matches[0]  # prefer the first (usually _evaluation)

    sku_hist = _state.history[_state.history["id"] == sku_id]

    last_date = pd.to_datetime(sku_hist["date"].max())
    last_price = float(sku_hist["sell_price"].iloc[-1]) if "sell_price" in sku_hist.columns else 0.0

    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]
    last_row = sku_hist.iloc[-1]

    rows = []
    for d in future_dates:
        row: dict = {
            "id":           sku_id,
            "date":         d,
            "sales":        np.nan,
            "sell_price":   last_price,
            "event_type_1": None,
        }
        # Carry forward categorical columns
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            if col in last_row.index:
                row[col] = last_row[col]
        rows.append(row)

    return pd.DataFrame(rows)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    loaded = "ready" if _state.model is not None else "model_not_loaded"
    return {"status": loaded}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate point forecasts and 80% prediction intervals for a SKU.

    Returns day-level forecasts for the next ``horizon_days`` days.
    Lag and rolling features are constructed from the training history tail
    stored at startup — no look-ahead into the future.
    """
    model = _require_model()
    future_df = _build_future_df(request.item_id, request.store_id, request.horizon_days)

    result = model.predict_with_intervals(future_df)
    result = result.sort_values("date").reset_index(drop=True)

    forecasts = [
        DayForecast(
            date=row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime")
                 else str(row["date"])[:10],
            forecast=round(float(row["forecast"]), 4),
            lower_80=round(float(row["lower_80"]), 4),
            upper_80=round(float(row["upper_80"]), 4),
        )
        for _, row in result.iterrows()
    ]

    return ForecastResponse(
        store_id=request.store_id,
        item_id=request.item_id,
        forecasts=forecasts,
    )


@app.post("/reorder", response_model=ReorderResponse)
def reorder(request: ReorderRequest) -> ReorderResponse:
    """Recommend whether to reorder and how much.

    Uses the newsvendor model with the LightGBM forecast to compute
    a reorder recommendation given current inventory and lead time.

    Decision logic
    --------------
    1. Forecast demand for the next ``lead_time_days`` days.
    2. Derive σ from the 80% prediction interval.
    3. Compute reorder point s = μ × L + z(SL) × σ × √L.
    4. If current_inventory ≤ s: reorder to S = s + μ × 7 (one week above ROP).
    5. The recommended quantity is max(0, S - current_inventory).
    """
    from forecasting.inventory import (
        std_from_interval,
        reorder_point as compute_rop,
        order_up_to_level,
        safety_stock as compute_ss,
    )

    model = _require_model()
    future_df = _build_future_df(request.item_id, request.store_id, request.lead_time_days)

    preds = model.predict_with_intervals(future_df)

    mean_d    = float(preds["forecast"].mean())
    mean_low  = float(preds["lower_80"].mean())
    mean_high = float(preds["upper_80"].mean())
    std_d     = std_from_interval(mean_low, mean_high)

    ss   = compute_ss(std_d, request.lead_time_days, request.service_level)
    rop  = compute_rop(mean_d, request.lead_time_days, std_d, request.service_level)
    oul  = order_up_to_level(rop, mean_d, review_period=7)

    should_reorder = request.current_inventory <= rop
    qty = max(0, math.ceil(oul - request.current_inventory)) if should_reorder else 0

    if should_reorder:
        reasoning = (
            f"Current inventory ({request.current_inventory} units) ≤ reorder point "
            f"({rop:.1f} units). Forecast: {mean_d:.2f} units/day over {request.lead_time_days}-day "
            f"lead time. Safety stock: {ss:.1f} units at {request.service_level:.0%} SL. "
            f"Order {qty} units to reach order-up-to level {oul:.1f}."
        )
    else:
        reasoning = (
            f"Current inventory ({request.current_inventory} units) > reorder point "
            f"({rop:.1f} units). No action needed. "
            f"Next review when inventory falls to {rop:.1f} units."
        )

    return ReorderResponse(
        store_id=request.store_id,
        item_id=request.item_id,
        reorder_now=should_reorder,
        recommended_quantity=qty,
        safety_stock=math.ceil(ss),
        reorder_point=math.ceil(rop),
        reasoning=reasoning,
    )
