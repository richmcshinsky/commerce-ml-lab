"""Demand Forecasting API.

Endpoints
---------
GET  /health          — liveness check
POST /forecast        — point forecasts + prediction intervals for a SKU
POST /reorder         — reorder recommendation given current inventory + lead time

Run locally:
    make serve-forecast
    # then open http://localhost:8001/docs
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from .schemas import ForecastRequest, ForecastResponse, ReorderRequest, ReorderResponse

app = FastAPI(
    title="Commerce ML — Demand Forecasting",
    description="Demand forecasts and inventory reorder recommendations.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate point forecasts and prediction intervals for a SKU.

    Returns day-level forecasts for the next ``horizon_days`` days,
    including 80% prediction intervals from quantile regression.
    """
    # TODO: load trained model and generate forecasts
    raise HTTPException(status_code=501, detail="Model not yet trained. Run `make train-forecast`.")


@app.post("/reorder", response_model=ReorderResponse)
def reorder(request: ReorderRequest) -> ReorderResponse:
    """Recommend whether to reorder and how much.

    Uses the newsvendor model with forecasted demand + prediction intervals
    to compute a reorder recommendation given current inventory and lead time.
    """
    # TODO: load trained model and run inventory optimisation
    raise HTTPException(status_code=501, detail="Model not yet trained. Run `make train-forecast`.")
