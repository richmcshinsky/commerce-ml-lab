"""Pydantic request/response schemas for the forecasting API."""
from __future__ import annotations
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request body for the /forecast endpoint."""

    store_id: str = Field(..., example="CA_1")
    item_id: str = Field(..., example="HOBBIES_1_001")
    horizon_days: int = Field(default=28, ge=1, le=90, description="Forecast horizon in days")


class DayForecast(BaseModel):
    """Point forecast and prediction interval for a single day."""

    date: str
    forecast: float = Field(..., ge=0)
    lower_80: float = Field(..., ge=0)
    upper_80: float = Field(..., ge=0)


class ForecastResponse(BaseModel):
    """Response body for the /forecast endpoint."""

    store_id: str
    item_id: str
    forecasts: list[DayForecast]


class ReorderRequest(BaseModel):
    """Request body for the /reorder endpoint."""

    store_id: str
    item_id: str
    current_inventory: int = Field(..., ge=0)
    lead_time_days: int = Field(..., ge=1)
    service_level: float = Field(default=0.95, ge=0.5, le=0.999)


class ReorderResponse(BaseModel):
    """Reorder recommendation from the inventory optimisation layer."""

    store_id: str
    item_id: str
    reorder_now: bool
    recommended_quantity: int
    safety_stock: int
    reorder_point: int
    reasoning: str
