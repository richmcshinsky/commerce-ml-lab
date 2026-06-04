"""Pydantic request/response schemas for the forecasting API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request body for the /forecast endpoint."""

    store_id: str = Field(..., examples=["CA_1"])
    item_id: str = Field(..., examples=["HOBBIES_1_001"])
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

    store_id: str = Field(..., examples=["CA_1"])
    item_id: str = Field(..., examples=["HOBBIES_1_001"])
    current_inventory: int = Field(..., ge=0, description="Current on-hand inventory (units)")
    lead_time_days: int = Field(default=7, ge=1, description="Replenishment lead time in days")
    service_level: float = Field(
        default=0.95, ge=0.5, le=0.999, description="Target cycle-service level (e.g. 0.95)"
    )
    cost_overstock: float = Field(
        default=1.0, gt=0, description="Per-unit overage cost (holding + disposal)"
    )
    cost_understock: float = Field(
        default=5.0, gt=0, description="Per-unit underage cost (lost margin + goodwill)"
    )


class ReorderResponse(BaseModel):
    """Reorder recommendation from the inventory optimisation layer."""

    store_id: str
    item_id: str
    reorder_now: bool
    recommended_quantity: int
    safety_stock: int
    reorder_point: int
    reasoning: str
