"""Pydantic schemas for the Shipping Optimisation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionFeatures(BaseModel):
    """Checkout session features used for shipping price optimisation."""

    cart_value: float = Field(..., gt=0, description="Total cart value in USD")
    n_items: int = Field(..., ge=1, le=50, description="Number of items in cart")
    is_returning: bool = Field(..., description="True if the customer has ordered before")
    session_depth: int = Field(..., ge=1, description="Number of pages visited this session")
    time_on_checkout: float = Field(..., gt=0, description="Seconds spent on checkout page")
    device_mobile: bool = Field(..., description="True if customer is on a mobile device")
    f0: float = Field(0.0, description="Anonymous feature 0")
    f1: float = Field(0.0, description="Anonymous feature 1")
    f2: float = Field(0.0, description="Anonymous feature 2")
    f3: float = Field(0.0, description="Anonymous feature 3")
    f4: float = Field(0.0, description="Anonymous feature 4")
    f5: float = Field(0.0, description="Anonymous feature 5")


class ShippingOptionResult(BaseModel):
    """A single evaluated shipping option."""

    name: str
    price: float
    transit_days: str
    p_convert: float = Field(..., description="Predicted conversion probability at this price")
    expected_margin: float = Field(..., description="Expected margin = P(convert) × (product_margin + price - ship_cost)")


class ShippingRecommendationRequest(BaseModel):
    """Request body for POST /shipping/recommend."""

    session: SessionFeatures
    min_conversion_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional floor on conversion probability. Options below this are excluded.",
    )


class ShippingRecommendationResponse(BaseModel):
    """Response from POST /shipping/recommend."""

    recommended_option: str
    recommended_price: float
    transit_days: str
    p_convert: float
    expected_margin: float
    all_options: list[ShippingOptionResult]
