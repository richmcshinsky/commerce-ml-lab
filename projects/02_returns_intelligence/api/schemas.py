"""Pydantic request/response schemas for the returns intelligence API."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Return likelihood ─────────────────────────────────────────────────────────

class ReturnScoreRequest(BaseModel):
    """Score the return likelihood for a single order at fulfillment time."""

    order_id: str = Field(..., examples=["O0001234"])
    customer_id: str = Field(..., examples=["C000042"])
    category: str = Field(..., examples=["apparel"])
    item_price: float = Field(..., gt=0)
    quantity: int = Field(default=1, ge=1)
    channel: str = Field(..., examples=["web"])
    account_age_days: int = Field(default=365, ge=0)
    customer_lifetime_orders: int = Field(default=5, ge=0)
    customer_lifetime_return_rate: float = Field(default=0.12, ge=0, le=1)


class ReturnScoreResponse(BaseModel):
    """Return likelihood score and risk tier."""

    order_id: str
    return_probability: float = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., examples=["high"])  # low / medium / high


# ── Fraud detection ───────────────────────────────────────────────────────────

class FraudScoreRequest(BaseModel):
    """Score a return event for potential fraud or abuse."""

    return_id: str = Field(..., examples=["R0001234"])
    order_id: str = Field(..., examples=["O0001234"])
    customer_id: str = Field(..., examples=["C000042"])
    category: str = Field(default="apparel", examples=["apparel"])
    item_price: float = Field(default=65.0, gt=0)
    days_to_return: int = Field(..., ge=0)
    condition_reported: str = Field(..., examples=["used"])
    reason_code: str = Field(..., examples=["changed_mind"])
    account_age_days: int = Field(default=365, ge=0)
    customer_return_rate: float = Field(..., ge=0, le=1)
    customer_total_orders: int = Field(default=5, ge=0)
    customer_total_returns: int = Field(default=1, ge=0)
    shared_address_count: int = Field(default=0, ge=0,
                                      description="Other customer accounts sharing this delivery address")
    shared_payment_count: int = Field(default=0, ge=0,
                                      description="Other accounts sharing the same payment method")


class FraudScoreResponse(BaseModel):
    """Fraud detection result with explanation."""

    return_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_flagged: bool
    top_reasons: list[str]


# ── Exchange recommendation ───────────────────────────────────────────────────

class ExchangeRequest(BaseModel):
    """Request exchange candidates for a return."""

    return_id: str = Field(..., examples=["R0001234"])
    customer_id: str = Field(..., examples=["C000042"])
    original_item_id: str = Field(..., examples=["apparel_042"])
    return_reason: str = Field(..., examples=["too_small"])
    original_price: float = Field(default=65.0, gt=0)
    top_k: int = Field(default=3, ge=1, le=10)


class ExchangeCandidate(BaseModel):
    """A single exchange recommendation."""

    item_id: str
    score: float = Field(..., ge=0, le=1)
    reason: str


class ExchangeResponse(BaseModel):
    """Ranked exchange recommendations."""

    return_id: str
    candidates: list[ExchangeCandidate]
