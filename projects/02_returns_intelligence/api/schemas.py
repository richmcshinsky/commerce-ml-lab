"""Pydantic request/response schemas for the returns intelligence API."""
from __future__ import annotations
from pydantic import BaseModel, Field


# ── Return likelihood ─────────────────────────────────────────────────────────

class ReturnScoreRequest(BaseModel):
    """Score the return likelihood for a single order."""
    order_id: str
    customer_id: str
    category: str = Field(..., example="apparel")
    item_price: float = Field(..., gt=0)
    customer_lifetime_orders: int = Field(..., ge=0)
    customer_lifetime_return_rate: float = Field(..., ge=0, le=1)
    days_since_last_order: int = Field(..., ge=0)
    channel: str = Field(..., example="web")


class ReturnScoreResponse(BaseModel):
    order_id: str
    return_probability: float = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., example="high")  # low / medium / high


# ── Fraud detection ───────────────────────────────────────────────────────────

class FraudScoreRequest(BaseModel):
    """Score a return event for potential fraud/abuse."""
    return_id: str
    customer_id: str
    order_id: str
    days_to_return: int = Field(..., ge=0)
    condition_reported: str = Field(..., example="used")
    reason_code: str = Field(..., example="changed_mind")
    customer_return_rate: float = Field(..., ge=0, le=1)
    shared_address_count: int = Field(default=0, ge=0)


class FraudScoreResponse(BaseModel):
    return_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_flagged: bool
    top_reasons: list[str]


# ── Exchange recommendation ───────────────────────────────────────────────────

class ExchangeRequest(BaseModel):
    """Request exchange recommendations for a return."""
    return_id: str
    customer_id: str
    original_item_id: str
    return_reason: str = Field(..., example="too_small")
    top_k: int = Field(default=3, ge=1, le=10)


class ExchangeCandidate(BaseModel):
    item_id: str
    score: float
    reason: str  # why this was recommended


class ExchangeResponse(BaseModel):
    return_id: str
    candidates: list[ExchangeCandidate]
