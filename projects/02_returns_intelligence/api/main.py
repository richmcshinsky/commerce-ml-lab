"""Returns Intelligence API.

Endpoints
---------
GET  /health                  — liveness check
POST /returns/score           — return likelihood score for an order
POST /returns/fraud           — fraud / abuse score for a return event
POST /returns/exchange        — exchange recommendations given return + reason

Run locally:
    make serve-returns
    # then open http://localhost:8002/docs
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from .schemas import (
    ReturnScoreRequest, ReturnScoreResponse,
    FraudScoreRequest, FraudScoreResponse,
    ExchangeRequest, ExchangeResponse,
)

app = FastAPI(
    title="Commerce ML — Returns Intelligence",
    description="Return likelihood scoring, fraud detection, and exchange recommendations.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/returns/score", response_model=ReturnScoreResponse)
def score_return(request: ReturnScoreRequest) -> ReturnScoreResponse:
    """Predict the probability that this order will be returned."""
    raise HTTPException(status_code=501, detail="Model not yet trained. Run `make train-returns`.")


@app.post("/returns/fraud", response_model=FraudScoreResponse)
def score_fraud(request: FraudScoreRequest) -> FraudScoreResponse:
    """Score a return event for potential fraud or abuse."""
    raise HTTPException(status_code=501, detail="Model not yet trained. Run `make train-returns`.")


@app.post("/returns/exchange", response_model=ExchangeResponse)
def recommend_exchange(request: ExchangeRequest) -> ExchangeResponse:
    """Recommend exchange candidates given a return reason."""
    raise HTTPException(status_code=501, detail="Model not yet trained. Run `make train-returns`.")
