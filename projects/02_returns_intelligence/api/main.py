"""Returns Intelligence API.

Endpoints
---------
GET  /health                  — liveness check
POST /returns/score           — return likelihood for an order
POST /returns/fraud           — fraud score for a return event
POST /returns/exchange        — exchange recommendations

Run locally:
    make serve-returns
    # then open http://localhost:8002/docs
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .schemas import (
    ExchangeCandidate,
    ExchangeRequest,
    ExchangeResponse,
    FraudScoreRequest,
    FraudScoreResponse,
    ReturnScoreRequest,
    ReturnScoreResponse,
)

logger = logging.getLogger(__name__)
_HERE    = Path(__file__).parent
_RESULTS = _HERE.parent / "results"


class _AppState:
    likelihood_model: Any = None
    fraud_model: Any = None
    exchange_model: Any = None


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load all three models at startup."""
    src = _HERE.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    for attr, fname, cls_name, mod_name in [
        ("likelihood_model", "likelihood_model.pkl", "ReturnLikelihoodModel", "returns.likelihood"),
        ("fraud_model",      "fraud_model.pkl",      "FraudDetectionModel",  "returns.fraud"),
        ("exchange_model",   "exchange_model.pkl",   "ExchangeRecommender",  "returns.exchange"),
    ]:
        path = _RESULTS / fname
        if not path.exists():
            logger.warning("%s not found — run `make train-returns` first.", path)
            continue
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            setattr(_state, attr, cls.load(path))
            logger.info("Loaded %s from %s", cls_name, path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", fname, exc)

    yield

    _state.likelihood_model = None
    _state.fraud_model = None
    _state.exchange_model = None


app = FastAPI(
    title="Commerce ML — Returns Intelligence",
    description="Return likelihood scoring, fraud detection, and exchange recommendations.",
    version="0.2.0",
    lifespan=lifespan,
)


def _require(model: Any, name: str) -> Any:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"{name} not loaded. Run `make train-returns` then restart.",
        )
    return model


@app.get("/health")
def health() -> dict[str, str]:
    loaded = {
        "likelihood": "ready" if _state.likelihood_model else "not_loaded",
        "fraud":      "ready" if _state.fraud_model else "not_loaded",
        "exchange":   "ready" if _state.exchange_model else "not_loaded",
    }
    return {"status": "ok", **loaded}


@app.post("/returns/score", response_model=ReturnScoreResponse)
def score_return(request: ReturnScoreRequest) -> ReturnScoreResponse:
    """Predict the probability that this order will be returned.

    Uses order-level and customer-level features available at fulfillment time.
    Returns a probability and a risk tier (low / medium / high).
    """
    import pandas as pd
    model = _require(_state.likelihood_model, "ReturnLikelihoodModel")

    order_df = pd.DataFrame([{
        "order_id":    request.order_id,
        "customer_id": request.customer_id,
        "category":    request.category,
        "item_price":  request.item_price,
        "quantity":    request.quantity,
        "channel":     request.channel,
    }])
    customer_df = pd.DataFrame([{
        "customer_id":            request.customer_id,
        "account_age_days":       request.account_age_days,
        "lifetime_return_rate":   request.customer_lifetime_return_rate,
        "total_orders":           request.customer_lifetime_orders,
    }])

    result = model.predict_with_tier(order_df, customer_df)
    row = result.iloc[0]
    return ReturnScoreResponse(
        order_id=request.order_id,
        return_probability=float(row["return_probability"]),
        risk_tier=str(row["risk_tier"]),
    )


@app.post("/returns/fraud", response_model=FraudScoreResponse)
def score_fraud(request: FraudScoreRequest) -> FraudScoreResponse:
    """Score a return event for potential fraud or abuse.

    Graph features (address/payment sharing) require customer network data;
    the API uses the values provided directly in the request.
    """
    import pandas as pd
    model = _require(_state.fraud_model, "FraudDetectionModel")

    returns_df = pd.DataFrame([{
        "return_id":     request.return_id,
        "order_id":      request.order_id,
        "customer_id":   request.customer_id,
        "days_to_return": request.days_to_return,
        "condition":     request.condition_reported,
        "reason_code":   request.reason_code,
        # Graph features provided directly from request
        "shared_address_count": request.shared_address_count,
        "shared_payment_count": request.shared_payment_count,
        "component_size":       max(1, request.shared_address_count + 1),
    }])
    orders_df = pd.DataFrame([{
        "order_id":   request.order_id,
        "category":   request.category,
        "item_price": request.item_price,
    }])
    customers_df = pd.DataFrame([{
        "customer_id":            request.customer_id,
        "account_age_days":       request.account_age_days,
        "lifetime_return_rate":   request.customer_return_rate,
        "total_orders":           request.customer_total_orders,
        "total_returns":          request.customer_total_returns,
        "address_id":             f"ADDR_{request.customer_id}",
        "payment_hash":           f"PAY_{request.customer_id}",
    }])

    # Build features directly without re-running graph computation
    from returns.fraud import _build_fraud_features
    feat_df = _build_fraud_features(returns_df, orders_df, customers_df)
    X = feat_df[[c for c in model.feature_cols_ if c in feat_df.columns]]
    # Pad any missing feature columns with 0
    for col in model.feature_cols_:
        if col not in X.columns:
            X = X.copy(); X[col] = 0
    X = X[model.feature_cols_]

    score = float(model.predict_proba_raw(X)[0])
    is_flagged = score >= model.threshold_

    # Top reasons from model importance
    top_reasons = list(model._shap_reason_codes(X)[0])

    return FraudScoreResponse(
        return_id=request.return_id,
        fraud_probability=round(score, 4),
        is_flagged=is_flagged,
        top_reasons=top_reasons,
    )


@app.post("/returns/exchange", response_model=ExchangeResponse)
def recommend_exchange(request: ExchangeRequest) -> ExchangeResponse:
    """Recommend exchange candidates for a return event."""
    model = _require(_state.exchange_model, "ExchangeRecommender")

    recs = model.recommend(
        item_id=request.original_item_id,
        reason_code=request.return_reason,
        original_price=request.original_price,
        top_k=request.top_k,
    )
    candidates = [
        ExchangeCandidate(
            item_id=row["item_id"],
            score=float(row["score"]),
            reason=str(row["rule_applied"]),
        )
        for _, row in recs.iterrows()
    ]
    return ExchangeResponse(
        return_id=request.return_id,
        candidates=candidates,
    )
