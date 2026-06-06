"""Shipping Price Optimisation API.

Endpoints
---------
GET  /health              — liveness check
POST /shipping/recommend  — optimal shipping option for a checkout session

Run locally:
    make serve-shipping
    # then open http://localhost:8003/docs
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .schemas import (
    ShippingOptionResult,
    ShippingRecommendationRequest,
    ShippingRecommendationResponse,
)

logger = logging.getLogger(__name__)
_HERE = Path(__file__).parent
_RESULTS = _HERE.parent / "results"
_SRC = _HERE.parent / "src"


class _AppState:
    elasticity_model: Any = None
    optimizer: Any = None


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load the elasticity model and build the optimizer at startup."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    model_path = _RESULTS / "elasticity_model.pkl"
    if not model_path.exists():
        logger.warning("elasticity_model.pkl not found — run `make train-shipping` first.")
    else:
        try:
            from shipping.elasticity import ConversionElasticityModel
            from shipping.optimizer import ShippingPriceOptimizer

            _state.elasticity_model = ConversionElasticityModel.load(model_path)
            _state.optimizer = ShippingPriceOptimizer(_state.elasticity_model)
            logger.info("Elasticity model loaded from %s.", model_path)
        except Exception as exc:
            logger.error("Failed to load elasticity model: %s", exc)

    yield

    _state.elasticity_model = None
    _state.optimizer = None


app = FastAPI(
    title="Commerce ML — Shipping Price Optimisation",
    description=(
        "Selects the shipping option that maximises expected margin per checkout session "
        "without materially hurting conversion rate."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _require_optimizer() -> Any:
    if _state.optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `make train-shipping` then restart.",
        )
    return _state.optimizer


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "elasticity_model": "ready" if _state.elasticity_model else "not_loaded",
    }


@app.post("/shipping/recommend", response_model=ShippingRecommendationResponse)
def recommend_shipping(request: ShippingRecommendationRequest) -> ShippingRecommendationResponse:
    """Return the optimal shipping option for a checkout session.

    Given session features, evaluates all shipping price tiers and returns the
    one that maximises ``P(convert) × (product_margin + price - ship_cost)``.
    """
    import pandas as pd

    optimizer = _require_optimizer()

    session = pd.Series(
        {
            "cart_value": request.session.cart_value,
            "n_items": request.session.n_items,
            "is_returning": request.session.is_returning,
            "session_depth": request.session.session_depth,
            "time_on_checkout": request.session.time_on_checkout,
            "device_mobile": request.session.device_mobile,
            "f0": request.session.f0,
            "f1": request.session.f1,
            "f2": request.session.f2,
            "f3": request.session.f3,
            "f4": request.session.f4,
            "f5": request.session.f5,
        }
    )

    if request.min_conversion_rate is not None:
        from shipping.optimizer import ShippingPriceOptimizer

        opt = ShippingPriceOptimizer(
            _state.elasticity_model,
            min_p_convert=request.min_conversion_rate,
        )
    else:
        opt = optimizer

    result = opt.recommend(session)

    return ShippingRecommendationResponse(
        recommended_option=result.recommended.name,
        recommended_price=result.recommended.price,
        transit_days=result.recommended.transit_days,
        p_convert=result.p_convert,
        expected_margin=result.expected_margin,
        all_options=[
            ShippingOptionResult(
                name=str(row["option"]),
                price=float(row["price"]),
                transit_days=str(row["transit_days"]),
                p_convert=float(row["p_convert"]),
                expected_margin=float(row["expected_margin"]),
            )
            for _, row in result.breakdown.iterrows()
        ],
    )
