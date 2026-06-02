"""Intent and uplift models.

Propensity (naive):
    Binary LightGBM classifier: P(convert | X).

Uplift — T-learner:
    Two separate models: mu1(X) = E[Y|X,W=1], mu0(X) = E[Y|X,W=0].
    CATE = mu1(X) - mu0(X).

Uplift — S-learner:
    Single model with treatment W as a feature.
    CATE = predict(X, W=1) - predict(X, W=0).
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)
