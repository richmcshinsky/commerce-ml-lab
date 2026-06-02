"""Forecasting models: baselines, ETS, and LightGBM global model.

Model progression:
1. Naive baseline (last observed value)
2. Seasonal naive (same day, prior year/week)
3. Moving average
4. ETS / Holt-Winters (via statsforecast)
5. LightGBM global model (one model, all SKUs)
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
