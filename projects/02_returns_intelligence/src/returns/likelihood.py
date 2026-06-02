"""Return likelihood model: P(return | order features).

Predicts the probability that an order will be returned. Used for:
- Risk scoring at fulfilment time
- Return forecasting (aggregate expected returns per store per week)
- Feature input to the fraud model
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
