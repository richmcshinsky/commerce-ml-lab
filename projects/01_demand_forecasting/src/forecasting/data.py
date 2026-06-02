"""M5 data loading and preprocessing for the demand forecasting project.

Handles the M5-specific transformation from wide format (one column per day)
to long format (one row per SKU-date), joins calendar and price tables, and
creates train/test splits respecting temporal ordering.
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from commerce_ml.data.loaders import load_m5_sales, load_m5_calendar, load_m5_prices

logger = logging.getLogger(__name__)
