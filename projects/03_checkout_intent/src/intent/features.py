"""Session feature engineering for the intent model.

Re-exports from shared session_features module and adds Criteo-specific
preprocessing.
"""
from __future__ import annotations
import pandas as pd
from commerce_ml.features.session_features import build_session_features

__all__ = ["build_session_features"]
