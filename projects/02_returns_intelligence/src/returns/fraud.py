"""Returns fraud / abuse detection model.

Detects fraud archetypes: wardrobers, velocity returners, and address-sharing
rings. Key design decisions:

1. Tabular LightGBM on behavioural and order features as the base model.
2. Graph features (networkx) computed from shared address/payment edges —
   these lift PR-AUC meaningfully on ring fraud.
3. Isotonic calibration so output probabilities are meaningful.
4. Cost-aware threshold selected via business loss matrix, not 0.5 default.
5. SHAP reason codes for every flagged return (explainability requirement).
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
