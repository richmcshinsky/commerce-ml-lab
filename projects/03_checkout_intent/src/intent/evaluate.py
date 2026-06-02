"""Evaluation utilities for uplift models (Qini curves, uplift at K)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from commerce_ml.evaluation.classification_metrics import qini_coefficient

__all__ = ["qini_coefficient"]
