"""Synthetic returns data generator for Project 02.

Re-exports from the shared commerce_ml library and adds project-specific
helpers for splitting and visualising the generated dataset.
"""
from __future__ import annotations

from commerce_ml.data.synthetic import SyntheticConfig, generate_returns_dataset

__all__ = ["generate_returns_dataset", "SyntheticConfig"]
