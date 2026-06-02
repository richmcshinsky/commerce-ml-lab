"""Exchange recommendation engine.

Given a return event and a stated reason, recommends the best exchange
candidate(s) to offer the customer.

Design:
1. Heuristic filter by reason code (fast, interpretable, covers ~70% of cases).
2. LightGBM ranker (LambdaRank objective) scores filtered candidates using
   item popularity, price proximity, customer history, and category affinity.

Heuristic rules:
- "too_small"      -> same item, next size up
- "too_large"      -> same item, next size down
- "wrong_color"    -> same item family, all available colors
- "changed_mind"   -> same category, top-N by popularity
- "defective"      -> exact replacement (same SKU)
"""
from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)
