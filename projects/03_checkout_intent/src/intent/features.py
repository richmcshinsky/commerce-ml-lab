"""Feature engineering for the Criteo uplift dataset.

The Criteo v2.1 features (f0-f11) are already aggregated session-level
statistics with anonymised names. This module adds:
- Standardisation (z-score) for the continuous features
- Interaction terms: treatment × f_i for the S-learner
- Convenience helpers for train/test splitting
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS: list[str] = [f"f{i}" for i in range(12)]
"""The 12 anonymised Criteo features, identical to the raw column names."""

TREATMENT_COL: str = "treatment"
OUTCOME_COL:   str = "conversion"


def preprocess(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    clip_std: float = 5.0,
) -> pd.DataFrame:
    """Standardise continuous features and clip outliers.

    Parameters
    ----------
    df:
        Raw DataFrame with feature_cols present.
    feature_cols:
        Columns to standardise.
    clip_std:
        Clip values beyond ±clip_std standard deviations.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with feature columns standardised in-place.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        mu  = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = ((df[col] - mu) / std).clip(-clip_std, clip_std)
    return df


def add_treatment_interactions(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    treatment_col: str = TREATMENT_COL,
) -> pd.DataFrame:
    """Add treatment × feature interaction columns for the S-learner.

    Multiplying treatment (0/1) with each feature gives the model a direct
    way to learn heterogeneous treatment effects without needing a separate
    model per treatment arm.

    Parameters
    ----------
    df:
        Feature DataFrame with ``treatment_col`` present.
    feature_cols:
        Columns to interact with treatment.
    treatment_col:
        Binary treatment indicator.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``{col}_x_treat`` columns added.
    """
    df = df.copy()
    w = df[treatment_col].values
    for col in feature_cols:
        if col in df.columns:
            df[f"{col}_x_treat"] = df[col] * w
    return df


def get_feature_cols(df: pd.DataFrame, include_interactions: bool = False) -> list[str]:
    """Return the feature column names to pass to a model.

    Parameters
    ----------
    df:
        Feature DataFrame.
    include_interactions:
        If True, include treatment-interaction columns (for S-learner).

    Returns
    -------
    list[str]
        Ordered list of column names.
    """
    base = [c for c in FEATURE_COLS if c in df.columns]
    if not include_interactions:
        return base
    inter = [f"{c}_x_treat" for c in base if f"{c}_x_treat" in df.columns]
    return base + [TREATMENT_COL] + inter


def temporal_split(
    df: pd.DataFrame,
    test_frac: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split (Criteo has no timestamp, so random is appropriate).

    Parameters
    ----------
    df:
        Full dataset.
    test_frac:
        Fraction for test set.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    train, test : pd.DataFrame
    """
    mask = np.random.default_rng(random_state).random(len(df)) < test_frac
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
