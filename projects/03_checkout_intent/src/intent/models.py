"""Intent and uplift models for the checkout conversion problem.

Three models, increasing sophistication:

1. **PropensityModel** — P(convert | X, ignoring treatment).
   The "naive" approach. Correct for predicting who will convert, but
   wrong for targeting interventions (sure-things have high propensity
   but zero uplift; targeting them wastes budget).

2. **TLearner** — Two separate models per treatment arm.
   mu1(X) = E[Y | X, W=1], mu0(X) = E[Y | X, W=0].
   CATE(X) = mu1(X) − mu0(X).
   Pros: simple, each model optimised for its arm.
   Cons: each model only sees half the data; extrapolation across arms.

3. **SLearner** — Single model with W as a feature plus interactions.
   CATE(X) = f(X, W=1, X*1) − f(X, W=0, X*0).
   Pros: uses all data; can regularise the treatment effect.
   Cons: treatment effect may be under-regularised / swamped by main effects.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from intent.features import (
    FEATURE_COLS,
    OUTCOME_COL,
    TREATMENT_COL,
    add_treatment_interactions,
    get_feature_cols,
    preprocess,
)

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def _make_lgbm(objective: str = "binary", **overrides: Any) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise ImportError("lightgbm required: pip install lightgbm") from e
    params = {**_DEFAULT_PARAMS, "objective": objective, **overrides}
    return LGBMClassifier(**params)


# ── Propensity model ──────────────────────────────────────────────────────────

class PropensityModel:
    """Standard binary classifier: P(convert | X).

    This is the *wrong* model for targeting interventions — it predicts who
    will convert, not who will convert *because of* the intervention.

    Included explicitly to demonstrate why propensity-based targeting is
    suboptimal: sure-things have high propensity scores but near-zero CATE.
    Targeting them wastes incentive budget on users who would convert anyway.

    Parameters
    ----------
    lgbm_params:
        LightGBM hyperparameters.
    feature_cols:
        Feature columns to use. Defaults to FEATURE_COLS (f0-f11).
    """

    def __init__(
        self,
        lgbm_params: Optional[dict[str, Any]] = None,
        feature_cols: list[str] = FEATURE_COLS,
    ) -> None:
        self.lgbm_params = lgbm_params or _DEFAULT_PARAMS.copy()
        self.feature_cols = feature_cols
        self.model_: Any = None

    @property
    def name(self) -> str:
        return "Propensity"

    def fit(
        self,
        train: pd.DataFrame,
        outcome_col: str = OUTCOME_COL,
    ) -> "PropensityModel":
        """Train on all rows (ignores treatment assignment).

        Parameters
        ----------
        train:
            Training DataFrame. Must contain feature_cols and outcome_col.
        outcome_col:
            Binary outcome column.

        Returns
        -------
        self
        """
        feat = preprocess(train)[self.feature_cols]
        y = train[outcome_col].values
        self.model_ = _make_lgbm(**self.lgbm_params)
        self.model_.fit(feat, y)
        logger.info("PropensityModel trained on %d rows.", len(train))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(convert) for each row."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        feat = preprocess(X)[self.feature_cols]
        return self.model_.predict_proba(feat)[:, 1]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "PropensityModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)


# ── T-learner ─────────────────────────────────────────────────────────────────

class TLearner:
    """Two-model uplift estimator.

    Trains separate LightGBM classifiers on treated and control rows.
    CATE(X) = mu1(X) − mu0(X).

    This is the most interpretable uplift approach. Each model is optimised
    for its treatment arm. The main weakness is that each model sees only
    half the data, which increases variance in small datasets.

    Parameters
    ----------
    lgbm_params:
        Shared hyperparameters for both arm models.
    feature_cols:
        Feature columns to use.
    """

    def __init__(
        self,
        lgbm_params: Optional[dict[str, Any]] = None,
        feature_cols: list[str] = FEATURE_COLS,
    ) -> None:
        self.lgbm_params = lgbm_params or _DEFAULT_PARAMS.copy()
        self.feature_cols = feature_cols
        self.mu1_: Any = None  # treated arm model
        self.mu0_: Any = None  # control arm model

    @property
    def name(self) -> str:
        return "T-learner"

    def fit(
        self,
        train: pd.DataFrame,
        treatment_col: str = TREATMENT_COL,
        outcome_col: str = OUTCOME_COL,
    ) -> "TLearner":
        """Train mu1 on treated rows and mu0 on control rows.

        Parameters
        ----------
        train:
            Training DataFrame with treatment_col and outcome_col.

        Returns
        -------
        self
        """
        feat_df = preprocess(train)
        treated = feat_df[feat_df[treatment_col] == 1]
        control = feat_df[feat_df[treatment_col] == 0]

        X1, y1 = treated[self.feature_cols], treated[outcome_col].values
        X0, y0 = control[self.feature_cols], control[outcome_col].values

        self.mu1_ = _make_lgbm(**self.lgbm_params)
        self.mu1_.fit(X1, y1)

        self.mu0_ = _make_lgbm(**self.lgbm_params)
        self.mu0_.fit(X0, y0)

        logger.info(
            "TLearner trained: %d treated, %d control rows.",
            len(treated), len(control),
        )
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate CATE = mu1(X) − mu0(X) for each row.

        Returns
        -------
        np.ndarray
            1-D array of uplift scores (can be negative for sleeping dogs).
        """
        if self.mu1_ is None or self.mu0_ is None:
            raise RuntimeError("Call fit() first.")
        feat = preprocess(X)[self.feature_cols]
        return self.mu1_.predict_proba(feat)[:, 1] - self.mu0_.predict_proba(feat)[:, 1]

    def predict_potential_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu0, mu1) for each row."""
        feat = preprocess(X)[self.feature_cols]
        mu0 = self.mu0_.predict_proba(feat)[:, 1]
        mu1 = self.mu1_.predict_proba(feat)[:, 1]
        return mu0, mu1

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "TLearner":
        with Path(path).open("rb") as f:
            return pickle.load(f)


# ── S-learner ─────────────────────────────────────────────────────────────────

class SLearner:
    """Single-model uplift estimator with treatment as a feature.

    Trains one LightGBM on all rows with treatment (W) and treatment×feature
    interaction terms as additional features.

    CATE(X) = f(X, W=1, X*1) − f(X, W=0, X*0).

    Pros over T-learner:
    - Uses all data — lower variance in small datasets.
    - Can regularise the treatment effect via shrinkage.

    Cons:
    - Treatment effect can be overwhelmed by main effects (under-shrinkage).
    - Interactions may not be expressive enough if non-linearity is large.

    Parameters
    ----------
    lgbm_params:
        LightGBM hyperparameters.
    feature_cols:
        Base feature columns (treatment + interactions added internally).
    """

    def __init__(
        self,
        lgbm_params: Optional[dict[str, Any]] = None,
        feature_cols: list[str] = FEATURE_COLS,
    ) -> None:
        self.lgbm_params = lgbm_params or _DEFAULT_PARAMS.copy()
        self.feature_cols = feature_cols
        self.model_: Any = None
        self._all_feat: list[str] = []

    @property
    def name(self) -> str:
        return "S-learner"

    def fit(
        self,
        train: pd.DataFrame,
        treatment_col: str = TREATMENT_COL,
        outcome_col: str = OUTCOME_COL,
    ) -> "SLearner":
        """Train on all rows with treatment as a feature.

        Returns
        -------
        self
        """
        feat_df = preprocess(train)
        feat_df = add_treatment_interactions(feat_df, self.feature_cols, treatment_col)
        self._all_feat = get_feature_cols(feat_df, include_interactions=True)
        X = feat_df[self._all_feat]
        y = train[outcome_col].values

        self.model_ = _make_lgbm(**self.lgbm_params)
        self.model_.fit(X, y)
        logger.info("SLearner trained on %d rows, %d features.", len(train), len(self._all_feat))
        return self

    def predict_cate(
        self,
        X: pd.DataFrame,
        treatment_col: str = TREATMENT_COL,
    ) -> np.ndarray:
        """Estimate CATE = f(X, W=1) − f(X, W=0).

        Returns
        -------
        np.ndarray
            1-D array of uplift scores.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")

        feat_df = preprocess(X)

        # Score with W=1
        df1 = feat_df.copy()
        df1[treatment_col] = 1
        df1 = add_treatment_interactions(df1, self.feature_cols, treatment_col)
        X1 = df1.reindex(columns=self._all_feat, fill_value=0)

        # Score with W=0
        df0 = feat_df.copy()
        df0[treatment_col] = 0
        df0 = add_treatment_interactions(df0, self.feature_cols, treatment_col)
        X0 = df0.reindex(columns=self._all_feat, fill_value=0)

        return (
            self.model_.predict_proba(X1)[:, 1]
            - self.model_.predict_proba(X0)[:, 1]
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "SLearner":
        with Path(path).open("rb") as f:
            return pickle.load(f)
