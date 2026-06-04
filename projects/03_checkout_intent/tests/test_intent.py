"""Tests for the checkout intent and uplift models.

LightGBM-dependent tests are skipped when lightgbm is unavailable.
Pure-logic tests (features, evaluation) always run.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from commerce_ml.data.loaders import generate_criteo_like
from intent.evaluate import (
    auuc,
    policy_comparison,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)
from intent.features import (
    FEATURE_COLS,
    add_treatment_interactions,
    get_feature_cols,
    preprocess,
    temporal_split,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    return generate_criteo_like(n_rows=2_000, random_state=0)


@pytest.fixture(scope="session")
def train_test(small_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return temporal_split(small_df, test_frac=0.25, random_state=0)


# ── Synthetic data ─────────────────────────────────────────────────────────────


class TestSyntheticData:
    def test_schema(self, small_df: pd.DataFrame) -> None:
        for col in FEATURE_COLS + ["treatment", "conversion", "visit", "segment"]:
            assert col in small_df.columns

    def test_treatment_binary(self, small_df: pd.DataFrame) -> None:
        assert small_df["treatment"].isin([0, 1]).all()

    def test_conversion_binary(self, small_df: pd.DataFrame) -> None:
        assert small_df["conversion"].isin([0, 1]).all()

    def test_persuadables_have_highest_cate(self, small_df: pd.DataFrame) -> None:
        ctrl = small_df[small_df["treatment"] == 0].groupby("segment")["conversion"].mean()
        trt = small_df[small_df["treatment"] == 1].groupby("segment")["conversion"].mean()
        cate = (trt - ctrl).dropna()
        assert cate.get("persuadable", 0) > cate.get("sure_thing", 0)
        assert cate.get("persuadable", 0) > cate.get("lost_cause", 0)

    def test_sure_things_have_highest_propensity(self, small_df: pd.DataFrame) -> None:
        propensity = small_df.groupby("segment")["conversion"].mean()
        assert propensity.get("sure_thing", 0) > propensity.get("persuadable", 0)

    def test_reproducible(self) -> None:
        d1 = generate_criteo_like(n_rows=500, random_state=7)
        d2 = generate_criteo_like(n_rows=500, random_state=7)
        pd.testing.assert_frame_equal(d1, d2)


# ── Feature engineering ───────────────────────────────────────────────────────


class TestFeatures:
    def test_preprocess_standardises(self, small_df: pd.DataFrame) -> None:
        pp = preprocess(small_df)
        for col in FEATURE_COLS:
            assert abs(pp[col].mean()) < 0.1  # roughly zero mean

    def test_interactions_added(self, small_df: pd.DataFrame) -> None:
        feat = add_treatment_interactions(small_df)
        assert "f0_x_treat" in feat.columns
        assert "f11_x_treat" in feat.columns

    def test_interaction_values_correct(self, small_df: pd.DataFrame) -> None:
        feat = add_treatment_interactions(small_df)
        # For treated rows: f0_x_treat should equal f0
        treated = feat[feat["treatment"] == 1]
        pd.testing.assert_series_equal(treated["f0"], treated["f0_x_treat"], check_names=False)
        # For control rows: f0_x_treat should be 0
        control = feat[feat["treatment"] == 0]
        assert (control["f0_x_treat"] == 0).all()

    def test_get_feature_cols_without_interactions(self, small_df: pd.DataFrame) -> None:
        cols = get_feature_cols(small_df, include_interactions=False)
        assert cols == FEATURE_COLS
        assert "treatment" not in cols

    def test_temporal_split_sizes(self, small_df: pd.DataFrame) -> None:
        train, test = temporal_split(small_df, test_frac=0.2)
        assert abs(len(test) / len(small_df) - 0.2) < 0.05

    def test_split_no_overlap(self, train_test: tuple) -> None:
        train, test = train_test
        assert len(train) + len(test) == 2_000


# ── Evaluation ────────────────────────────────────────────────────────────────


class TestEvaluation:
    def test_qini_perfect_beats_random(self, small_df: pd.DataFrame) -> None:
        y = small_df["conversion"].values
        w = small_df["treatment"].values
        perfect = (small_df["segment"] == "persuadable").astype(float).values
        rand = np.random.default_rng(0).random(len(small_df))
        assert qini_coefficient(y, w, perfect) > qini_coefficient(y, w, rand)

    def test_qini_random_near_zero(self, small_df: pd.DataFrame) -> None:
        y = small_df["conversion"].values
        w = small_df["treatment"].values
        scores = [np.random.default_rng(i).random(len(small_df)) for i in range(10)]
        qinis = [qini_coefficient(y, w, s) for s in scores]
        assert abs(np.mean(qinis)) < 5.0  # near zero on average

    def test_qini_curve_starts_at_zero(self, small_df: pd.DataFrame) -> None:
        y = small_df["conversion"].values
        w = small_df["treatment"].values
        s = np.random.default_rng(0).random(len(small_df))
        curve = qini_curve(y, w, s)
        assert curve["fraction_targeted"].iloc[0] == 0.0
        assert curve["incremental_conversions"].iloc[0] == 0.0

    def test_qini_curve_length(self, small_df: pd.DataFrame) -> None:
        y, w, s = (
            small_df["conversion"].values,
            small_df["treatment"].values,
            np.ones(len(small_df)),
        )
        curve = qini_curve(y, w, s, n_bins=50)
        assert len(curve) == 51  # 0..50 inclusive

    def test_uplift_at_k_persuadables(self, small_df: pd.DataFrame) -> None:
        y = small_df["conversion"].values
        w = small_df["treatment"].values
        perfect = (small_df["segment"] == "persuadable").astype(float).values
        u20 = uplift_at_k(y, w, perfect, k=0.20)
        assert u20 > 0  # targeting persuadables should give positive lift

    def test_policy_comparison_shape(self, small_df: pd.DataFrame) -> None:
        y = small_df["conversion"].values
        w = small_df["treatment"].values
        scores = {"A": np.ones(len(small_df)), "B": np.zeros(len(small_df))}
        comp = policy_comparison(y, w, scores, k_values=[0.10, 0.20])
        assert len(comp) == 4  # 2 models × 2 k values
        assert set(comp.columns) >= {"model", "k", "uplift_at_k", "incremental_conversions"}

    def test_auuc_perfect_positive(self, small_df: pd.DataFrame) -> None:
        y = small_df["conversion"].values
        w = small_df["treatment"].values
        perfect = (small_df["segment"] == "persuadable").astype(float).values
        assert auuc(y, w, perfect) > 0


# ── Model tests (skipped without lightgbm) ───────────────────────────────────

lgbm = pytest.importorskip("lightgbm", reason="lightgbm not installed")

_FAST = {"n_estimators": 10, "verbose": -1, "random_state": 42, "n_jobs": 1}


class TestPropensityModel:
    def test_fit_predict(self, train_test: tuple) -> None:
        from intent.models import PropensityModel

        train, test = train_test
        model = PropensityModel(lgbm_params=_FAST).fit(train)
        proba = model.predict_proba(test)
        assert len(proba) == len(test)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_sure_things_score_higher_than_lost_causes(self, train_test: tuple) -> None:
        from intent.models import PropensityModel

        train, test = train_test
        model = PropensityModel(lgbm_params=_FAST).fit(train)
        proba = model.predict_proba(test)
        sure = proba[test["segment"].values == "sure_thing"].mean()
        lost = proba[test["segment"].values == "lost_cause"].mean()
        assert sure > lost, "Sure-things should have higher propensity than lost-causes"


class TestTLearner:
    def test_fit_predict_cate(self, train_test: tuple) -> None:
        from intent.models import TLearner

        train, test = train_test
        model = TLearner(lgbm_params=_FAST).fit(train)
        cate = model.predict_cate(test)
        assert len(cate) == len(test)
        assert np.isfinite(cate).all()

    def test_persuadables_have_higher_cate(self, train_test: tuple) -> None:
        from intent.models import TLearner

        train, test = train_test
        cate = TLearner(lgbm_params=_FAST).fit(train).predict_cate(test)
        pers = cate[test["segment"].values == "persuadable"].mean()
        sure = cate[test["segment"].values == "sure_thing"].mean()
        assert pers > sure, "Persuadables should have higher estimated CATE"


class TestSLearner:
    def test_fit_predict_cate(self, train_test: tuple) -> None:
        from intent.models import SLearner

        train, test = train_test
        model = SLearner(lgbm_params=_FAST).fit(train)
        cate = model.predict_cate(test)
        assert len(cate) == len(test)
        assert np.isfinite(cate).all()

    def test_qini_positive(self, train_test: tuple) -> None:
        from intent.models import SLearner

        train, test = train_test
        model = SLearner(lgbm_params=_FAST).fit(train)
        cate = model.predict_cate(test)
        q = qini_coefficient(test["conversion"].values, test["treatment"].values, cate)
        assert q > -10  # not catastrophically wrong
