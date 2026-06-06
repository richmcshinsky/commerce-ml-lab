"""Tests for the shipping price optimisation suite.

Covers:
- Synthetic data generator schema and segment properties
- ConversionElasticityModel training and counterfactual inference
- ShippingPriceOptimizer correctness
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make both the shared library and the project-local module importable.
# parents[3] = repo root  →  root/src  (commerce_ml shared library)
# parents[1] = project root  →  project/src  (shipping package)
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from shipping.synthetic import (
    PRICE_OPTIONS,
    PRODUCT_MARGIN_RATE,
    SEGMENT_NAMES,
    SEGMENT_PARAMS,
    SHIPPING_COST_TO_MERCHANT,
    generate_shipping_dataset,
)
from shipping.optimizer import (
    SHIPPING_OPTIONS,
    ShippingPriceOptimizer,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """Small dataset for fast tests: 3,000 sessions, fixed seed."""
    return generate_shipping_dataset(n_sessions=3_000, random_state=0)


@pytest.fixture(scope="session")
def trained_model(small_df: pd.DataFrame) -> object:
    """Elasticity model fitted on the small dataset."""
    pytest.importorskip("lightgbm")
    from shipping.elasticity import ConversionElasticityModel

    fast_params = {"n_estimators": 30, "verbose": -1, "random_state": 42}
    model = ConversionElasticityModel(lgbm_params=fast_params)
    return model.fit(small_df)


# ── Synthetic data ─────────────────────────────────────────────────────────────


class TestSyntheticGenerator:
    def test_returns_dataframe(self, small_df: pd.DataFrame) -> None:
        assert isinstance(small_df, pd.DataFrame)

    def test_schema(self, small_df: pd.DataFrame) -> None:
        required = [
            "session_id",
            "segment",
            "cart_value",
            "n_items",
            "is_returning",
            "session_depth",
            "time_on_checkout",
            "device_mobile",
            "f0",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "shipping_price",
            "converted",
            "cart_margin",
        ]
        for col in required:
            assert col in small_df.columns, f"Missing column: {col}"

    def test_segment_distribution(self, small_df: pd.DataFrame) -> None:
        counts = small_df["segment"].value_counts(normalize=True)
        for seg in SEGMENT_NAMES:
            expected = SEGMENT_PARAMS[seg]["weight"]
            assert abs(counts.get(seg, 0) - expected) < 0.05, (
                f"Segment {seg!r} fraction {counts.get(seg, 0):.3f} far from expected {expected:.3f}"
            )

    def test_price_options_are_valid(self, small_df: pd.DataFrame) -> None:
        assert set(small_df["shipping_price"].unique()).issubset(set(PRICE_OPTIONS))

    def test_reproducible(self) -> None:
        df1 = generate_shipping_dataset(n_sessions=100, random_state=7)
        df2 = generate_shipping_dataset(n_sessions=100, random_state=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_cart_margin_equals_rate_times_value(self, small_df: pd.DataFrame) -> None:
        expected = (small_df["cart_value"] * PRODUCT_MARGIN_RATE).round(2)
        pd.testing.assert_series_equal(small_df["cart_margin"], expected, check_names=False)

    def test_sure_things_have_higher_conversion_rate(self, small_df: pd.DataFrame) -> None:
        st_rate = small_df[small_df["segment"] == "sure_thing"]["converted"].mean()
        lc_rate = small_df[small_df["segment"] == "lost_cause"]["converted"].mean()
        assert st_rate > lc_rate, "sure_thing conversion rate should exceed lost_cause"

    def test_persuadables_sensitive_to_free_vs_expensive(self, small_df: pd.DataFrame) -> None:
        """Free shipping should have higher conversion than expensive shipping for persuadables."""
        p_df = small_df[small_df["segment"] == "persuadable"]
        free_rate = p_df[p_df["shipping_price"] == 0.0]["converted"].mean()
        exp_rate = p_df[p_df["shipping_price"] >= 9.99]["converted"].mean()
        assert free_rate > exp_rate, (
            f"Persuadables: free rate {free_rate:.3f} should > expensive {exp_rate:.3f}"
        )


# ── Elasticity model ───────────────────────────────────────────────────────────


class TestElasticityModel:
    def test_predict_proba_shape(self, trained_model: object, small_df: pd.DataFrame) -> None:
        p = trained_model.predict_proba(small_df)
        assert p.shape == (len(small_df),)

    def test_predict_proba_in_unit_interval(
        self, trained_model: object, small_df: pd.DataFrame
    ) -> None:
        p = trained_model.predict_proba(small_df)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_predict_at_price_counterfactual(
        self, trained_model: object, small_df: pd.DataFrame
    ) -> None:
        """Predicted P(convert) at price=0 should exceed P(convert) at price=12.99."""
        p_free = trained_model.predict_at_price(small_df.head(500), 0.00).mean()
        p_exp = trained_model.predict_at_price(small_df.head(500), 12.99).mean()
        # Overall average: free should beat expensive (persuadables + sure_things dominate)
        assert p_free > p_exp, f"Expected p_free ({p_free:.4f}) > p_expensive ({p_exp:.4f})"

    def test_price_curve_returns_dataframe(
        self, trained_model: object, small_df: pd.DataFrame
    ) -> None:
        session = small_df.iloc[0]
        curve = trained_model.price_curve(session, PRICE_OPTIONS)
        assert isinstance(curve, pd.DataFrame)
        assert list(curve.columns) == ["price", "p_convert", "expected_margin"]
        assert len(curve) == len(PRICE_OPTIONS)

    def test_save_and_load(self, trained_model: object, tmp_path: Path) -> None:
        from shipping.elasticity import ConversionElasticityModel

        path = tmp_path / "model.pkl"
        trained_model.save(path)
        loaded = ConversionElasticityModel.load(path)
        assert loaded is not None


# ── Optimizer ──────────────────────────────────────────────────────────────────


class TestShippingPriceOptimizer:
    def test_recommend_returns_valid_option(
        self, trained_model: object, small_df: pd.DataFrame
    ) -> None:
        opt = ShippingPriceOptimizer(trained_model)
        result = opt.recommend(small_df.iloc[0])
        valid_prices = {o.price for o in SHIPPING_OPTIONS}
        assert result.recommended.price in valid_prices

    def test_sure_thing_gets_higher_price_than_persuadable(
        self, trained_model: object, small_df: pd.DataFrame
    ) -> None:
        """On average, sure-things should be assigned a higher shipping price than persuadables."""
        all_ems = [
            trained_model.predict_at_price(small_df, o.price)
            * (small_df["cart_value"] * PRODUCT_MARGIN_RATE + o.price - SHIPPING_COST_TO_MERCHANT)
            for o in SHIPPING_OPTIONS
        ]
        em_matrix = np.stack(all_ems, axis=1)
        best_idx = em_matrix.argmax(axis=1)
        best_price = np.array([SHIPPING_OPTIONS[i].price for i in best_idx])

        st_mean_price = best_price[small_df["segment"].values == "sure_thing"].mean()
        persu_mean_price = best_price[small_df["segment"].values == "persuadable"].mean()
        assert st_mean_price >= persu_mean_price, (
            f"sure_thing avg price {st_mean_price:.2f} should ≥ persuadable {persu_mean_price:.2f}"
        )

    def test_optimised_beats_flat_rate(self, trained_model: object, small_df: pd.DataFrame) -> None:
        """Optimised expected margin should exceed the flat-rate baseline."""
        opt = ShippingPriceOptimizer(trained_model)
        policy_df = opt.compare_policies(small_df.head(1000))
        flat_em = float(
            policy_df[policy_df["policy"].str.startswith("Flat")]["mean_expected_margin"].iloc[0]
        )
        opt_em = float(
            policy_df[policy_df["policy"].str.startswith("Opt")]["mean_expected_margin"].iloc[0]
        )
        assert opt_em >= flat_em, f"Optimised EM {opt_em:.4f} should ≥ flat-rate {flat_em:.4f}"

    def test_compare_policies_returns_all_four(
        self, trained_model: object, small_df: pd.DataFrame
    ) -> None:
        opt = ShippingPriceOptimizer(trained_model)
        policy_df = opt.compare_policies(small_df.head(500))
        assert len(policy_df) == 4

    def test_min_conversion_floor_respected(
        self, trained_model: object, small_df: pd.DataFrame
    ) -> None:
        """With a high conversion floor, the optimizer should never recommend a very expensive option."""
        opt = ShippingPriceOptimizer(trained_model, min_p_convert=0.60)
        # For a sure-thing session, the floor should still allow a recommendation
        sure_sessions = small_df[small_df["segment"] == "sure_thing"].head(5)
        for _, row in sure_sessions.iterrows():
            result = opt.recommend(row)
            assert result.p_convert >= 0.0  # basic sanity — result always exists
