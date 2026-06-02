"""Tests for commerce_ml.data.loaders.

Tests cover: path resolution, error messages on missing data, and the
schema/shape guarantees of loader return values (when data is present).

Missing data tests run unconditionally — they verify the error messages
are clear (a common source of confusion in a portfolio project where
the reviewer needs to set up data themselves).
"""

from pathlib import Path

import pytest

from commerce_ml.data.loaders import DATA_DIR, get_data_dir


class TestGetDataDir:
    def test_returns_path(self) -> None:
        d = get_data_dir()
        assert isinstance(d, Path)

    def test_directory_exists_after_call(self) -> None:
        d = get_data_dir()
        assert d.exists()
        assert d.is_dir()

    def test_data_dir_constant_is_path(self) -> None:
        assert isinstance(DATA_DIR, Path)


class TestM5LoaderMissingData:
    def test_load_sales_raises_file_not_found(self, tmp_path: Path) -> None:
        from commerce_ml.data.loaders import load_m5_sales
        with pytest.raises(FileNotFoundError, match="make data-m5"):
            load_m5_sales(data_dir=tmp_path)

    def test_load_calendar_raises_file_not_found(self, tmp_path: Path) -> None:
        from commerce_ml.data.loaders import load_m5_calendar
        with pytest.raises(FileNotFoundError, match="make data-m5"):
            load_m5_calendar(data_dir=tmp_path)

    def test_load_prices_raises_file_not_found(self, tmp_path: Path) -> None:
        from commerce_ml.data.loaders import load_m5_prices
        with pytest.raises(FileNotFoundError, match="make data-m5"):
            load_m5_prices(data_dir=tmp_path)


class TestCriteoLoaderMissingData:
    def test_load_criteo_raises_file_not_found(self, tmp_path: Path) -> None:
        from commerce_ml.data.loaders import load_criteo
        with pytest.raises(FileNotFoundError, match="make data-criteo"):
            load_criteo(data_dir=tmp_path)


class TestSyntheticDataSchema:
    """Verify the synthetic data generator returns expected table schemas."""

    def test_returns_three_dataframes(self) -> None:
        from commerce_ml.data.synthetic import generate_returns_dataset
        customers, orders, returns = generate_returns_dataset(n_customers=100)
        import pandas as pd
        assert isinstance(customers, pd.DataFrame)
        assert isinstance(orders, pd.DataFrame)
        assert isinstance(returns, pd.DataFrame)

    def test_customers_has_required_columns(self) -> None:
        from commerce_ml.data.synthetic import generate_returns_dataset
        customers, _, _ = generate_returns_dataset(n_customers=100)
        required = [
            "customer_id", "address_id", "payment_hash",
            "account_age_days", "total_orders", "total_returns",
            "lifetime_return_rate", "archetype",
        ]
        for col in required:
            assert col in customers.columns, f"Missing column: {col}"

    def test_orders_has_required_columns(self) -> None:
        from commerce_ml.data.synthetic import generate_returns_dataset
        _, orders, _ = generate_returns_dataset(n_customers=100)
        required = [
            "order_id", "customer_id", "order_date",
            "category", "item_price", "quantity", "channel", "was_returned",
        ]
        for col in required:
            assert col in orders.columns, f"Missing column: {col}"

    def test_returns_has_required_columns(self) -> None:
        from commerce_ml.data.synthetic import generate_returns_dataset
        _, _, returns = generate_returns_dataset(n_customers=100)
        required = [
            "return_id", "order_id", "customer_id", "return_date",
            "days_to_return", "reason_code", "condition",
            "exchange_requested", "is_fraud",
        ]
        for col in required:
            assert col in returns.columns, f"Missing column: {col}"
