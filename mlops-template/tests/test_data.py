"""Tests for data generation and validation."""

import pytest
from pydantic import ValidationError
from src.data.generate import generate_churn_dataset
from src.data.validate import CustomerFeatures


def test_generate_churn_dataset_shape():
    df = generate_churn_dataset(n_samples=100, random_state=0)
    assert len(df) == 100
    assert "churn" in df.columns


def test_churn_rate_realistic():
    df = generate_churn_dataset(n_samples=1000, random_state=42)
    rate = df["churn"].mean()
    assert 0.10 < rate < 0.50, f"Churn rate {rate:.2%} looks unrealistic"


def test_no_nulls():
    df = generate_churn_dataset(n_samples=200, random_state=42)
    assert df.isnull().sum().sum() == 0


def test_customer_features_valid():
    feat = CustomerFeatures(
        tenure_months=24, monthly_charges=50.0, total_charges=1200.0,
        num_services=3, contract_type=1, paperless_billing=1,
        tech_support=0, online_security=1, payment_method=2, avg_monthly_gb=10.0,
    )
    assert feat.tenure_months == 24


def test_customer_features_invalid_total_charges():
    with pytest.raises(ValidationError):
        CustomerFeatures(
            tenure_months=1, monthly_charges=100.0, total_charges=50.0,  # total < monthly
            num_services=1, contract_type=0, paperless_billing=0,
            tech_support=0, online_security=0, payment_method=0, avg_monthly_gb=5.0,
        )


def test_customer_features_out_of_range():
    with pytest.raises(ValidationError):
        CustomerFeatures(
            tenure_months=200,  # > 120
            monthly_charges=50.0, total_charges=10000.0,
            num_services=3, contract_type=1, paperless_billing=1,
            tech_support=0, online_security=1, payment_method=2, avg_monthly_gb=10.0,
        )
