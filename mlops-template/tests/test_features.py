"""Tests for feature engineering."""

import pandas as pd
from src.features.engineering import add_engineered_features, prepare_features, FEATURE_COLUMNS


def test_engineered_features_added(churn_df):
    df = add_engineered_features(churn_df)
    assert "charges_per_month_ratio" in df.columns
    assert "is_long_term_customer" in df.columns
    assert "services_per_month" in df.columns
    assert "high_value_flag" in df.columns


def test_is_long_term_customer_binary(churn_df):
    df = add_engineered_features(churn_df)
    assert set(df["is_long_term_customer"].unique()).issubset({0, 1})


def test_prepare_features_shape(feature_matrix):
    X, y = feature_matrix
    assert y is not None
    assert len(X) == len(y)
    assert X.shape[1] > 0


def test_no_nulls_in_features(feature_matrix):
    X, _ = feature_matrix
    assert X.isnull().sum().sum() == 0
