"""
Feature engineering pipeline.
Transforms raw customer data into model-ready features.
All transforms are implemented as pure functions so they're easy to test and audit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_services",
    "contract_type",
    "paperless_billing",
    "tech_support",
    "online_security",
    "payment_method",
    "avg_monthly_gb",
    # Engineered features below
    "charges_per_month_ratio",
    "is_long_term_customer",
    "services_per_month",
    "high_value_flag",
]

TARGET_COLUMN = "churn"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model signal."""
    df = df.copy()

    # Ratio of total to monthly charges — captures lifetime value signal
    df["charges_per_month_ratio"] = (
        df["total_charges"] / (df["monthly_charges"] + 1e-6)
    ).clip(upper=100)

    # Binary: has the customer been around for more than 2 years?
    df["is_long_term_customer"] = (df["tenure_months"] >= 24).astype(int)

    # Services adopted per month of tenure — engagement proxy
    df["services_per_month"] = (
        df["num_services"] / (df["tenure_months"] + 1)
    ).round(4)

    # Flag for customers spending more than $80/month (top quartile)
    df["high_value_flag"] = (df["monthly_charges"] > 80).astype(int)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Full feature prep: add engineered features, select columns, split X/y.
    Returns (X, y) where y is None if 'churn' column is absent.
    """
    df = add_engineered_features(df)

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available]

    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    logger.info("Feature matrix: %s rows × %s cols", *X.shape)
    return X, y
