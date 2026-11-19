"""Feature engineering.

``StartupFeatureBuilder`` is a scikit-learn transformer that turns a *raw*
startup profile (the same 18 fields the API accepts) into model-ready columns.
It is the first step of the saved model pipeline, so training and serving
share exactly one implementation and cannot drift apart (no train/serve skew).

Rules that are learned from the training data (rare-category grouping) happen
in ``fit``; everything else is a deterministic function of the input row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.schema import (
    BINARY_COLUMNS,
    NUMERIC_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TOP_STATES,
)

# Columns produced by the builder, grouped by how they are encoded next.
ENGINEERED_NUMERIC = [
    "funding_efficiency",
    "investor_strength",
    "milestone_speed",
    "network_funding_ratio",
    "log_funding",
    "funding_span_years",
    "rounds_per_year",
]
ENGINEERED_BINARY = ["is_top_state", "funded_before_founding", "has_milestone"]
CATEGORICAL_OUT = ["state_group", "category_group"]

MODEL_NUMERIC = NUMERIC_COLUMNS + ENGINEERED_NUMERIC
MODEL_BINARY = BINARY_COLUMNS + ENGINEERED_BINARY
MODEL_CATEGORICAL = CATEGORICAL_OUT

EXPLICIT_STATES = ("CA", "NY", "MA", "TX")
RARE_CATEGORY_MIN_COUNT = 20


class StartupFeatureBuilder(BaseEstimator, TransformerMixin):
    """Clean a raw profile and derive features.

    Parameters
    ----------
    rare_min_count:
        Categories seen fewer than this many times during ``fit`` are merged
        into ``"other"`` so the one-hot encoder stays small and stable.
    """

    def __init__(self, rare_min_count: int = RARE_CATEGORY_MIN_COUNT):
        self.rare_min_count = rare_min_count

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y=None) -> StartupFeatureBuilder:
        self._check_columns(X)
        counts = X["category_code"].astype(str).value_counts()
        self.known_categories_ = sorted(
            counts[counts >= self.rare_min_count].index.tolist()
        )
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        df = X[RAW_FEATURE_COLUMNS].copy()

        # --- cleaning -------------------------------------------------------
        # Funding dated before founding is kept as information, then clipped.
        df["funded_before_founding"] = (df["age_first_funding_year"] < 0).astype(int)
        for col in (
            "age_first_funding_year",
            "age_last_funding_year",
            "age_first_milestone_year",
            "age_last_milestone_year",
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["has_milestone"] = (df["milestones"] > 0).astype(int)
        for col in ("age_first_funding_year", "age_last_funding_year"):
            df[col] = df[col].clip(lower=0)
        # Missing milestone age means "no milestone"; 0 + has_milestone flag.
        for col in ("age_first_milestone_year", "age_last_milestone_year"):
            df[col] = df[col].fillna(0).clip(lower=0)

        # --- engineered numeric features -----------------------------------
        funding = df["funding_total_usd"].clip(lower=0)
        df["log_funding"] = np.log1p(funding)
        df["funding_efficiency"] = np.log1p(funding / (df["funding_rounds"] + 1))
        df["investor_strength"] = (
            3 * df["has_VC"]
            + 2 * df["has_angel"]
            + df["has_roundA"]
            + df["has_roundB"]
            + df["has_roundC"]
            + df["has_roundD"]
        )
        df["milestone_speed"] = df["milestones"] / (df["age_first_milestone_year"] + 1)
        df["network_funding_ratio"] = df["relationships"] / (df["log_funding"] + 1)
        df["funding_span_years"] = (
            df["age_last_funding_year"] - df["age_first_funding_year"]
        ).clip(lower=0)
        df["rounds_per_year"] = df["funding_rounds"] / (df["funding_span_years"] + 1)

        # --- categorical grouping ------------------------------------------
        state = df["state_code"].astype(str).str.upper()
        df["is_top_state"] = state.isin(TOP_STATES).astype(int)
        df["state_group"] = np.where(state.isin(EXPLICIT_STATES), state, "other")
        category = df["category_code"].astype(str)
        known = set(getattr(self, "known_categories_", []))
        df["category_group"] = np.where(category.isin(known), category, "other")

        return df[MODEL_NUMERIC + MODEL_BINARY + MODEL_CATEGORICAL]

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")

    def get_feature_names_out(self, input_features=None):
        return np.array(MODEL_NUMERIC + MODEL_BINARY + MODEL_CATEGORICAL)
