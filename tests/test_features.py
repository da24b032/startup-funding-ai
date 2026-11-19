"""Unit tests for StartupFeatureBuilder (shared by training and serving)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    MODEL_BINARY,
    MODEL_CATEGORICAL,
    MODEL_NUMERIC,
    StartupFeatureBuilder,
)


def _frame(profile: dict, **overrides) -> pd.DataFrame:
    return pd.DataFrame([{**profile, **overrides}])


def test_output_columns_match_contract(profile):
    out = StartupFeatureBuilder().fit(_frame(profile)).transform(_frame(profile))
    assert list(out.columns) == MODEL_NUMERIC + MODEL_BINARY + MODEL_CATEGORICAL


def test_funding_before_founding_is_flagged_then_clipped(profile):
    frame = _frame(profile, age_first_funding_year=-2.0)
    out = StartupFeatureBuilder().fit(frame).transform(frame)
    assert out.loc[0, "funded_before_founding"] == 1
    assert out.loc[0, "age_first_funding_year"] == 0


def test_missing_milestone_ages_become_zero_with_flag(profile):
    frame = _frame(
        profile, milestones=0, age_first_milestone_year=np.nan, age_last_milestone_year=np.nan
    )
    out = StartupFeatureBuilder().fit(frame).transform(frame)
    assert out.loc[0, "has_milestone"] == 0
    assert out.loc[0, "age_first_milestone_year"] == 0


def test_investor_strength_weights(profile):
    frame = _frame(profile, has_VC=1, has_angel=1, has_roundA=1, has_roundB=0, has_roundC=0, has_roundD=0)
    out = StartupFeatureBuilder().fit(frame).transform(frame)
    assert out.loc[0, "investor_strength"] == 3 + 2 + 1


def test_rare_categories_are_grouped_as_other(profile):
    train = pd.concat([_frame(profile, category_code="web")] * 25, ignore_index=True)
    builder = StartupFeatureBuilder(rare_min_count=20).fit(train)
    out = builder.transform(_frame(profile, category_code="never_seen"))
    assert out.loc[0, "category_group"] == "other"
    assert builder.transform(_frame(profile, category_code="web")).loc[0, "category_group"] == "web"


def test_missing_input_column_raises(profile):
    frame = _frame(profile).drop(columns=["relationships"])
    with pytest.raises(ValueError, match="Missing input columns"):
        StartupFeatureBuilder().fit(frame)
