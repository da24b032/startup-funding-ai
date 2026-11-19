"""API request and response models.

Validation bounds come from ``src.data.schema.VALUE_RANGES`` so the API, the
data pipeline and the docs all agree on what a plausible startup looks like.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.schema import VALUE_RANGES

Flag = Literal[0, 1]


class StartupProfile(BaseModel):
    """One startup, described with raw, human-readable inputs (18 fields)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "state_code": "CA",
                "category_code": "software",
                "age_first_funding_year": 1.5,
                "age_last_funding_year": 3.5,
                "funding_rounds": 3,
                "funding_total_usd": 12000000,
                "avg_participants": 3.0,
                "has_VC": 1,
                "has_angel": 0,
                "has_roundA": 1,
                "has_roundB": 1,
                "has_roundC": 0,
                "has_roundD": 0,
                "is_top500": 1,
                "relationships": 8,
                "milestones": 2,
                "age_first_milestone_year": 2.0,
                "age_last_milestone_year": 4.0,
            }
        },
    )

    state_code: str = Field(..., min_length=2, max_length=2, description="US state code")
    category_code: str = Field(..., min_length=1, max_length=40, description="Industry")

    age_first_funding_year: float = Field(..., ge=VALUE_RANGES["age_first_funding_year"][0], le=VALUE_RANGES["age_first_funding_year"][1])
    age_last_funding_year: float = Field(..., ge=VALUE_RANGES["age_last_funding_year"][0], le=VALUE_RANGES["age_last_funding_year"][1])
    funding_rounds: int = Field(..., ge=0, le=VALUE_RANGES["funding_rounds"][1])
    funding_total_usd: float = Field(..., ge=0, le=VALUE_RANGES["funding_total_usd"][1])
    avg_participants: float = Field(..., ge=0, le=VALUE_RANGES["avg_participants"][1])

    has_VC: Flag
    has_angel: Flag
    has_roundA: Flag
    has_roundB: Flag
    has_roundC: Flag
    has_roundD: Flag
    is_top500: Flag

    relationships: int = Field(..., ge=0, le=VALUE_RANGES["relationships"][1])
    milestones: int = Field(..., ge=0, le=VALUE_RANGES["milestones"][1])
    age_first_milestone_year: float | None = Field(
        None, ge=VALUE_RANGES["age_first_milestone_year"][0], le=VALUE_RANGES["age_first_milestone_year"][1]
    )
    age_last_milestone_year: float | None = Field(
        None, ge=VALUE_RANGES["age_last_milestone_year"][0], le=VALUE_RANGES["age_last_milestone_year"][1]
    )

    @field_validator("state_code")
    @classmethod
    def _upper_state(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("state_code must be two letters, e.g. 'CA'")
        return value.upper()

    @field_validator("category_code")
    @classmethod
    def _clean_category(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")

    @model_validator(mode="after")
    def _cross_field_checks(self) -> StartupProfile:
        if self.age_last_funding_year < self.age_first_funding_year:
            raise ValueError(
                "age_last_funding_year cannot be earlier than age_first_funding_year"
            )
        if (
            self.age_first_milestone_year is not None
            and self.age_last_milestone_year is not None
            and self.age_last_milestone_year < self.age_first_milestone_year
        ):
            raise ValueError(
                "age_last_milestone_year cannot be earlier than age_first_milestone_year"
            )
        if self.milestones == 0 and (
            self.age_first_milestone_year is not None
            or self.age_last_milestone_year is not None
        ):
            raise ValueError("milestone ages must be empty when milestones is 0")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[StartupProfile] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    # Fields named model_* are intentional; silence pydantic's protected-namespace warning.
    model_config = ConfigDict(protected_namespaces=())

    prediction: Flag
    outcome: Literal["Acquired", "Closed"]
    acquired_probability: float
    closed_probability: float
    threshold: float
    model_name: str
    model_run_id: str
    latency_ms: float


class Factor(BaseModel):
    field: str
    label: str
    value: Any = None
    impact: float
    direction: Literal["raises", "lowers", "neutral"]


class ExplainedPrediction(PredictionResponse):
    top_factors: list[Factor]
    explanation_method: str


class BatchResponse(BaseModel):
    total: int
    predicted_acquired: int
    predicted_closed: int
    acquired_rate: float
    predictions: list[PredictionResponse]


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    request_id: str | None = None
