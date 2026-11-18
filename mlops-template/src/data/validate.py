"""
Data validation — Pydantic schemas for inference input and training row.
Catches bad data at the API boundary before it reaches the model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Single customer's feature vector for inference."""

    tenure_months: float = Field(..., ge=0, le=120, description="Months as a customer.")
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float = Field(..., ge=0)
    num_services: int = Field(..., ge=1, le=10)
    contract_type: int = Field(..., ge=0, le=2, description="0=monthly, 1=annual, 2=biennial")
    paperless_billing: int = Field(..., ge=0, le=1)
    tech_support: int = Field(..., ge=0, le=1)
    online_security: int = Field(..., ge=0, le=1)
    payment_method: int = Field(..., ge=0, le=3)
    avg_monthly_gb: float = Field(..., ge=0, le=1000)

    @field_validator("total_charges")
    @classmethod
    def total_gte_monthly(cls, v: float, info) -> float:
        monthly = info.data.get("monthly_charges", 0)
        if v < monthly:
            raise ValueError("total_charges must be >= monthly_charges")
        return v


class BatchRequest(BaseModel):
    """Batch of customers for bulk inference."""
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)
