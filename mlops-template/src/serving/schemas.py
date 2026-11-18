"""FastAPI request / response schemas."""

from pydantic import BaseModel
from src.data.validate import CustomerFeatures


class PredictionResponse(BaseModel):
    customer_features: dict
    churn_probability: float
    churn_prediction: int       # 0 = no churn, 1 = churn
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str
    total: int
    predicted_churn_count: int
    churn_rate: float
