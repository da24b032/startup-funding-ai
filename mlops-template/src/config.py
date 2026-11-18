"""Centralised configuration loaded from .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "churn-prediction"

    # Model registry
    model_dir: str = "./models"
    model_name: str = "churn_classifier"

    # Training hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    random_state: int = 42
    test_size: float = 0.2

    # Data
    data_path: str = "./data/raw/churn_data.csv"
    n_synthetic_samples: int = 5000

    # Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Monitoring
    prometheus_port: int = 8001
    drift_threshold: float = 0.05

    log_level: str = "INFO"


settings = Settings()
