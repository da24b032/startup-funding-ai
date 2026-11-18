import pytest
import pandas as pd
from src.data.generate import generate_churn_dataset
from src.features.engineering import prepare_features


@pytest.fixture(scope="session")
def churn_df() -> pd.DataFrame:
    return generate_churn_dataset(n_samples=200, random_state=42)


@pytest.fixture(scope="session")
def feature_matrix(churn_df):
    X, y = prepare_features(churn_df)
    return X, y
