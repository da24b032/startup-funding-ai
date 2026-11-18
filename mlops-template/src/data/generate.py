"""
Synthetic churn dataset generator.
Produces a realistic SaaS/Telecom churn dataset so the pipeline runs
without needing proprietary data. Swap this with a real CSV loader in production.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer churn data with realistic correlations:
    - High monthly charges + month-to-month contract → higher churn probability.
    - Long tenure + annual contract → lower churn probability.
    - Multiple services → lower churn (stickiness effect).
    """
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(1, 73, size=n_samples).astype(float)
    monthly_charges = rng.uniform(20, 120, size=n_samples)
    num_services = rng.integers(1, 7, size=n_samples).astype(float)
    contract_type = rng.choice([0, 1, 2], size=n_samples, p=[0.45, 0.25, 0.30])
    # 0=month-to-month, 1=one-year, 2=two-year
    paperless_billing = rng.integers(0, 2, size=n_samples).astype(float)
    tech_support = rng.integers(0, 2, size=n_samples).astype(float)
    online_security = rng.integers(0, 2, size=n_samples).astype(float)
    payment_method = rng.integers(0, 4, size=n_samples).astype(float)
    # 0=electronic, 1=mailed, 2=bank transfer, 3=credit card
    total_charges = tenure * monthly_charges * rng.uniform(0.9, 1.1, size=n_samples)
    avg_monthly_gb = rng.uniform(1, 50, size=n_samples)

    # Churn probability model with realistic signal
    log_odds = (
        -1.7
        + 0.03 * monthly_charges
        - 0.04 * tenure
        - 0.3 * num_services
        - 0.8 * (contract_type == 1)
        - 1.5 * (contract_type == 2)
        + 0.4 * paperless_billing
        - 0.3 * tech_support
        - 0.2 * online_security
        + rng.normal(0, 0.3, size=n_samples)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = (rng.uniform(size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n_samples)],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "num_services": num_services,
        "contract_type": contract_type,        # 0=monthly, 1=annual, 2=biennial
        "paperless_billing": paperless_billing,
        "tech_support": tech_support,
        "online_security": online_security,
        "payment_method": payment_method,
        "avg_monthly_gb": avg_monthly_gb.round(2),
        "churn": churn,
    })

    churn_rate = churn.mean()
    logger.info("Generated %d samples | churn rate=%.1f%%", n_samples, churn_rate * 100)
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Dataset saved to %s", path)


def load_or_generate(path: str, n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    if Path(path).exists():
        df = pd.read_csv(path)
        logger.info("Loaded dataset from %s (%d rows)", path, len(df))
        return df
    df = generate_churn_dataset(n_samples=n_samples, random_state=random_state)
    save_dataset(df, path)
    return df
