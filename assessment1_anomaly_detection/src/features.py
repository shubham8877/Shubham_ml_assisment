"""
Feature Engineering for Anomaly Detection
Transforms raw transaction fields into model-ready features.
Kept as a standalone module so it's reused identically in training and inference.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import logging
import os

logger = logging.getLogger(__name__)

# Features the model actually uses (excludes IDs, timestamps, and the label)
NUMERIC_FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "transactions_last_1h",
    "transactions_last_24h",
    "avg_amount_last_30d",
    "distance_from_home_km",
    "is_foreign_transaction",
]

CATEGORICAL_FEATURES = ["merchant_category"]

# Derived feature names after one-hot encoding merchant categories
MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "online_retail",
    "travel", "entertainment", "healthcare", "utilities"
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-driven features on top of raw fields.
    These capture behavioral patterns that raw fields miss.
    """
    df = df.copy()

    # Amount relative to the account's 30-day baseline
    # A score >> 1 means the transaction is unusually large for this customer
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_last_30d"] + 1e-9)

    # Velocity spike: transactions in last hour vs. daily average
    df["velocity_ratio"] = df["transactions_last_1h"] / (
        (df["transactions_last_24h"] / 24.0) + 1e-9
    )

    # Unusual hour flag: transactions between midnight and 5am are higher risk
    df["is_odd_hour"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)

    # High distance combined with foreign flag is a strong fraud signal
    df["foreign_distance_interaction"] = (
        df["is_foreign_transaction"] * df["distance_from_home_km"]
    )

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a sklearn ColumnTransformer that scales numerics and
    one-hot encodes the merchant category.
    """
    derived_numeric = NUMERIC_FEATURES + [
        "amount_to_avg_ratio",
        "velocity_ratio",
        "is_odd_hour",
        "foreign_distance_interaction",
    ]

    numeric_pipe = Pipeline([("scaler", StandardScaler())])

    categorical_pipe = Pipeline([
        ("ohe", OneHotEncoder(
            categories=[MERCHANT_CATEGORIES],
            handle_unknown="ignore",
            sparse_output=False,
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, derived_numeric),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_dataframe(records: list[dict]) -> pd.DataFrame:
    """Converts a list of transaction dicts into a feature DataFrame."""
    df = pd.DataFrame(records)
    df = add_derived_features(df)
    return df


def save_preprocessor(preprocessor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Preprocessor saved → {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    preprocessor = joblib.load(path)
    logger.info(f"Preprocessor loaded ← {path}")
    return preprocessor
