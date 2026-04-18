"""
Real-Time Anomaly Detection API
FastAPI server that accepts transaction data and returns fraud predictions.
Designed for sub-10ms inference latency.
"""

import os
import time
import logging
import numpy as np
import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd

# Add src to path when running from Docker
import sys
sys.path.append(os.path.dirname(__file__))

from features import prepare_dataframe, load_preprocessor
from train import TransactionAutoencoder, compute_ensemble_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "../models")

# ──────────────────────────────────────────────
# Load Models at Startup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="Real-time fraud detection using Isolation Forest + Autoencoder ensemble",
    version="1.0.0",
)

_preprocessor = None
_if_model = None
_ae_model = None
_threshold = None
_input_dim = None


@app.on_event("startup")
def load_models():
    global _preprocessor, _if_model, _ae_model, _threshold, _input_dim

    try:
        _preprocessor = load_preprocessor(os.path.join(MODEL_DIR, "preprocessor.pkl"))
        _if_model = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

        meta = joblib.load(os.path.join(MODEL_DIR, "threshold.pkl"))
        _threshold = meta["threshold"]
        _input_dim = meta["input_dim"]

        _ae_model = TransactionAutoencoder(input_dim=_input_dim)
        _ae_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "autoencoder.pt"), map_location="cpu")
        )
        _ae_model.eval()

        logger.info(f"Models loaded | threshold={_threshold:.4f}")
    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}. Run train.py first.")


# ──────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id: str
    timestamp: str
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    transactions_last_1h: int = Field(..., ge=0)
    transactions_last_24h: int = Field(..., ge=0)
    avg_amount_last_30d: float = Field(..., gt=0)
    distance_from_home_km: float = Field(..., ge=0)
    is_foreign_transaction: int = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn-001",
                "timestamp": "2024-01-15T14:30:00",
                "amount": 150.0,
                "merchant_category": "grocery",
                "hour_of_day": 14,
                "day_of_week": 1,
                "transactions_last_1h": 1,
                "transactions_last_24h": 5,
                "avg_amount_last_30d": 120.0,
                "distance_from_home_km": 3.5,
                "is_foreign_transaction": 0,
            }
        }


class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    anomaly_score: float
    confidence: str  # "low" | "medium" | "high"
    latency_ms: float


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    models_loaded = all([_preprocessor, _if_model, _ae_model, _threshold is not None])
    return {"status": "ok" if models_loaded else "models_not_loaded", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest):
    if any(m is None for m in [_preprocessor, _if_model, _ae_model]):
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    start = time.perf_counter()

    # Convert request to DataFrame with all derived features
    record = txn.dict()
    df = prepare_dataframe([record])

    # Preprocess
    X = _preprocessor.transform(df)

    # Score
    score = compute_ensemble_scores(X, _if_model, _ae_model)[0]
    is_fraud = bool(score >= _threshold)

    # Confidence tier based on distance from threshold
    gap = abs(score - _threshold)
    if gap < 0.05:
        confidence = "low"
    elif gap < 0.15:
        confidence = "medium"
    else:
        confidence = "high"

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"txn={txn.transaction_id} | score={score:.4f} | "
        f"fraud={is_fraud} | latency={latency_ms:.2f}ms"
    )

    return PredictionResponse(
        transaction_id=txn.transaction_id,
        is_fraud=is_fraud,
        anomaly_score=round(float(score), 4),
        confidence=confidence,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionRequest]):
    """Process multiple transactions in a single call — useful for replay/backfill."""
    if len(transactions) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500 transactions.")

    results = [predict(txn) for txn in transactions]
    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
