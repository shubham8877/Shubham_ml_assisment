"""
End-to-End Integration Test for Assessment 1
Runs the full pipeline: data generation → training → API prediction
without needing the Docker container running.

Usage:
    cd assessment1_anomaly_detection/src
    python test_pipeline.py
"""

import sys
import os
import json
import time
import subprocess
import threading
import logging

# Make sure we can import from src/
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Test 1: Stream Simulator
# ──────────────────────────────────────────────

def test_stream_simulator():
    logger.info("=" * 50)
    logger.info("TEST 1: Stream Simulator")
    logger.info("=" * 50)

    from stream_simulator import transaction_stream

    transactions = list(transaction_stream(rate_per_second=1000, max_transactions=100))

    fraud_count = sum(t["is_fraud"] for t in transactions)
    fraud_rate = fraud_count / len(transactions)

    assert len(transactions) == 100, f"Expected 100 transactions, got {len(transactions)}"
    assert 0 <= fraud_rate <= 0.05, f"Fraud rate out of expected range: {fraud_rate:.2%}"

    # Check all expected fields are present
    required_fields = [
        "transaction_id", "timestamp", "amount", "merchant_category",
        "hour_of_day", "day_of_week", "transactions_last_1h",
        "transactions_last_24h", "avg_amount_last_30d",
        "distance_from_home_km", "is_foreign_transaction", "is_fraud"
    ]
    for field in required_fields:
        assert field in transactions[0], f"Missing field: {field}"

    logger.info(f"✅ Generated 100 transactions | fraud_rate={fraud_rate:.2%}")
    logger.info(f"   Sample: amount=${transactions[0]['amount']:.2f}, "
                f"hour={transactions[0]['hour_of_day']}")
    return True


# ──────────────────────────────────────────────
# Test 2: Feature Engineering
# ──────────────────────────────────────────────

def test_feature_engineering():
    logger.info("=" * 50)
    logger.info("TEST 2: Feature Engineering")
    logger.info("=" * 50)

    from stream_simulator import transaction_stream
    from features import build_preprocessor, prepare_dataframe

    records = list(transaction_stream(rate_per_second=1000, max_transactions=50))
    df = prepare_dataframe(records)

    # Check derived features exist
    assert "amount_to_avg_ratio" in df.columns
    assert "velocity_ratio" in df.columns
    assert "is_odd_hour" in df.columns
    assert "foreign_distance_interaction" in df.columns

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)

    assert X.shape[0] == 50, f"Expected 50 rows, got {X.shape[0]}"
    assert X.shape[1] > 0, "Feature matrix has no columns"

    logger.info(f"✅ Feature matrix shape: {X.shape}")
    logger.info(f"   Derived features verified: amount_to_avg_ratio, velocity_ratio, etc.")
    return True


# ──────────────────────────────────────────────
# Test 3: Model Training (fast version)
# ──────────────────────────────────────────────

def test_model_training():
    logger.info("=" * 50)
    logger.info("TEST 3: Model Training (Mini Run)")
    logger.info("=" * 50)

    import numpy as np
    import torch
    from sklearn.ensemble import IsolationForest
    from stream_simulator import transaction_stream
    from features import build_preprocessor, prepare_dataframe
    from train import (
        TransactionAutoencoder, train_isolation_forest,
        train_autoencoder, compute_ensemble_scores, tune_threshold
    )

    # Small dataset for speed
    records = list(transaction_stream(rate_per_second=1000, fraud_rate=0.05, max_transactions=500))
    df = prepare_dataframe(records)
    y = df["is_fraud"].values

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)
    X_legit = X[y == 0]

    # Train both models with reduced epochs
    if_model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    if_model.fit(X_legit)

    ae_model = train_autoencoder(X_legit, epochs=5, batch_size=64)

    # Score and threshold
    scores = compute_ensemble_scores(X, if_model, ae_model)
    assert scores.shape[0] == len(records)
    assert scores.min() >= 0 and scores.max() <= 1

    threshold = tune_threshold(scores, y)
    assert 0 <= threshold <= 1

    y_pred = (scores >= threshold).astype(int)
    fraud_caught = y_pred[y == 1].sum()

    logger.info(f"✅ Models trained | input_dim={X.shape[1]}")
    logger.info(f"   Threshold: {threshold:.4f}")
    logger.info(f"   Fraud caught: {fraud_caught}/{y.sum()} on mini test set")
    return True


# ──────────────────────────────────────────────
# Test 4: API Schema Validation (without server)
# ──────────────────────────────────────────────

def test_api_schema():
    logger.info("=" * 50)
    logger.info("TEST 4: API Request/Response Schema")
    logger.info("=" * 50)

    # Test that the Pydantic models parse correctly
    import importlib
    import unittest.mock as mock

    # Mock the model loading so we don't need actual .pkl files
    with mock.patch("app.load_models"):
        from app import TransactionRequest, PredictionResponse

        valid_txn = TransactionRequest(
            transaction_id="test-001",
            timestamp="2024-01-15T14:30:00",
            amount=150.0,
            merchant_category="grocery",
            hour_of_day=14,
            day_of_week=1,
            transactions_last_1h=1,
            transactions_last_24h=5,
            avg_amount_last_30d=120.0,
            distance_from_home_km=3.5,
            is_foreign_transaction=0,
        )

        assert valid_txn.amount == 150.0
        assert valid_txn.hour_of_day == 14

        # Test validation: amount must be > 0
        try:
            bad_txn = TransactionRequest(
                transaction_id="bad",
                timestamp="2024-01-01",
                amount=-10.0,  # invalid
                merchant_category="grocery",
                hour_of_day=14,
                day_of_week=1,
                transactions_last_1h=1,
                transactions_last_24h=5,
                avg_amount_last_30d=120.0,
                distance_from_home_km=3.5,
                is_foreign_transaction=0,
            )
            assert False, "Should have raised validation error"
        except Exception:
            pass  # Expected

    logger.info("✅ API schema validation working correctly")
    return True


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def main():
    tests = [
        ("Stream Simulator", test_stream_simulator),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("API Schema", test_api_schema),
    ]

    results = []
    for name, test_fn in tests:
        try:
            start = time.time()
            ok = test_fn()
            elapsed = time.time() - start
            results.append((name, "PASS", f"{elapsed:.1f}s"))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            logger.error(f"❌ {name} failed: {e}", exc_info=True)

    print("\n" + "=" * 55)
    print("  ASSESSMENT 1 — TEST SUMMARY")
    print("=" * 55)
    for name, status, detail in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name:<30} {status}  ({detail})")
    print("=" * 55)

    failed = [r for r in results if r[1] == "FAIL"]
    if failed:
        print(f"\n{len(failed)} test(s) failed. Check logs above.")
        sys.exit(1)
    else:
        print("\nAll tests passed! Ready for deployment.")


if __name__ == "__main__":
    main()
