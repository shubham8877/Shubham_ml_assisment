"""
Performance Report Generator
Evaluates the deployed API on a test set and writes a structured report.
Measures F1, precision, recall, ROC-AUC, and per-request latency.
"""

import time
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score
)
import json
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from stream_simulator import transaction_stream
from features import prepare_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"
N_TEST_SAMPLES = 500
FRAUD_RATE = 0.02  # slightly higher to get enough fraud samples for evaluation


def run_evaluation():
    logger.info(f"Generating {N_TEST_SAMPLES} test transactions...")
    records = list(transaction_stream(
        rate_per_second=1000,
        fraud_rate=FRAUD_RATE,
        max_transactions=N_TEST_SAMPLES,
    ))

    y_true = [r["is_fraud"] for r in records]
    y_pred = []
    scores = []
    latencies = []

    logger.info("Sending transactions to API...")
    for i, record in enumerate(records):
        # Remove label before sending to API (API doesn't use it)
        payload = {k: v for k, v in record.items() if k != "is_fraud"}

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            result = response.json()
            y_pred.append(int(result["is_fraud"]))
            scores.append(result["anomaly_score"])
            latencies.append(result["latency_ms"])
        except Exception as e:
            logger.error(f"Request {i} failed: {e}")
            y_pred.append(0)
            scores.append(0.0)
            latencies.append(0.0)

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{N_TEST_SAMPLES}")

    # ── Metrics ──────────────────────────────────────────
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=["Legitimate", "Fraud"]
    )

    latency_arr = np.array(latencies)

    # ── Print Report ──────────────────────────────────────
    print("\n" + "="*60)
    print("      ANOMALY DETECTION - PERFORMANCE REPORT")
    print("="*60)
    print(f"\nTest samples  : {N_TEST_SAMPLES}")
    print(f"Fraud samples : {sum(y_true)} ({sum(y_true)/N_TEST_SAMPLES:.1%})")
    print(f"\n{report}")
    print(f"ROC-AUC Score : {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
    print(f"\n── Latency (ms) ──────────────────────────────────")
    print(f"  Mean   : {latency_arr.mean():.2f} ms")
    print(f"  Median : {np.median(latency_arr):.2f} ms")
    print(f"  P95    : {np.percentile(latency_arr, 95):.2f} ms")
    print(f"  P99    : {np.percentile(latency_arr, 99):.2f} ms")
    print(f"  Max    : {latency_arr.max():.2f} ms")
    print("="*60)

    # ── Save JSON Report ──────────────────────────────────
    report_data = {
        "test_samples": N_TEST_SAMPLES,
        "fraud_samples": int(sum(y_true)),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "latency_ms": {
            "mean": round(float(latency_arr.mean()), 2),
            "median": round(float(np.median(latency_arr)), 2),
            "p95": round(float(np.percentile(latency_arr, 95)), 2),
            "p99": round(float(np.percentile(latency_arr, 99)), 2),
            "max": round(float(latency_arr.max()), 2),
        }
    }

    output_path = os.path.join(os.path.dirname(__file__), "performance_report.json")
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Report saved → {output_path}")


if __name__ == "__main__":
    run_evaluation()
