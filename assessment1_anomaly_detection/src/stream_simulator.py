"""
Transaction Stream Simulator
Generates realistic financial transaction data with injected anomalies.
Fraud rate is kept below 1% to reflect real-world class imbalance.
"""

import random
import time
import uuid
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    transaction_id: str
    timestamp: str
    amount: float
    merchant_category: str
    hour_of_day: int
    day_of_week: int
    transactions_last_1h: int
    transactions_last_24h: int
    avg_amount_last_30d: float
    distance_from_home_km: float
    is_foreign_transaction: int
    is_fraud: int  # ground truth label (not sent to model during inference)


MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "online_retail",
    "travel", "entertainment", "healthcare", "utilities"
]

# Typical ranges for legitimate transactions
LEGIT_PROFILE = {
    "amount_range": (5, 300),
    "velocity_1h": (0, 3),
    "velocity_24h": (1, 10),
    "avg_amount_range": (50, 200),
    "distance_range": (0, 50),
}

# Anomalous patterns: high amounts, unusual hours, high velocity
FRAUD_PROFILE = {
    "amount_range": (500, 5000),
    "velocity_1h": (5, 20),
    "velocity_24h": (10, 40),
    "avg_amount_range": (50, 200),  # avg stays realistic (account is compromised)
    "distance_range": (500, 15000),
}


def _build_transaction(is_fraud: bool) -> Transaction:
    profile = FRAUD_PROFILE if is_fraud else LEGIT_PROFILE

    amount = round(random.uniform(*profile["amount_range"]), 2)
    hour = random.choice([1, 2, 3]) if is_fraud else random.randint(7, 22)

    return Transaction(
        transaction_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        amount=amount,
        merchant_category=random.choice(MERCHANT_CATEGORIES),
        hour_of_day=hour,
        day_of_week=random.randint(0, 6),
        transactions_last_1h=random.randint(*profile["velocity_1h"]),
        transactions_last_24h=random.randint(*profile["velocity_24h"]),
        avg_amount_last_30d=round(random.uniform(*profile["avg_amount_range"]), 2),
        distance_from_home_km=round(random.uniform(*profile["distance_range"]), 2),
        is_foreign_transaction=1 if is_fraud and random.random() > 0.4 else 0,
        is_fraud=int(is_fraud),
    )


def transaction_stream(
    rate_per_second: float = 10.0,
    fraud_rate: float = 0.008,
    max_transactions: int = None,
) -> Generator[dict, None, None]:
    """
    Yields transactions as dicts at a configurable rate.
    fraud_rate: probability of any given transaction being fraudulent (~0.8%)
    """
    count = 0
    delay = 1.0 / rate_per_second

    logger.info(f"Stream started | rate={rate_per_second}/s | fraud_rate={fraud_rate:.1%}")

    while True:
        if max_transactions and count >= max_transactions:
            break

        is_fraud = random.random() < fraud_rate
        txn = _build_transaction(is_fraud)
        yield asdict(txn)

        count += 1
        time.sleep(delay)


if __name__ == "__main__":
    print("Streaming 20 sample transactions (mix of legit and fraud):\n")
    for i, txn in enumerate(transaction_stream(rate_per_second=5, max_transactions=20)):
        label = "🚨 FRAUD" if txn["is_fraud"] else "✅ LEGIT"
        print(f"[{i+1:02d}] {label} | amount=${txn['amount']:>8.2f} | "
              f"hour={txn['hour_of_day']:02d}h | "
              f"dist={txn['distance_from_home_km']:.1f}km | "
              f"vel_1h={txn['transactions_last_1h']}")
