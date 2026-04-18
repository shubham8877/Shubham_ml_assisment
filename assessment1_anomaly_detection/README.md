# Assessment 1: Real-Time Anomaly Detection System

A high-throughput fraud detection pipeline using an ensemble of **Isolation Forest** and a **Shallow Autoencoder**, designed for sub-10ms inference latency.

## Architecture

```
Transaction Stream → Feature Engineering → IF + Autoencoder Ensemble → Anomaly Score → Threshold → Fraud/Legit
```

**Why this ensemble?**
- Isolation Forest: Fast tree-based anomaly detection, no labeled data needed
- Autoencoder: Learns a compressed representation of normal transactions; fraudulent ones produce high reconstruction error
- Combining both reduces false positives from either model alone

## Project Structure

```
assessment1_anomaly_detection/
├── src/
│   ├── stream_simulator.py  # Synthetic transaction stream generator
│   ├── features.py          # Feature engineering (shared by train + API)
│   ├── train.py             # Model training pipeline
│   └── app.py               # FastAPI inference server
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/                  # Auto-created after training
├── reports/
│   └── evaluate.py          # Performance benchmarking script
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
cd src
python train.py
```
This generates synthetic data, trains both models, tunes the decision threshold, and saves artifacts to `../models/`.

### 3. Start the API server
```bash
python app.py
# OR
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Test a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "test-001",
    "timestamp": "2024-01-15T14:30:00",
    "amount": 4500.0,
    "merchant_category": "online_retail",
    "hour_of_day": 2,
    "day_of_week": 1,
    "transactions_last_1h": 8,
    "transactions_last_24h": 20,
    "avg_amount_last_30d": 100.0,
    "distance_from_home_km": 8000.0,
    "is_foreign_transaction": 1
  }'
```

### 5. Docker deployment
```bash
cd docker
docker-compose up --build
```

### 6. Run performance evaluation
```bash
# (with API running on port 8000)
cd reports
python evaluate.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health check |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score up to 500 transactions |

## Design Decisions

- **Training on legitimate data only**: Since fraud labels are unavailable in real-time streams, the models learn the distribution of normal transactions. Anything far from that distribution is flagged.
- **Threshold tuning**: The decision boundary is chosen to maximize F1-score on a validation set, not accuracy. This is critical for imbalanced datasets.
- **Derived features**: `amount_to_avg_ratio`, `velocity_ratio`, and `foreign_distance_interaction` encode domain knowledge that raw features miss.
