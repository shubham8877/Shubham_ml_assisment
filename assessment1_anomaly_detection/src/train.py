"""
Anomaly Detection Model Training
Uses an ensemble of Isolation Forest (unsupervised) and a shallow Autoencoder
(semi-supervised) to handle the extreme class imbalance (<1% fraud).

Training strategy:
- Train ONLY on legitimate transactions (unsupervised approach)
- Anomaly score = average of IF score + autoencoder reconstruction error
- Threshold is tuned on a labeled validation set to maximize F1
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, f1_score, precision_recall_curve, roc_auc_score
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from features import (
    build_preprocessor, prepare_dataframe,
    save_preprocessor, load_preprocessor,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from stream_simulator import transaction_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "../models"
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
IF_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
AE_MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.pt")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Autoencoder Definition
# ──────────────────────────────────────────────

class TransactionAutoencoder(nn.Module):
    """
    Shallow autoencoder. Learns to reconstruct normal transactions.
    Fraudulent ones have higher reconstruction error since they're
    out-of-distribution relative to the training data.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            errors = torch.mean((x - recon) ** 2, dim=1)
        return errors.numpy()


# ──────────────────────────────────────────────
# Data Generation
# ──────────────────────────────────────────────

def generate_dataset(n_samples: int = 10000, fraud_rate: float = 0.008) -> pd.DataFrame:
    logger.info(f"Generating {n_samples} synthetic transactions...")
    records = list(transaction_stream(
        rate_per_second=1000,
        fraud_rate=fraud_rate,
        max_transactions=n_samples,
    ))
    df = prepare_dataframe(records)
    logger.info(f"Fraud count: {df['is_fraud'].sum()} / {len(df)} "
                f"({df['is_fraud'].mean():.2%})")
    return df


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_isolation_forest(X_legit: np.ndarray) -> IsolationForest:
    logger.info("Training Isolation Forest on legitimate transactions...")
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.008,  # expected fraud rate
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_legit)
    logger.info("Isolation Forest trained.")
    return model


def train_autoencoder(
    X_legit: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> TransactionAutoencoder:
    logger.info(f"Training Autoencoder | input_dim={X_legit.shape[1]}")

    tensor = torch.FloatTensor(X_legit)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    model = TransactionAutoencoder(input_dim=X_legit.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / len(loader)
            logger.info(f"  Epoch [{epoch+1}/{epochs}] loss={avg:.6f}")

    logger.info("Autoencoder trained.")
    return model


# ──────────────────────────────────────────────
# Scoring & Threshold Tuning
# ──────────────────────────────────────────────

def compute_ensemble_scores(
    X: np.ndarray,
    if_model: IsolationForest,
    ae_model: TransactionAutoencoder,
) -> np.ndarray:
    """
    Combines IF decision scores and AE reconstruction errors.
    Both are min-max normalized before averaging so neither dominates.
    """
    # IF returns negative scores; flip so higher = more anomalous
    if_scores = -if_model.score_samples(X)

    ae_errors = ae_model.reconstruction_error(torch.FloatTensor(X))

    # Min-max normalize both to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    combined = 0.5 * normalize(if_scores) + 0.5 * normalize(ae_errors)
    return combined


def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Sweeps precision-recall curve to find the threshold that maximizes F1.
    This is more appropriate than accuracy for imbalanced datasets.
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    logger.info(f"Best threshold={best_threshold:.4f} → "
                f"P={precisions[best_idx]:.3f} R={recalls[best_idx]:.3f} "
                f"F1={f1_scores[best_idx]:.3f}")
    return float(best_threshold)


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def main():
    # 1. Generate data
    df_train = generate_dataset(n_samples=8000, fraud_rate=0.008)
    df_val = generate_dataset(n_samples=2000, fraud_rate=0.008)

    # 2. Fit preprocessor on training data only
    preprocessor = build_preprocessor()
    X_train_all = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)

    y_train = df_train["is_fraud"].values
    y_val = df_val["is_fraud"].values

    # Train models ONLY on legitimate records (unsupervised approach)
    X_train_legit = X_train_all[y_train == 0]
    logger.info(f"Training on {len(X_train_legit)} legitimate transactions.")

    # 3. Train models
    if_model = train_isolation_forest(X_train_legit)
    ae_model = train_autoencoder(X_train_legit)

    # 4. Compute ensemble scores on validation set
    val_scores = compute_ensemble_scores(X_val, if_model, ae_model)

    # 5. Tune threshold
    best_threshold = tune_threshold(val_scores, y_val)

    # 6. Evaluate
    y_pred = (val_scores >= best_threshold).astype(int)
    auc = roc_auc_score(y_val, val_scores)

    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(classification_report(y_val, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {auc:.4f}")

    # 7. Save everything
    save_preprocessor(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(if_model, IF_MODEL_PATH)
    torch.save(ae_model.state_dict(), AE_MODEL_PATH)
    joblib.dump({
        "threshold": best_threshold,
        "input_dim": X_train_all.shape[1]
    }, THRESHOLD_PATH)

    logger.info("All model artifacts saved to ./models/")
    logger.info("Done! Run 'python app.py' to start the API server.")


if __name__ == "__main__":
    main()
