"""
Retraining pipeline.

Mirrors Day 2's train.py — same architecture, same hyperparameters, same
reproducibility — but parameterized: takes a labeled dataset as input
instead of reading a fixed CSV.

Two responsibilities, separated:
  - load_dataset(data_source, reference_date) -> labeled DataFrame
  - retrain(df, seed) -> RetrainArtifacts (model state_dict + scaler + metadata)

Phase 4 handles persisting artifacts to GCS. This module stays pure.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing._data import StandardScaler as _ScalerType
from torch.utils.data import DataLoader, TensorDataset

from features import FEATURE_COLUMNS, MIN_TENURE_DAYS, MIN_ORDERS, build_features
from labeling import compute_signals
from model import ChurnNet


# Hyperparameters — frozen to Day 2's values for fair champion/challenger comparison.
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42


# -------- artifact dataclass --------

@dataclass
class RetrainArtifacts:
    """In-memory result of a single training run. Phase 4 persists these."""
    model_state_dict: dict[str, torch.Tensor]
    scaler: _ScalerType
    metadata: dict[str, Any]
    eval_metrics: dict[str, float]


# -------- data loading --------

def load_dataset(
    data_source: Literal["real", "simulated", "union"],
    reference_date: pd.Timestamp,
    simulator_batch_id: str | None = None,
) -> pd.DataFrame:
    """
    Pull customers + orders from Cloud SQL, build features and labels,
    apply Day 2's training filter, return one row per customer.

    data_source semantics:
      - 'real'      : use the real customers + orders tables (Day 1+2 data)
      - 'simulated' : use simulated_customers + simulated_orders only
                      (optionally filtered to one batch_id)
      - 'union'     : real data UNION simulated data (production-realistic)
    """
    from sqlalchemy import text
    from database import get_engine

    if data_source not in ("real", "simulated", "union"):
        raise ValueError(f"data_source must be real|simulated|union, got {data_source!r}")

    engine = get_engine()
    with engine.connect() as conn:
        if data_source in ("real", "union"):
            real_cust = pd.read_sql(text(
                "SELECT customer_id, join_date, customer_tier FROM customers"
            ), conn)
            real_orders = pd.read_sql(text(
                "SELECT customer_id, order_date, order_amount, product_id, "
                "payment_method, order_status FROM orders"
            ), conn)
        else:
            real_cust = pd.DataFrame(columns=["customer_id", "join_date", "customer_tier"])
            real_orders = pd.DataFrame(columns=[
                "customer_id", "order_date", "order_amount",
                "product_id", "payment_method", "order_status",
            ])

        if data_source in ("simulated", "union"):
            sim_filter = ""
            params = {}
            if simulator_batch_id is not None:
                sim_filter = " WHERE batch_id = :bid"
                params = {"bid": simulator_batch_id}
            sim_cust = pd.read_sql(text(
                f"SELECT customer_id, join_date, customer_tier "
                f"FROM simulated_customers{sim_filter}"
            ), conn, params=params)
            # Simulated orders carry no order_status today (added by simulator
            # but not yet persisted by the loader); default to 'Delivered'
            # so labeling treats them as engagement orders.
            sim_orders = pd.read_sql(text(
                f"SELECT customer_id, order_date, order_amount, product_id, "
                f"payment_method, order_status, batch_id FROM simulated_orders{sim_filter}"
            ), conn, params=params)
            if not sim_orders.empty:
                sim_orders = sim_orders.drop(columns=["batch_id"])
        else:
            sim_cust = pd.DataFrame(columns=real_cust.columns)
            sim_orders = pd.DataFrame(columns=real_orders.columns)

    # Reindex simulated customer_ids to avoid clashes with real IDs in 'union' mode.
    if data_source == "union" and not sim_cust.empty:
        offset = int(real_cust["customer_id"].max()) + 1_000_000
        sim_cust = sim_cust.copy()
        sim_orders = sim_orders.copy()
        sim_cust["customer_id"] = sim_cust["customer_id"] + offset
        sim_orders["customer_id"] = sim_orders["customer_id"] + offset

    # Drop empty/all-NA frames before concat to silence pandas FutureWarning
    # about dtype handling for empty inputs.
    cust_frames = [d for d in (real_cust, sim_cust) if not d.empty]
    order_frames = [d for d in (real_orders, sim_orders) if not d.empty]
    customers = pd.concat(cust_frames, ignore_index=True) if cust_frames else pd.DataFrame()
    orders = pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame()

    if customers.empty:
        raise ValueError(f"No customers found for data_source={data_source!r}")
    if orders.empty:
        raise ValueError(f"No orders found for data_source={data_source!r}")

    feats = build_features(customers, orders, reference_date)
    labels = compute_signals(orders, reference_date)
    df = feats.merge(labels[["customer_id", "churned", "signals_fired"]], on="customer_id", how="inner")
    df = df[(df["tenure_days"] >= MIN_TENURE_DAYS) & (df["total_orders"] >= MIN_ORDERS)].reset_index(drop=True)
    return df


# -------- training --------

def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_loaders(X: np.ndarray, y: np.ndarray, seed: int) -> tuple:
    """Stratified 60/20/20 split + scaler. Identical to Day 2's make_loaders."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=seed,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    def to_loader(X_, y_, shuffle):
        ds = TensorDataset(torch.from_numpy(X_).float(), torch.from_numpy(y_).float())
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        to_loader(X_train_s, y_train, True),
        to_loader(X_val_s, y_val, False),
        to_loader(X_test_s, y_test, False),
        scaler, (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test),
    )


def _epoch_train(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb).squeeze(-1)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


def _epoch_eval(model, loader, loss_fn) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            total += loss.item() * len(xb)
    return total / len(loader.dataset)


def _evaluate_test(model, X_test_s: np.ndarray, y_test: np.ndarray, threshold: float = 0.4) -> dict[str, float]:
    """Compute final test metrics. Threshold default matches Day 2 v2 (0.40)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_s).float()).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Lightweight ROC-AUC via rank-based formula (no sklearn import needed).
    pos_scores = probs[y_test == 1]
    neg_scores = probs[y_test == 0]
    if len(pos_scores) and len(neg_scores):
        comparisons = (pos_scores[:, None] > neg_scores[None, :]).sum() + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).sum()
        roc_auc = float(comparisons) / (len(pos_scores) * len(neg_scores))
    else:
        roc_auc = float("nan")

    return {
        "threshold": threshold,
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_test": int(len(y_test)),
        "n_test_positive": int(y_test.sum()),
    }


def retrain(df: pd.DataFrame, seed: int = DEFAULT_SEED, threshold: float = 0.4) -> RetrainArtifacts:
    """
    Train a fresh ChurnNet on the given labeled DataFrame.
    df must have FEATURE_COLUMNS + 'churned' columns.
    """
    if "churned" not in df.columns:
        raise ValueError("df must include a 'churned' column")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"df missing FEATURE_COLUMNS: {missing}")

    _seed_everything(seed)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["churned"].values.astype(np.float32)
    train_loader, val_loader, _test_loader, scaler, (X_tr, y_tr), _val, (X_te, y_te) = _make_loaders(X, y, seed)

    pos_weight_value = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

    model = ChurnNet(input_dim=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        _epoch_train(model, train_loader, optimizer, loss_fn)
        val_loss = _epoch_eval(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint recorded — training never ran.")

    # Restore best weights and evaluate on test.
    model.load_state_dict(best_state)
    eval_metrics = _evaluate_test(model, X_te, y_te, threshold=threshold)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "input_dim": len(FEATURE_COLUMNS),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "pos_weight": pos_weight_value,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_churn_rate": float(y_tr.mean()),
        "seed": seed,
        "threshold": threshold,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "weight_decay": WEIGHT_DECAY,
            "hidden1": 32, "hidden2": 16, "dropout": 0.3,
        },
    }
    return RetrainArtifacts(
        model_state_dict=best_state,
        scaler=scaler,
        metadata=metadata,
        eval_metrics=eval_metrics,
    )
