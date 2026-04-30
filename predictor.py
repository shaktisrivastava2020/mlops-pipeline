"""
Production predictor — loads currently-promoted model from GCS.

Hot-reloads when production pointer changes (checked on each predict call,
cached for 60 seconds). For Phase 5 we keep it simple: cache the loaded
model and version in module state, refresh on demand or at startup.
"""
from __future__ import annotations
from threading import Lock
from typing import Any

import numpy as np
import torch

from features import FEATURE_COLUMNS
from model import ChurnNet
from registry import (
    get_production_version,
    load_metadata,
    load_model_state,
    load_scaler,
)


_lock = Lock()
_state: dict[str, Any] = {"version": None, "model": None, "scaler": None, "metadata": None}


def _load(version: str) -> None:
    state_dict = load_model_state(version)
    scaler = load_scaler(version)
    metadata = load_metadata(version)
    hp = metadata.get("hyperparameters", {})
    model = ChurnNet(
        input_dim=metadata.get("input_dim", len(FEATURE_COLUMNS)),
        hidden1=hp.get("hidden1", 32),
        hidden2=hp.get("hidden2", 16),
        dropout=hp.get("dropout", 0.3),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _state["version"] = version
    _state["model"] = model
    _state["scaler"] = scaler
    _state["metadata"] = metadata


def reload() -> str | None:
    """Force a refresh from the production pointer. Returns the loaded version."""
    with _lock:
        version = get_production_version()
        if version is None:
            return None
        _load(version)
        return version


def current_version() -> str | None:
    return _state["version"]


def predict_one(features: dict[str, Any], threshold: float = 0.4) -> dict[str, Any]:
    if _state["model"] is None:
        if reload() is None:
            raise RuntimeError("No production model registered.")

    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    x = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
    x_scaled = _state["scaler"].transform(x)
    with torch.no_grad():
        logits = _state["model"](torch.from_numpy(x_scaled).float()).squeeze(-1)
        prob = torch.sigmoid(logits).item()

    return {
        "churn_probability": round(prob, 4),
        "churned_label": int(prob >= threshold),
        "threshold_used": threshold,
        "model_version": _state["version"],
    }


def info() -> dict[str, Any]:
    if _state["metadata"] is None:
        reload()
    return {
        "production_version": _state["version"],
        "metadata": _state["metadata"],
    }
