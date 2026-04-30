"""
Model registry — versioned storage of model artifacts in GCS.

Layout (in gs://<MODELS_BUCKET>):
  models/v{N}/model.pth
  models/v{N}/scaler.joblib
  models/v{N}/metadata.json
  models/v{N}/eval_report.json
  production/current.json   ->  {"version": "v{N}", "promoted_at": "<iso>"}
  drift/<batch_id>.json     ->  per-run drift reports
  audit/<batch_id>.json     ->  per-run pipeline audit logs

Versions are immutable once written. Promotion only updates production/current.json.
"""
from __future__ import annotations
import io
import json
from datetime import datetime, timezone
from typing import Any

import joblib
import torch
from google.cloud import storage

from config import settings
from retrain import RetrainArtifacts


_client: storage.Client | None = None


def _get_client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client(project=settings.GCP_PROJECT_ID)
    return _client


def _bucket() -> storage.Bucket:
    return _get_client().bucket(settings.GCS_MODELS_BUCKET)


# ---------- version discovery ----------

def list_versions() -> list[str]:
    """Return all registered versions, sorted ascending (v0, v1, v2, ...)."""
    blobs = _get_client().list_blobs(settings.GCS_MODELS_BUCKET, prefix="models/")
    versions = set()
    for b in blobs:
        # Path format: models/v{N}/...
        parts = b.name.split("/")
        if len(parts) >= 2 and parts[1].startswith("v") and parts[1][1:].isdigit():
            versions.add(parts[1])
    return sorted(versions, key=lambda v: int(v[1:]))


def next_version() -> str:
    """Next sequential version id."""
    existing = list_versions()
    if not existing:
        return "v0"
    last = max(int(v[1:]) for v in existing)
    return f"v{last + 1}"


# ---------- write ----------

def register(version: str, artifacts: RetrainArtifacts) -> dict[str, str]:
    """
    Upload a trained model + scaler + metadata + eval to GCS under models/<version>/.
    Returns the GCS URIs written. Idempotent within a session, NOT across sessions
    (re-registering an existing version overwrites it — caller must use next_version()).
    """
    bucket = _bucket()
    prefix = f"models/{version}"

    # model.pth
    buf = io.BytesIO()
    torch.save(artifacts.model_state_dict, buf)
    bucket.blob(f"{prefix}/model.pth").upload_from_string(
        buf.getvalue(), content_type="application/octet-stream"
    )

    # scaler.joblib
    buf2 = io.BytesIO()
    joblib.dump(artifacts.scaler, buf2)
    bucket.blob(f"{prefix}/scaler.joblib").upload_from_string(
        buf2.getvalue(), content_type="application/octet-stream"
    )

    # metadata.json
    md = {**artifacts.metadata, "version": version, "registered_at": datetime.now(timezone.utc).isoformat()}
    bucket.blob(f"{prefix}/metadata.json").upload_from_string(
        json.dumps(md, indent=2), content_type="application/json"
    )

    # eval_report.json
    bucket.blob(f"{prefix}/eval_report.json").upload_from_string(
        json.dumps(artifacts.eval_metrics, indent=2), content_type="application/json"
    )

    return {
        "model": f"gs://{settings.GCS_MODELS_BUCKET}/{prefix}/model.pth",
        "scaler": f"gs://{settings.GCS_MODELS_BUCKET}/{prefix}/scaler.joblib",
        "metadata": f"gs://{settings.GCS_MODELS_BUCKET}/{prefix}/metadata.json",
        "eval_report": f"gs://{settings.GCS_MODELS_BUCKET}/{prefix}/eval_report.json",
    }


# ---------- read ----------

def load_metadata(version: str) -> dict[str, Any]:
    blob = _bucket().blob(f"models/{version}/metadata.json")
    if not blob.exists():
        raise FileNotFoundError(f"metadata.json not found for {version}")
    return json.loads(blob.download_as_text())


def load_eval_report(version: str) -> dict[str, Any]:
    blob = _bucket().blob(f"models/{version}/eval_report.json")
    if not blob.exists():
        raise FileNotFoundError(f"eval_report.json not found for {version}")
    return json.loads(blob.download_as_text())


def load_model_state(version: str) -> dict[str, torch.Tensor]:
    blob = _bucket().blob(f"models/{version}/model.pth")
    if not blob.exists():
        raise FileNotFoundError(f"model.pth not found for {version}")
    buf = io.BytesIO(blob.download_as_bytes())
    return torch.load(buf, weights_only=True)


def load_scaler(version: str):
    blob = _bucket().blob(f"models/{version}/scaler.joblib")
    if not blob.exists():
        raise FileNotFoundError(f"scaler.joblib not found for {version}")
    buf = io.BytesIO(blob.download_as_bytes())
    return joblib.load(buf)


# ---------- production pointer ----------

def get_production_version() -> str | None:
    blob = _bucket().blob(settings.PRODUCTION_POINTER_KEY)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())["version"]


def set_production_version(version: str, reason: str = "") -> dict[str, str]:
    """Promote a registered version to production. Atomic via single blob write."""
    if version not in list_versions():
        raise ValueError(f"Cannot promote unregistered version: {version}")
    payload = {
        "version": version,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    _bucket().blob(settings.PRODUCTION_POINTER_KEY).upload_from_string(
        json.dumps(payload, indent=2), content_type="application/json"
    )
    return payload


# ---------- audit & drift logs ----------

def write_drift_report(batch_id: str, report_dict: dict[str, Any]) -> str:
    """Persist a drift detection result. Append-only by batch_id."""
    key = f"drift/{batch_id}.json"
    _bucket().blob(key).upload_from_string(
        json.dumps(report_dict, indent=2), content_type="application/json"
    )
    return f"gs://{settings.GCS_MODELS_BUCKET}/{key}"


def write_audit_log(batch_id: str, log_entry: dict[str, Any]) -> str:
    """Persist a full pipeline-run audit entry. Append-only."""
    key = f"audit/{batch_id}.json"
    _bucket().blob(key).upload_from_string(
        json.dumps(log_entry, indent=2), content_type="application/json"
    )
    return f"gs://{settings.GCS_MODELS_BUCKET}/{key}"
