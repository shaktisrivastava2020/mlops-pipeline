"""API router — drift / pipeline / predict / model info / audit."""
from __future__ import annotations
import json
from fastapi import APIRouter, HTTPException, Query

import predictor
from config import settings
from registry import _bucket, list_versions, load_metadata
from pipeline import run_pipeline
from schemas import (
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RunPipelineRequest,
    RunPipelineResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(version="0.1.0")


@router.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    info = predictor.info()
    return ModelInfoResponse(
        production_version=info["production_version"],
        versions_available=list_versions(),
        metadata=info["metadata"],
    )


@router.post("/admin/reload-model", tags=["admin"])
def reload_model():
    """Re-read production pointer from GCS and load the new model into memory."""
    v = predictor.reload()
    if v is None:
        raise HTTPException(status_code=404, detail="No production model registered")
    return {"loaded_version": v}


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: CustomerFeatures) -> PredictionResponse:
    try:
        return PredictionResponse(**predictor.predict_one(features.model_dump()))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/run-pipeline", response_model=RunPipelineResponse, tags=["mlops"])
def run_pipeline_endpoint(req: RunPipelineRequest) -> RunPipelineResponse:
    """Trigger one full pipeline run: drift check -> retrain -> promote (if criteria met)."""
    try:
        result = run_pipeline(
            inject_drift_preset=req.inject_drift_preset,
            skip_retraining=req.skip_retraining,
        )
        # If a new version was promoted, refresh the in-memory predictor.
        if result.promoted:
            predictor.reload()
        return RunPipelineResponse(
            batch_id=result.batch_id,
            drift_detected=result.drift_detected,
            drift_reason=result.drift_reason,
            n_features_drifted=result.n_features_drifted,
            retrained=result.retrained,
            challenger_version=result.challenger_version,
            promoted=result.promoted,
            promotion_reasons=result.promotion_reasons,
            champion_f1=result.champion_f1,
            challenger_f1=result.challenger_f1,
            f1_delta=result.f1_delta,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/models", tags=["mlops"])
def models_list():
    """List every registered model version with its eval metrics."""
    versions = list_versions()
    items = []
    for v in versions:
        try:
            md = load_metadata(v)
            items.append({
                "version": v,
                "trained_at": md.get("trained_at"),
                "n_train": md.get("n_train"),
                "registered_at": md.get("registered_at"),
            })
        except Exception:
            items.append({"version": v, "error": "metadata unavailable"})
    return {"versions": items, "count": len(items)}


@router.get("/audit/recent", tags=["mlops"])
def audit_recent(limit: int = Query(10, ge=1, le=100)):
    """Return the most recent N pipeline-run audit entries."""
    blobs = list(_bucket().list_blobs(prefix="audit/"))
    blobs.sort(key=lambda b: b.name, reverse=True)
    entries = []
    for b in blobs[:limit]:
        try:
            entries.append(json.loads(b.download_as_text()))
        except Exception:
            continue
    return {"entries": entries, "count": len(entries)}
