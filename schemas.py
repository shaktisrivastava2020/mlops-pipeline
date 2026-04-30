"""Pydantic schemas for Day 3 MLOps API."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
    service: str = "mlops-pipeline"
    version: str


class CustomerFeatures(BaseModel):
    tenure_days: float = Field(..., ge=0)
    total_orders: int = Field(..., ge=0)
    total_spend: float = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    days_since_last_order: float = Field(..., ge=0)
    orders_per_month: float = Field(..., ge=0)
    unique_products: int = Field(..., ge=0)
    unique_payment_methods: int = Field(..., ge=0)
    weekday_order_ratio: float = Field(..., ge=0, le=1)
    tier_standard: Literal[0, 1]
    tier_premium: Literal[0, 1]
    tier_gold: Literal[0, 1]


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    churned_label: Literal[0, 1]
    threshold_used: float
    model_version: str


class ModelInfoResponse(BaseModel):
    production_version: str | None
    versions_available: list[str]
    metadata: dict[str, Any] | None


class RunPipelineRequest(BaseModel):
    inject_drift_preset: str | None = Field(
        default=None,
        description="Optional: simulate a drift batch (engagement_decline | high_value_shift | premium_surge | new_payment_method)",
    )
    skip_retraining: bool = Field(default=False, description="If true, only run drift check")


class RunPipelineResponse(BaseModel):
    batch_id: str
    drift_detected: bool
    drift_reason: str
    n_features_drifted: int
    retrained: bool
    challenger_version: str | None
    promoted: bool
    promotion_reasons: list[str]
    champion_f1: float | None
    challenger_f1: float | None
    f1_delta: float | None
