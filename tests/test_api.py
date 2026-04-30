"""Integration tests for the FastAPI app — endpoint contracts only, no live retrain."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


VALID_FEATURES = {
    "tenure_days": 365, "total_orders": 8, "total_spend": 80000,
    "avg_order_value": 10000, "days_since_last_order": 15, "orders_per_month": 2.0,
    "unique_products": 6, "unique_payment_methods": 3, "weekday_order_ratio": 0.7,
    "tier_standard": 0, "tier_premium": 1, "tier_gold": 0,
}


class TestSystem:
    def test_root_returns_service_info(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "mlops-pipeline"

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_model_info_has_production_version(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        assert "production_version" in body
        assert "versions_available" in body


class TestPredict:
    def test_valid_input_returns_prediction(self, client):
        r = client.post("/predict", json=VALID_FEATURES)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churned_label"] in (0, 1)
        assert body["model_version"] is not None

    def test_negative_value_rejected(self, client):
        bad = {**VALID_FEATURES, "tenure_days": -1}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_weekday_ratio_above_one_rejected(self, client):
        bad = {**VALID_FEATURES, "weekday_order_ratio": 1.5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_missing_field_rejected(self, client):
        bad = {k: v for k, v in VALID_FEATURES.items() if k != "tier_gold"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


class TestModels:
    def test_models_list_returns_versions(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        body = r.json()
        assert "versions" in body
        assert body["count"] >= 1


class TestAudit:
    def test_audit_recent_returns_list(self, client):
        r = client.get("/audit/recent?limit=5")
        assert r.status_code == 200
        body = r.json()
        assert "entries" in body
        assert body["count"] >= 0

    def test_audit_limit_enforced(self, client):
        r = client.get("/audit/recent?limit=200")
        assert r.status_code == 422  # > 100 max


class TestRunPipelineSkipRetraining:
    def test_drift_only_run(self, client):
        """Drift-check-only run completes without firing retrain (fast)."""
        r = client.post("/run-pipeline", json={"skip_retraining": True})
        assert r.status_code == 200
        body = r.json()
        assert "batch_id" in body
        assert body["retrained"] is False
        assert body["promoted"] is False
