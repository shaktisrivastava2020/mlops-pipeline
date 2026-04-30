"""
Centralized config for the MLOps pipeline.
Loads from environment variables (.env in dev, Cloud Run env vars in prod).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Secrets via .env, behavior via defaults."""

    # --- GCP infrastructure (reused from Day 1 + Day 2) ---
    GCP_PROJECT_ID: str = "project-c5185bee-a238-4d53-b9b"
    GCP_REGION: str = "asia-south1"
    CLOUD_SQL_INSTANCE: str = "project-c5185bee-a238-4d53-b9b:asia-south1:quickshop-db"
    DB_NAME: str = "quickshop"
    DB_USER: str = "postgres"
    DB_PASSWORD: str  # required, from .env

    # --- Model registry (GCS) ---
    GCS_MODELS_BUCKET: str = "mlops-models-218990051802"  # to be created in Phase 4
    PRODUCTION_POINTER_KEY: str = "production/current.json"

    # --- Drift detection thresholds ---
    PSI_DRIFT_THRESHOLD: float = 0.20    # PSI > 0.20 => meaningful drift (industry standard)
    KS_DRIFT_PVALUE: float = 0.05         # KS p-value < 0.05 => distribution shifted
    DRIFT_FEATURE_FRACTION: float = 0.50  # >50% of features drifted => trigger retrain (calibrated against synthetic baseline noise)

    # --- Promotion criteria (champion vs challenger) ---
    MIN_F1_IMPROVEMENT: float = 0.02     # challenger must beat champion F1 by >= 0.02
    MAX_SEGMENT_REGRESSION: float = 0.05  # no slice may regress by more than 5%

    # Severity-based drift triggers — any single feature with this much
    # shift triggers retrain, regardless of how many other features moved.
    SEVERE_PSI_THRESHOLD: float = 0.25
    SEVERE_KS_STATISTIC: float = 0.25

    # --- Reference data ---
    REFERENCE_STATS_PATH: str = "reference_stats.json"

    # --- Alerting ---
    SLACK_WEBHOOK_URL: str = ""  # optional, set in Phase 6

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
