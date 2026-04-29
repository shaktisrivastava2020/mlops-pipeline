"""
One-time script: snapshot statistics from Day 2's training.csv to serve as
the immutable reference distribution for drift detection.

Run this ONCE. The output (reference_stats.json) is committed to git and
treated as read-only by the rest of the pipeline.

Why immutable: in production MLOps, the "reference" is the data the
deployed model was trained on. It does not move when fresh data arrives.
Drift means "current data differs from training data" — that comparison
is meaningless if the reference itself drifts.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Feature contract — must match Day 2's features.py exactly.
# Order is preserved for reproducibility but not load-bearing for stats.
FEATURE_COLUMNS = [
    "tenure_days",
    "total_orders",
    "total_spend",
    "avg_order_value",
    "days_since_last_order",
    "orders_per_month",
    "unique_products",
    "unique_payment_methods",
    "weekday_order_ratio",
    "tier_standard",
    "tier_premium",
    "tier_gold",
]

# One-hot tier columns are binary — profile as proportions, not quantiles.
BINARY_FEATURES = {"tier_standard", "tier_premium", "tier_gold"}

INPUT_CSV = Path("data/training.csv")
OUTPUT_JSON = Path("reference_stats.json")


def file_sha256(path: Path) -> str:
    """Hash the source file so the reference is provably tied to its origin."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_continuous(series: pd.Series) -> dict:
    """Distribution summary for a continuous feature."""
    return {
        "type": "continuous",
        "n": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def profile_binary(series: pd.Series) -> dict:
    """Distribution summary for a binary (0/1) feature."""
    return {
        "type": "binary",
        "n": int(series.count()),
        "proportion_one": float(series.mean()),  # for 0/1 column, mean == proportion of 1s
    }


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Expected {INPUT_CSV}. Did Phase 1 step 16 run?")

    df = pd.read_csv(INPUT_CSV)

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required features: {sorted(missing)}")

    feature_stats: dict[str, dict] = {}
    for col in FEATURE_COLUMNS:
        series = df[col].dropna()
        if col in BINARY_FEATURES:
            feature_stats[col] = profile_binary(series)
        else:
            feature_stats[col] = profile_continuous(series)

    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(INPUT_CSV),
        "source_sha256": file_sha256(INPUT_CSV),
        "n_rows": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "feature_stats": feature_stats,
    }

    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUTPUT_JSON} — {len(FEATURE_COLUMNS)} features, {len(df)} rows.")


if __name__ == "__main__":
    main()
