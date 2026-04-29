"""
One-time script: snapshot statistics from Day 2's training.csv to serve as
the immutable reference distribution for drift detection.

Run this ONCE. The output (reference_stats.json) is committed to git and
treated as read-only by the rest of the pipeline.

Why immutable: in production MLOps, the "reference" is the data the
deployed model was trained on. It does not move when fresh data arrives.
Drift means "current data differs from training data" — that comparison
is meaningless if the reference itself drifts.

Feature type detection (decided at profile time, used by drift.py):
  - "binary"     : declared via BINARY_FEATURES
  - "discrete"   : non-binary numeric with <= DISCRETE_CARDINALITY_THRESHOLD unique values
                   (KS test is unreliable on heavy-tied data; PSI on observed bins is correct)
  - "continuous" : everything else (KS test applies)
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Feature contract — must match Day 2's features.py exactly.
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

# One-hot tier columns are binary indicators — declared explicitly.
BINARY_FEATURES = {"tier_standard", "tier_premium", "tier_gold"}

# Cardinality threshold: features with <= this many unique values
# are profiled as discrete (PSI on observed bins) instead of continuous (KS).
# Industry convention: 10-20. We use 10 to be conservative.
DISCRETE_CARDINALITY_THRESHOLD = 10

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
    """Distribution summary for a continuous feature (KS-testable)."""
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


def profile_discrete(series: pd.Series) -> dict:
    """
    Distribution summary for a low-cardinality discrete feature.
    Stores observed value -> proportion. PSI compares these proportions.
    Keys are JSON-string-safe (cast from numpy types).
    """
    counts = series.value_counts(normalize=True).sort_index()
    proportions = {str(k): float(v) for k, v in counts.items()}
    return {
        "type": "discrete",
        "n": int(series.count()),
        "n_unique": int(series.nunique()),
        "proportions": proportions,
    }


def profile_binary(series: pd.Series) -> dict:
    """Distribution summary for a binary (0/1) feature (PSI-testable)."""
    return {
        "type": "binary",
        "n": int(series.count()),
        "proportion_one": float(series.mean()),
    }


def classify_feature(col: str, series: pd.Series) -> str:
    """Decide which profiler to use for this feature."""
    if col in BINARY_FEATURES:
        return "binary"
    if series.nunique() <= DISCRETE_CARDINALITY_THRESHOLD:
        return "discrete"
    return "continuous"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Expected {INPUT_CSV}. Did Phase 1 step 16 run?")

    df = pd.read_csv(INPUT_CSV)

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required features: {sorted(missing)}")

    feature_stats: dict[str, dict] = {}
    type_counts = {"binary": 0, "discrete": 0, "continuous": 0}
    for col in FEATURE_COLUMNS:
        series = df[col].dropna()
        ftype = classify_feature(col, series)
        type_counts[ftype] += 1
        if ftype == "binary":
            feature_stats[col] = profile_binary(series)
        elif ftype == "discrete":
            feature_stats[col] = profile_discrete(series)
        else:
            feature_stats[col] = profile_continuous(series)

    output = {
        "schema_version": "1.1",   # bumped: discrete type added
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(INPUT_CSV),
        "source_sha256": file_sha256(INPUT_CSV),
        "n_rows": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "feature_stats": feature_stats,
    }

    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(
        f"Wrote {OUTPUT_JSON} — {len(FEATURE_COLUMNS)} features, {len(df)} rows. "
        f"Types: binary={type_counts['binary']}, discrete={type_counts['discrete']}, "
        f"continuous={type_counts['continuous']}."
    )


if __name__ == "__main__":
    main()
