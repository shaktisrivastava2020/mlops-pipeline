"""
Synthetic data simulator for the MLOps pipeline.

Two modes:
  - 'baseline' : data drawn from Day 2's empirical training distribution
                 (loaded from reference_stats.json). Drift detection
                 should report ~no drift on this output.
  - 'drift'    : 'baseline' with one or more features deliberately shifted
                 via a named preset.

Reproducibility: every call takes a `seed`. Same seed -> same data.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


REFERENCE_PATH = Path("reference_stats.json")


def _load_payment_methods_from_db() -> list[str]:
    """Read the canonical payment-method list from the production DB.

    Why DB-backed: the simulator must produce values that match what the
    model actually saw in training. Hardcoded names diverge silently and
    create false drift signals (we hit this in Phase 3 dev).
    """
    from sqlalchemy import text
    from database import get_engine
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT payment_method FROM orders WHERE payment_method IS NOT NULL"
        )).all()
    return sorted(r[0] for r in rows)


def _load_baseline_from_reference(path: Path = REFERENCE_PATH) -> dict:
    """Seed simulator parameters from Day 2's empirical distribution."""
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    ref = json.loads(path.read_text())
    fs = ref["feature_stats"]

    def stat_mean(feat_name: str) -> float:
        """Get mean of a feature regardless of whether it's stored as continuous or discrete."""
        s = fs[feat_name]
        if s["type"] == "continuous":
            return s["mean"]
        if s["type"] == "discrete":
            # Weighted mean of bin values (string keys -> floats).
            return sum(float(k) * v for k, v in s["proportions"].items())
        if s["type"] == "binary":
            return s["proportion_one"]
        raise ValueError(f"Unknown feature type for {feat_name}: {s['type']}")

    # Tier proportions from the three one-hot columns.
    tier_props = {
        "Standard": fs["tier_standard"]["proportion_one"],
        "Premium":  fs["tier_premium"]["proportion_one"],
        "Gold":     fs["tier_gold"]["proportion_one"],
    }
    s = sum(tier_props.values())
    tier_props = {k: v / s for k, v in tier_props.items()}

    return {
        "tenure_days_mean":          stat_mean("tenure_days"),
        "tenure_days_std":           fs["tenure_days"]["std"],
        "orders_per_customer_mean":  stat_mean("total_orders"),
        "order_amount_mean":         stat_mean("avg_order_value"),
        "order_amount_std":          fs["avg_order_value"]["std"],
        "tier_proportions":          tier_props,
        "weekday_order_probability": stat_mean("weekday_order_ratio"),
        "n_payment_methods":         int(round(stat_mean("unique_payment_methods"))),
        "n_products":                int(round(stat_mean("unique_products"))),
        "payment_methods":           _load_payment_methods_from_db(),
        "n_total_products":          100,
    }


# Drift presets — each is a *delta* applied on top of the baseline.
DRIFT_PRESETS: dict[str, dict] = {
    "engagement_decline": {
        "orders_per_customer_mean_mult": 0.4,    # 60% drop
        "weekday_order_probability":     0.55,
    },
    "premium_surge": {
        "tier_proportions": {"Standard": 0.30, "Premium": 0.40, "Gold": 0.30},
    },
    "high_value_shift": {
        "order_amount_mean_mult": 1.6,
        "order_amount_std_mult":  1.5,
    },
    "new_payment_method": {
        "payment_methods_extra": ["BNPL"],
    },
}


@dataclass
class SimulatorConfig:
    n_customers: int = 100
    mode: Literal["baseline", "drift"] = "baseline"
    drift_preset: str | None = None
    seed: int = 42
    batch_id: str = field(default_factory=lambda: f"sim_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}")
    reference_date: datetime | pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now())


def _resolved_params(config: SimulatorConfig) -> dict:
    """Apply the drift preset (if any) on top of the empirical baseline."""
    params = _load_baseline_from_reference()
    if config.mode == "drift":
        if not config.drift_preset:
            raise ValueError("mode='drift' requires drift_preset to be set")
        if config.drift_preset not in DRIFT_PRESETS:
            raise ValueError(
                f"Unknown drift_preset '{config.drift_preset}'. "
                f"Available: {sorted(DRIFT_PRESETS.keys())}"
            )
        delta = DRIFT_PRESETS[config.drift_preset]
        for k, v in delta.items():
            if k.endswith("_mult"):
                base_key = k[:-5]
                params[base_key] = params[base_key] * v
            elif k.endswith("_extra"):
                base_key = k[:-6]
                params[base_key] = list(params[base_key]) + list(v)
            else:
                params[k] = v
    s = sum(params["tier_proportions"].values())
    if not 0.99 <= s <= 1.01:
        raise ValueError(f"tier_proportions must sum to ~1.0, got {s}")
    return params


def simulate(config: SimulatorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    p = _resolved_params(config)
    ref_dt = pd.Timestamp(config.reference_date)

    # ---- customers ----
    tiers = list(p["tier_proportions"].keys())
    tier_probs = list(p["tier_proportions"].values())
    tenure_days = np.maximum(
        1,
        rng.normal(p["tenure_days_mean"], p["tenure_days_std"], size=config.n_customers).astype(int),
    )
    customer_tiers = rng.choice(tiers, size=config.n_customers, p=tier_probs)
    join_dates = [(ref_dt - pd.Timedelta(days=int(t))).date() for t in tenure_days]

    customers_df = pd.DataFrame({
        "_local_idx": np.arange(config.n_customers),
        "customer_tier": customer_tiers,
        "join_date": join_dates,
        "batch_id": config.batch_id,
    })

    # ---- orders ----
    n_orders = rng.poisson(p["orders_per_customer_mean"], size=config.n_customers)
    n_orders = np.maximum(n_orders, 1)

    rows = []
    for cust_idx, n in enumerate(n_orders):
        cust_tenure = int(tenure_days[cust_idx])
        for _ in range(n):
            # Bias order dates toward recent (matches real customer behaviour:
            # orders cluster recently, not uniformly across full tenure).
            offset_days = int(rng.beta(1.5, 4.0) * cust_tenure)
            order_dt = ref_dt - pd.Timedelta(days=offset_days)

            # Weekday vs weekend nudge.
            target_weekday = rng.random() < p["weekday_order_probability"]
            for _ in range(7):
                if (order_dt.weekday() < 5) == target_weekday:
                    break
                order_dt -= pd.Timedelta(days=1)

            order_amount = max(1.0, float(rng.normal(p["order_amount_mean"], p["order_amount_std"])))
            payment = rng.choice(p["payment_methods"])
            product_id = int(rng.integers(1, p["n_total_products"] + 1))
            quantity = int(rng.integers(1, 5))

            rows.append({
                "_local_cust_idx": cust_idx,
                "product_id": product_id,
                "quantity": quantity,
                "order_amount": round(order_amount, 2),
                "payment_method": payment,
                "order_date": order_dt,
                "batch_id": config.batch_id,
            })

    orders_df = pd.DataFrame(rows)
    return customers_df, orders_df


def summarize(customers_df: pd.DataFrame, orders_df: pd.DataFrame) -> dict:
    return {
        "n_customers": len(customers_df),
        "n_orders": len(orders_df),
        "mean_orders_per_customer": float(orders_df.groupby("_local_cust_idx").size().mean()),
        "mean_order_amount": float(orders_df["order_amount"].mean()),
        "tier_proportions": customers_df["customer_tier"].value_counts(normalize=True).to_dict(),
        "weekday_ratio": float((pd.to_datetime(orders_df["order_date"]).dt.weekday < 5).mean()),
        "unique_payment_methods": int(orders_df["payment_method"].nunique()),
    }
