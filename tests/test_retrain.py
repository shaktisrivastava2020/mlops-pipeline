"""Unit tests for retrain: contract, reproducibility, and metric sanity."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS
from retrain import RetrainArtifacts, retrain


def _synthetic_dataset(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A tiny labeled dataset with realistic-ish feature ranges and a learnable signal.

    The label is set so that customers with high days_since_last_order tend to
    churn — a real-but-noisy signal a 32-16-1 NN can pick up.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure_days": rng.integers(60, 500, n),
        "total_orders": rng.integers(2, 12, n),
        "total_spend": rng.uniform(10000, 80000, n),
        "avg_order_value": rng.uniform(2000, 15000, n),
        "days_since_last_order": rng.uniform(1, 200, n),
        "orders_per_month": rng.uniform(0.1, 2.0, n),
        "unique_products": rng.integers(2, 10, n),
        "unique_payment_methods": rng.integers(1, 6, n),
        "weekday_order_ratio": rng.uniform(0.3, 0.9, n),
        "tier_standard": rng.integers(0, 2, n),
        "tier_premium": 0,
        "tier_gold": 0,
    })
    # Make tier columns mutually exclusive
    df["tier_premium"] = ((df["tier_standard"] == 0) & (rng.random(n) < 0.6)).astype(int)
    df["tier_gold"] = ((df["tier_standard"] == 0) & (df["tier_premium"] == 0)).astype(int)
    # Label with a noisy signal on days_since_last_order
    score = (df["days_since_last_order"] / 200) + rng.normal(0, 0.3, n)
    df["churned"] = (score > 0.55).astype(int)
    return df


class TestRetrainContract:
    def test_returns_artifacts(self):
        df = _synthetic_dataset()
        out = retrain(df, seed=42)
        assert isinstance(out, RetrainArtifacts)
        assert out.model_state_dict is not None
        assert out.scaler is not None
        assert "feature_columns" in out.metadata
        assert out.metadata["feature_columns"] == FEATURE_COLUMNS

    def test_missing_churned_column_raises(self):
        df = _synthetic_dataset().drop(columns=["churned"])
        with pytest.raises(ValueError, match="churned"):
            retrain(df)

    def test_missing_feature_column_raises(self):
        df = _synthetic_dataset().drop(columns=["tenure_days"])
        with pytest.raises(ValueError, match="missing FEATURE_COLUMNS"):
            retrain(df)


class TestRetrainReproducibility:
    def test_same_seed_same_metadata(self):
        df = _synthetic_dataset()
        a = retrain(df, seed=7)
        b = retrain(df, seed=7)
        assert a.metadata["best_epoch"] == b.metadata["best_epoch"]
        assert a.metadata["best_val_loss"] == pytest.approx(b.metadata["best_val_loss"], rel=1e-6)
        assert a.metadata["pos_weight"] == pytest.approx(b.metadata["pos_weight"], rel=1e-6)

    def test_different_seeds_can_differ(self):
        df = _synthetic_dataset()
        a = retrain(df, seed=1)
        b = retrain(df, seed=2)
        # Splits differ -> at least one of these should differ.
        assert (a.metadata["best_epoch"] != b.metadata["best_epoch"]) or (
            a.metadata["best_val_loss"] != b.metadata["best_val_loss"]
        )


class TestRetrainEvalMetrics:
    def test_metrics_in_valid_ranges(self):
        df = _synthetic_dataset(n=300, seed=0)
        out = retrain(df, seed=42)
        m = out.eval_metrics
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["n_test"]

    def test_threshold_propagated_to_metrics(self):
        df = _synthetic_dataset(n=300, seed=0)
        out = retrain(df, seed=42, threshold=0.6)
        assert out.eval_metrics["threshold"] == 0.6
        assert out.metadata["threshold"] == 0.6
