"""Unit tests for evaluation + promotion logic."""
from __future__ import annotations
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation import (
    ModelEvalResult,
    PromotionDecision,
    SliceMetric,
    decide_promotion,
    evaluate,
)
from features import FEATURE_COLUMNS
from model import ChurnNet


def _trained_artifacts(seed: int = 0):
    """Train a tiny dummy model so we have a real state_dict + scaler to feed evaluate()."""
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    n = 80
    df = pd.DataFrame({c: rng.uniform(0, 10, n) for c in FEATURE_COLUMNS})
    df["tier_standard"] = rng.integers(0, 2, n)
    df["tier_premium"] = (1 - df["tier_standard"]) * rng.integers(0, 2, n)
    df["tier_gold"] = 1 - df["tier_standard"] - df["tier_premium"]
    df["churned"] = rng.integers(0, 2, n)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    model = ChurnNet(input_dim=len(FEATURE_COLUMNS))
    return model.state_dict(), scaler, df


class TestEvaluate:
    def test_returns_overall_and_slices(self):
        state, scaler, df = _trained_artifacts(0)
        result = evaluate(state, scaler, df, threshold=0.5)
        assert isinstance(result, ModelEvalResult)
        assert 0.0 <= result.overall_f1 <= 1.0
        assert result.threshold == 0.5
        assert result.n_test == len(df)
        assert all(isinstance(s, SliceMetric) for s in result.slices)

    def test_missing_churned_raises(self):
        state, scaler, df = _trained_artifacts(0)
        with pytest.raises(ValueError, match="churned"):
            evaluate(state, scaler, df.drop(columns=["churned"]))


def _result(f1: float, slice_f1s: dict[str, float]) -> ModelEvalResult:
    return ModelEvalResult(
        overall_f1=f1, overall_precision=f1, overall_recall=f1,
        overall_roc_auc=f1, n_test=100, n_positive=30, threshold=0.4,
        slices=[SliceMetric(slice_name=k, n=20, n_positive=8, f1=v) for k, v in slice_f1s.items()],
    )


class TestDecidePromotion:
    def test_promote_when_f1_improves_and_no_regression(self):
        c = _result(0.50, {"tier_standard": 0.50, "tier_premium": 0.50})
        ch = _result(0.55, {"tier_standard": 0.55, "tier_premium": 0.52})
        d = decide_promotion(c, ch)
        assert d.promote is True
        assert any("PASSED" in r for r in d.reasons)

    def test_reject_when_f1_improvement_below_threshold(self):
        c = _result(0.50, {"tier_standard": 0.50})
        ch = _result(0.51, {"tier_standard": 0.51})
        d = decide_promotion(c, ch, min_f1_improvement=0.02)
        assert d.promote is False
        assert any("F1 improvement" in r for r in d.reasons)

    def test_reject_when_segment_regresses(self):
        c = _result(0.50, {"tier_standard": 0.50, "tier_premium": 0.70})
        ch = _result(0.60, {"tier_standard": 0.60, "tier_premium": 0.55})  # -0.15 on premium
        d = decide_promotion(c, ch, max_segment_regression=0.05)
        assert d.promote is False
        assert d.worst_segment_name == "tier_premium"
        assert d.worst_segment_delta < -0.05

    def test_decision_serializes_to_dict(self):
        c = _result(0.50, {"tier_standard": 0.50})
        ch = _result(0.55, {"tier_standard": 0.55})
        d = decide_promotion(c, ch)
        as_dict = d.to_dict()
        assert "promote" in as_dict
        assert "reasons" in as_dict
        assert "f1_delta" in as_dict
