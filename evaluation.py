"""
Champion vs challenger comparison + slice analysis.

Promotion criteria (all must hold for promote=True):
  1. Challenger overall F1 >= champion F1 + MIN_F1_IMPROVEMENT
  2. No segment regresses by more than MAX_SEGMENT_REGRESSION (in F1)
  3. Both models evaluated on the same holdout set with the same threshold

Segments: customer_tier (Standard / Premium / Gold) + tenure_band (short/medium/long).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
import torch

from features import FEATURE_COLUMNS
from model import ChurnNet


# ---------- result classes ----------

@dataclass(frozen=True)
class SliceMetric:
    slice_name: str
    n: int
    n_positive: int
    f1: float


@dataclass(frozen=True)
class ModelEvalResult:
    overall_f1: float
    overall_precision: float
    overall_recall: float
    overall_roc_auc: float
    n_test: int
    n_positive: int
    threshold: float
    slices: list[SliceMetric]


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: list[str]
    champion: ModelEvalResult
    challenger: ModelEvalResult
    f1_delta: float
    worst_segment_delta: float
    worst_segment_name: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------- evaluation helpers ----------

def _model_from_state(state_dict: dict[str, torch.Tensor], hidden1=32, hidden2=16, dropout=0.3) -> ChurnNet:
    m = ChurnNet(input_dim=len(FEATURE_COLUMNS), hidden1=hidden1, hidden2=hidden2, dropout=dropout)
    m.load_state_dict(state_dict, strict=True)
    m.eval()
    return m


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _roc_auc(probs: np.ndarray, y: np.ndarray) -> float:
    pos = probs[y == 1]
    neg = probs[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    cmp = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(cmp) / (len(pos) * len(neg))


def _tenure_band(days: float) -> str:
    if days < 180:
        return "short"
    if days < 365:
        return "medium"
    return "long"


def evaluate(
    state_dict: dict[str, torch.Tensor],
    scaler,
    test_df: pd.DataFrame,
    threshold: float = 0.4,
) -> ModelEvalResult:
    """
    Evaluate a model on a holdout DataFrame containing FEATURE_COLUMNS + 'churned'.
    Returns overall + per-slice metrics.
    """
    if "churned" not in test_df.columns:
        raise ValueError("test_df must include 'churned'")
    X = test_df[FEATURE_COLUMNS].values.astype(np.float32)
    y = test_df["churned"].values.astype(np.float32).astype(int)
    Xs = scaler.transform(X)

    model = _model_from_state(state_dict)
    with torch.no_grad():
        logits = model(torch.from_numpy(Xs).float()).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)

    overall_f1 = _f1(y, preds)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    roc_auc = _roc_auc(probs, y)

    # Slices: by tier + by tenure band.
    slices: list[SliceMetric] = []
    for tier in ("standard", "premium", "gold"):
        col = f"tier_{tier}"
        mask = test_df[col].astype(int).values == 1
        if mask.sum() >= 5:  # require minimum sample
            slices.append(SliceMetric(
                slice_name=f"tier_{tier}",
                n=int(mask.sum()),
                n_positive=int(y[mask].sum()),
                f1=_f1(y[mask], preds[mask]),
            ))
    for band in ("short", "medium", "long"):
        mask = test_df["tenure_days"].apply(_tenure_band).values == band
        if mask.sum() >= 5:
            slices.append(SliceMetric(
                slice_name=f"tenure_{band}",
                n=int(mask.sum()),
                n_positive=int(y[mask].sum()),
                f1=_f1(y[mask], preds[mask]),
            ))

    return ModelEvalResult(
        overall_f1=overall_f1,
        overall_precision=precision,
        overall_recall=recall,
        overall_roc_auc=roc_auc,
        n_test=int(len(y)),
        n_positive=int(y.sum()),
        threshold=threshold,
        slices=slices,
    )


# ---------- promotion ----------

def decide_promotion(
    champion: ModelEvalResult,
    challenger: ModelEvalResult,
    min_f1_improvement: float = 0.02,
    max_segment_regression: float = 0.05,
) -> PromotionDecision:
    """Apply the promotion criteria; return a decision with explicit reasons."""
    reasons: list[str] = []

    f1_delta = challenger.overall_f1 - champion.overall_f1
    if f1_delta < min_f1_improvement:
        reasons.append(
            f"F1 improvement {f1_delta:+.4f} < required {min_f1_improvement:+.4f}"
        )

    # Segment regression check — only on slices present in both.
    champ_slices = {s.slice_name: s.f1 for s in champion.slices}
    chal_slices = {s.slice_name: s.f1 for s in challenger.slices}
    common = sorted(set(champ_slices) & set(chal_slices))
    worst_delta = 0.0
    worst_name = ""
    for name in common:
        delta = chal_slices[name] - champ_slices[name]
        if delta < worst_delta:
            worst_delta = delta
            worst_name = name
    if -worst_delta > max_segment_regression:
        reasons.append(
            f"Segment '{worst_name}' regresses by {worst_delta:+.4f} (> {max_segment_regression})"
        )

    promote = len(reasons) == 0
    if promote:
        reasons.append(
            f"PASSED: F1 +{f1_delta:.4f}, worst segment {worst_name or 'n/a'} {worst_delta:+.4f}"
        )

    return PromotionDecision(
        promote=promote,
        reasons=reasons,
        champion=champion,
        challenger=challenger,
        f1_delta=f1_delta,
        worst_segment_delta=worst_delta,
        worst_segment_name=worst_name or "n/a",
    )
