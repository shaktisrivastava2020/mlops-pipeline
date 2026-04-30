"""
Drift detection module.

Compares a fresh dataset against a frozen reference distribution
(reference_stats.json from Phase 1) and reports per-feature drift.

Three tests, dispatched by feature type:
  - Continuous features  -> Kolmogorov-Smirnov test (compares full distributions)
  - Discrete features    -> PSI on observed-value proportions (avoids KS bias on tied data)
  - Binary features      -> PSI on 2-bin proportions

Why three: a single statistic cannot fairly judge continuous, discrete,
and binary features. Mixing them is a common silent failure in ad-hoc
drift implementations — KS on heavy-tied data produces false positives
(see development log for the diagnosed wrinkle).

Decision rule (configurable via config.py):
  - PSI > PSI_DRIFT_THRESHOLD (default 0.20)         => binary/discrete drifted
  - KS p-value < KS_DRIFT_PVALUE (default 0.05)      => continuous drifted
  - drifted_fraction > DRIFT_FEATURE_FRACTION (0.25) => system-level drift, retrain
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------- structured, serializable output ----------

@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift outcome for a single feature."""
    feature: str
    feature_type: str          # "continuous" | "discrete" | "binary"
    test: str                  # "ks" | "psi"
    statistic: float           # KS D-statistic OR PSI value
    p_value: float | None      # set for KS only
    drifted: bool
    reference_n: int
    current_n: int
    notes: str = ""


@dataclass(frozen=True)
class DriftReport:
    """System-level drift summary across all features."""
    n_features: int
    n_drifted: int
    drifted_fraction: float
    system_drift: bool         # True if EITHER rule below trips
    fraction_rule_tripped: bool   # > drift_feature_fraction features drifted
    severity_rule_tripped: bool   # >= 1 feature has severe drift (PSI or KS)
    feature_results: list[FeatureDriftResult]
    reference_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------- pure stat functions (easy to unit-test) ----------

def population_stability_index(
    reference_proportion: float,
    current_proportion: float,
    epsilon: float = 1e-6,
) -> float:
    """
    PSI for a binary feature, computed on 2 bins (0s and 1s).
    Formula: sum over bins of (p_cur - p_ref) * ln(p_cur / p_ref)

    Industry convention:
      < 0.10 -> stable
      0.10 - 0.25 -> moderate shift
      > 0.25 -> significant shift

    epsilon avoids log(0) when a bin has zero observations.
    """
    if not 0.0 <= reference_proportion <= 1.0:
        raise ValueError(f"reference_proportion must be in [0,1], got {reference_proportion}")
    if not 0.0 <= current_proportion <= 1.0:
        raise ValueError(f"current_proportion must be in [0,1], got {current_proportion}")

    p_ref_1 = max(reference_proportion, epsilon)
    p_cur_1 = max(current_proportion, epsilon)
    p_ref_0 = max(1.0 - reference_proportion, epsilon)
    p_cur_0 = max(1.0 - current_proportion, epsilon)

    psi = (
        (p_cur_1 - p_ref_1) * np.log(p_cur_1 / p_ref_1)
        + (p_cur_0 - p_ref_0) * np.log(p_cur_0 / p_ref_0)
    )
    return float(psi)


def population_stability_index_multibin(
    reference_proportions: dict[str, float],
    current_values: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    PSI for a discrete feature, computed across observed bins.

    Reference proportions come from the snapshot (str-keyed, e.g. {"2": 0.4, "3": 0.6}).
    Current proportions are computed on the fly from `current_values`.

    Bin handling:
      - Bins present in reference but missing in current   -> current proportion = epsilon
      - Bins present in current but new (not in reference) -> contribute (p_cur - eps) * ln(p_cur / eps)
        i.e., a brand-new value is treated as drift, not silently ignored.

    The "new value" handling is critical: if the model has never seen
    `unique_payment_methods=7` and suddenly 30% of customers have it, that
    IS drift, and silently dropping unknown bins would hide it.
    """
    if len(current_values) == 0:
        raise ValueError("current_values must be non-empty for PSI computation")

    # Compute current proportions, with bin keys as strings to match reference.
    cur_series = pd.Series(current_values)
    cur_counts = cur_series.value_counts(normalize=True)
    current_proportions: dict[str, float] = {str(k): float(v) for k, v in cur_counts.items()}

    all_bins = set(reference_proportions.keys()) | set(current_proportions.keys())
    psi = 0.0
    for b in all_bins:
        p_ref = max(reference_proportions.get(b, 0.0), epsilon)
        p_cur = max(current_proportions.get(b, 0.0), epsilon)
        psi += (p_cur - p_ref) * np.log(p_cur / p_ref)
    return float(psi)


def ks_test_against_reference_summary(
    reference_summary: dict[str, float],
    current_values: np.ndarray,
) -> tuple[float, float]:
    """
    KS comparison using a reconstructed reference sample drawn from the
    reference summary's quantiles via piecewise linear interpolation.

    This pragmatic approach is required because we keep stats only, not raw
    training data (privacy + storage). Trade-off: KS p-values are slightly
    less precise than with the raw data — acceptable for a *trigger* signal.

    Caveat: low-cardinality continuous features can produce false positives
    here (linear reconstruction smooths what is actually stepped). For that
    reason, low-cardinality numerics are profiled as "discrete" and routed
    to PSI in detect_drift(). See development notes for the diagnosed case.

    Returns (D-statistic, p-value).
    """
    quantile_points = np.array([0.0, 0.25, 0.50, 0.75, 0.95, 0.99, 1.00])
    quantile_values = np.array([
        reference_summary["min"],
        reference_summary["p25"],
        reference_summary["p50"],
        reference_summary["p75"],
        reference_summary["p95"],
        reference_summary["p99"],
        reference_summary["max"],
    ])

    n = max(len(current_values), 50)  # floor for stable interpolation
    uniform = np.linspace(0.0, 1.0, n)
    reference_sample = np.interp(uniform, quantile_points, quantile_values)

    result = stats.ks_2samp(reference_sample, current_values)
    return float(result.statistic), float(result.pvalue)


# ---------- top-level orchestration ----------

def load_reference(path: str | Path = "reference_stats.json") -> dict[str, Any]:
    """Load the immutable reference produced in Phase 1."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference file not found: {p}")
    return json.loads(p.read_text())


def detect_drift(
    current_df: pd.DataFrame,
    reference: dict[str, Any],
    psi_threshold: float = 0.20,
    ks_pvalue_threshold: float = 0.05,
    drift_feature_fraction: float = 0.25,
    severe_psi_threshold: float = 0.50,
    severe_ks_statistic: float = 0.30,
) -> DriftReport:
    """Compute drift for every feature in the reference; return system-level summary."""
    reference_features: dict[str, dict] = reference["feature_stats"]
    feature_columns: list[str] = reference["feature_columns"]

    missing = [c for c in feature_columns if c not in current_df.columns]
    if missing:
        raise ValueError(f"Current data missing reference features: {missing}")

    results: list[FeatureDriftResult] = []
    for feature in feature_columns:
        ref_stats = reference_features[feature]
        ref_type = ref_stats["type"]
        current_values = current_df[feature].dropna().to_numpy()

        if len(current_values) == 0:
            results.append(FeatureDriftResult(
                feature=feature, feature_type=ref_type, test="none",
                statistic=float("nan"), p_value=None, drifted=False,
                reference_n=ref_stats["n"], current_n=0,
                notes="empty current sample — drift not computable",
            ))
            continue

        if ref_type == "binary":
            current_proportion = float(np.mean(current_values))
            psi = population_stability_index(
                ref_stats["proportion_one"], current_proportion,
            )
            results.append(FeatureDriftResult(
                feature=feature, feature_type="binary", test="psi",
                statistic=psi, p_value=None,
                drifted=psi > psi_threshold,
                reference_n=ref_stats["n"], current_n=len(current_values),
            ))
        elif ref_type == "discrete":
            psi = population_stability_index_multibin(
                ref_stats["proportions"], current_values,
            )
            results.append(FeatureDriftResult(
                feature=feature, feature_type="discrete", test="psi",
                statistic=psi, p_value=None,
                drifted=psi > psi_threshold,
                reference_n=ref_stats["n"], current_n=len(current_values),
            ))
        else:  # continuous
            d_stat, p_value = ks_test_against_reference_summary(
                ref_stats, current_values,
            )
            results.append(FeatureDriftResult(
                feature=feature, feature_type="continuous", test="ks",
                statistic=d_stat, p_value=p_value,
                drifted=p_value < ks_pvalue_threshold,
                reference_n=ref_stats["n"], current_n=len(current_values),
            ))

    n_drifted = sum(1 for r in results if r.drifted)
    fraction = n_drifted / len(results) if results else 0.0

    fraction_rule = fraction > drift_feature_fraction
    severity_rule = any(
        (r.test == "psi" and r.statistic >= severe_psi_threshold)
        or (r.test == "ks" and r.statistic >= severe_ks_statistic)
        for r in results
    )

    return DriftReport(
        n_features=len(results),
        n_drifted=n_drifted,
        drifted_fraction=fraction,
        system_drift=fraction_rule or severity_rule,
        fraction_rule_tripped=fraction_rule,
        severity_rule_tripped=severity_rule,
        feature_results=results,
        reference_sha256=reference["source_sha256"],
    )
