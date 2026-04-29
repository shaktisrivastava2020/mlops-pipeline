"""
Unit tests for the drift detection module.

Coverage strategy:
  - PURE FUNCTIONS: PSI (binary + multibin) and KS — test math correctness on
    constructed inputs where the answer is known a priori.
  - DISPATCHER (detect_drift): test that the right test gets routed for each
    feature type, and that system-level decisions follow the threshold rule.
  - EDGE CASES: empty input, missing feature, new bins, identical distributions,
    boundary values (0, 1, all-same).

We don't test scipy's KS implementation — that's not our code. We test that
WE call it correctly and interpret its result correctly.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift import (
    DriftReport,
    FeatureDriftResult,
    detect_drift,
    ks_test_against_reference_summary,
    load_reference,
    population_stability_index,
    population_stability_index_multibin,
)


# ----------------------- BINARY PSI -----------------------

class TestBinaryPSI:
    def test_identical_proportions_yields_zero(self):
        """PSI should be ~0 when reference and current are identical."""
        psi = population_stability_index(0.5, 0.5)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_extreme_shift_yields_large_psi(self):
        """A shift from 10% to 90% should yield a large PSI (>> threshold)."""
        psi = population_stability_index(0.1, 0.9)
        assert psi > 0.20  # well above the drift threshold

    def test_symmetric_in_direction(self):
        """PSI(a,b) and PSI(b,a) should produce the same magnitude."""
        psi_ab = population_stability_index(0.3, 0.7)
        psi_ba = population_stability_index(0.7, 0.3)
        assert psi_ab == pytest.approx(psi_ba, rel=1e-9)

    def test_proportion_out_of_range_raises(self):
        """Inputs outside [0,1] are programmer errors, not data issues."""
        with pytest.raises(ValueError):
            population_stability_index(-0.1, 0.5)
        with pytest.raises(ValueError):
            population_stability_index(0.5, 1.1)

    def test_zero_proportion_does_not_blow_up(self):
        """Epsilon guard must prevent log(0) when a bin is empty."""
        psi = population_stability_index(0.0, 0.5)
        assert np.isfinite(psi)


# ----------------------- DISCRETE (MULTIBIN) PSI -----------------------

class TestMultibinPSI:
    def test_identical_distributions_yields_zero(self):
        ref = {"1": 0.5, "2": 0.3, "3": 0.2}
        current = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3])  # 50/30/20
        psi = population_stability_index_multibin(ref, current)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_new_bin_in_current_registers_as_drift(self):
        """A value not seen in reference must contribute to PSI."""
        ref = {"1": 0.5, "2": 0.5}
        current = np.array([1, 1, 2, 2, 99, 99, 99, 99])  # value 99 is new
        psi = population_stability_index_multibin(ref, current)
        assert psi > 0.20  # new value is drift

    def test_missing_bin_in_current_registers_as_drift(self):
        """A reference bin that vanishes must contribute to PSI."""
        ref = {"1": 0.5, "2": 0.5}
        current = np.array([1, 1, 1, 1])  # value 2 has vanished
        psi = population_stability_index_multibin(ref, current)
        assert psi > 0.20

    def test_empty_current_raises(self):
        """Cannot compute proportions on no data."""
        with pytest.raises(ValueError):
            population_stability_index_multibin({"1": 1.0}, np.array([]))


# ----------------------- KS RECONSTRUCTION -----------------------

class TestKSReconstruction:
    def test_identical_summary_low_d_statistic(self):
        """When current data matches the reference quantiles, D should be small."""
        ref = {"min": 0.0, "p25": 25.0, "p50": 50.0, "p75": 75.0,
               "p95": 95.0, "p99": 99.0, "max": 100.0}
        current = np.linspace(0, 100, 200)
        d, p = ks_test_against_reference_summary(ref, current)
        assert d < 0.10
        assert p > 0.05

    def test_completely_shifted_distribution_high_d_statistic(self):
        """Current data shifted way past the reference range -> D near 1."""
        ref = {"min": 0.0, "p25": 25.0, "p50": 50.0, "p75": 75.0,
               "p95": 95.0, "p99": 99.0, "max": 100.0}
        current = np.linspace(500, 600, 200)  # entirely outside reference range
        d, p = ks_test_against_reference_summary(ref, current)
        assert d > 0.9
        assert p < 0.05


# ----------------------- DISPATCHER -----------------------

@pytest.fixture
def synthetic_reference(tmp_path: Path) -> dict:
    """A minimal 3-feature reference covering all three feature types."""
    return {
        "schema_version": "1.1",
        "source_sha256": "deadbeef" * 8,
        "feature_columns": ["age", "tier_premium", "n_logins"],
        "feature_stats": {
            "age": {
                "type": "continuous", "n": 1000,
                "mean": 40.0, "std": 10.0,
                "min": 18.0, "p25": 32.0, "p50": 40.0,
                "p75": 48.0, "p95": 58.0, "p99": 65.0, "max": 70.0,
            },
            "tier_premium": {
                "type": "binary", "n": 1000,
                "proportion_one": 0.30,
            },
            "n_logins": {
                "type": "discrete", "n": 1000, "n_unique": 4,
                "proportions": {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1},
            },
        },
    }


class TestDispatcher:
    def test_no_drift_when_current_matches_reference(self, synthetic_reference):
        """All three feature types: identical-shaped current data -> no drift."""
        rng = np.random.default_rng(42)
        n = 500
        current = pd.DataFrame({
            "age": rng.normal(40, 10, n).clip(18, 70),
            "tier_premium": rng.binomial(1, 0.30, n),
            "n_logins": rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        })
        report = detect_drift(current, synthetic_reference)
        assert report.system_drift is False
        assert report.n_drifted <= 1  # allow at most 1 sampling-noise false positive

    def test_severe_drift_in_majority_triggers_system_drift(self, synthetic_reference):
        """If most features visibly shift, system_drift should fire."""
        rng = np.random.default_rng(42)
        n = 500
        current = pd.DataFrame({
            "age": rng.normal(80, 5, n),                          # shifted
            "tier_premium": rng.binomial(1, 0.90, n),             # shifted
            "n_logins": rng.choice([5, 6, 7], size=n),            # all-new bins
        })
        report = detect_drift(current, synthetic_reference)
        assert report.system_drift is True
        assert report.n_drifted == 3

    def test_missing_feature_raises(self, synthetic_reference):
        """Schema mismatch must fail loudly, not silently."""
        current = pd.DataFrame({"age": [40.0], "tier_premium": [1]})  # missing n_logins
        with pytest.raises(ValueError, match="missing reference features"):
            detect_drift(current, synthetic_reference)

    def test_empty_current_for_one_feature_yields_notes(self, synthetic_reference):
        """Empty values for one feature should report 'not computable', not crash."""
        current = pd.DataFrame({
            "age": [40.0, 41.0, 42.0],
            "tier_premium": [1, 0, 1],
            "n_logins": [np.nan, np.nan, np.nan],  # all-NaN -> empty after dropna
        })
        report = detect_drift(current, synthetic_reference)
        n_logins_result = next(r for r in report.feature_results if r.feature == "n_logins")
        assert n_logins_result.test == "none"
        assert n_logins_result.drifted is False
        assert "not computable" in n_logins_result.notes

    def test_report_carries_reference_provenance(self, synthetic_reference):
        """Audit trail: every report must carry the SHA of its reference."""
        current = pd.DataFrame({"age": [40.0], "tier_premium": [1], "n_logins": [0]})
        report = detect_drift(current, synthetic_reference)
        assert report.reference_sha256 == synthetic_reference["source_sha256"]


# ----------------------- INTEGRATION: REAL REFERENCE + SELF -----------------------

class TestAgainstRealReference:
    def test_training_data_against_its_own_reference_no_system_drift(self):
        """The dataset that produced the reference must not trigger system-level drift."""
        ref = load_reference("reference_stats.json")
        training = pd.read_csv("data/training.csv")
        report = detect_drift(training, ref)
        # System-level must be False; a single feature may be a documented false positive.
        assert report.system_drift is False
        assert report.n_drifted <= 2
