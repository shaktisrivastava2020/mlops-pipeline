"""Unit tests for data_simulator: knobs do what they claim, output schemas hold."""
from __future__ import annotations
import pandas as pd
import pytest

from data_simulator import (
    DRIFT_PRESETS,
    SimulatorConfig,
    _resolved_params,
    simulate,
    summarize,
)


REF_DATE = pd.Timestamp("2026-04-29")


class TestSimulatorContract:
    def test_baseline_returns_two_dataframes(self):
        cust, orders = simulate(SimulatorConfig(n_customers=50, mode="baseline", seed=1, reference_date=REF_DATE))
        assert isinstance(cust, pd.DataFrame)
        assert isinstance(orders, pd.DataFrame)
        assert len(cust) == 50
        assert len(orders) > 0

    def test_required_customer_columns(self):
        cust, _ = simulate(SimulatorConfig(n_customers=20, mode="baseline", seed=1, reference_date=REF_DATE))
        for col in ["_local_idx", "customer_tier", "join_date", "batch_id"]:
            assert col in cust.columns

    def test_required_order_columns(self):
        _, orders = simulate(SimulatorConfig(n_customers=20, mode="baseline", seed=1, reference_date=REF_DATE))
        for col in ["_local_cust_idx", "product_id", "quantity", "order_amount",
                    "payment_method", "order_status", "order_date", "batch_id"]:
            assert col in orders.columns

    def test_seed_is_deterministic(self):
        cust1, ord1 = simulate(SimulatorConfig(n_customers=30, seed=99, reference_date=REF_DATE, batch_id="b"))
        cust2, ord2 = simulate(SimulatorConfig(n_customers=30, seed=99, reference_date=REF_DATE, batch_id="b"))
        pd.testing.assert_frame_equal(cust1, cust2)
        pd.testing.assert_frame_equal(ord1, ord2)

    def test_drift_mode_requires_preset(self):
        with pytest.raises(ValueError, match="drift_preset"):
            simulate(SimulatorConfig(mode="drift", drift_preset=None, n_customers=10, reference_date=REF_DATE))

    def test_unknown_drift_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown drift_preset"):
            simulate(SimulatorConfig(mode="drift", drift_preset="not_a_preset", n_customers=10, reference_date=REF_DATE))


class TestDriftPresets:
    """Each preset must measurably move the feature it claims to move."""

    def test_high_value_shift_increases_order_amount(self):
        _, ord_baseline = simulate(SimulatorConfig(n_customers=200, mode="baseline", seed=42, reference_date=REF_DATE))
        _, ord_drift = simulate(SimulatorConfig(n_customers=200, mode="drift", drift_preset="high_value_shift", seed=42, reference_date=REF_DATE))
        assert ord_drift["order_amount"].mean() > ord_baseline["order_amount"].mean() * 1.3

    def test_engagement_decline_reduces_orders_per_customer(self):
        _, ord_baseline = simulate(SimulatorConfig(n_customers=200, mode="baseline", seed=42, reference_date=REF_DATE))
        _, ord_drift = simulate(SimulatorConfig(n_customers=200, mode="drift", drift_preset="engagement_decline", seed=42, reference_date=REF_DATE))
        baseline_per = len(ord_baseline) / 200
        drift_per = len(ord_drift) / 200
        assert drift_per < baseline_per * 0.7

    def test_premium_surge_shifts_tier_distribution(self):
        cust_baseline, _ = simulate(SimulatorConfig(n_customers=300, mode="baseline", seed=42, reference_date=REF_DATE))
        cust_drift, _ = simulate(SimulatorConfig(n_customers=300, mode="drift", drift_preset="premium_surge", seed=42, reference_date=REF_DATE))
        gold_baseline = (cust_baseline["customer_tier"] == "Gold").mean()
        gold_drift = (cust_drift["customer_tier"] == "Gold").mean()
        assert gold_drift > gold_baseline + 0.10

    def test_new_payment_method_introduces_new_value(self):
        _, ord_baseline = simulate(SimulatorConfig(n_customers=300, mode="baseline", seed=42, reference_date=REF_DATE))
        _, ord_drift = simulate(SimulatorConfig(n_customers=300, mode="drift", drift_preset="new_payment_method", seed=42, reference_date=REF_DATE))
        assert "BNPL" not in ord_baseline["payment_method"].unique()
        assert "BNPL" in ord_drift["payment_method"].unique()


class TestSummarize:
    def test_summarize_returns_expected_keys(self):
        cust, orders = simulate(SimulatorConfig(n_customers=50, seed=1, reference_date=REF_DATE))
        summary = summarize(cust, orders)
        for k in ["n_customers", "n_orders", "mean_orders_per_customer", "mean_order_amount",
                  "tier_proportions", "weekday_ratio", "unique_payment_methods"]:
            assert k in summary
