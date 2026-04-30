"""
End-to-end pipeline orchestrator.

run_pipeline() does: drift check -> retrain (if drift) -> evaluate vs champion ->
promote (if criteria met) -> log everything to GCS audit/.

Designed to be called from:
  - the FastAPI /run-pipeline endpoint (manual / on-demand)
  - a Cloud Run Job triggered by Cloud Scheduler (weekly)
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config import settings
from data_simulator import SimulatorConfig, persist, simulate
from drift import detect_drift, load_reference
from evaluation import decide_promotion, evaluate
from features import FEATURE_COLUMNS, MIN_ORDERS, MIN_TENURE_DAYS, build_features
from labeling import compute_signals
from alerter import (
    alert_drift_detected,
    alert_pipeline_error,
    alert_promotion,
    alert_promotion_rejected,
)
from registry import (
    get_production_version,
    list_versions,
    load_eval_report,
    load_model_state,
    load_scaler,
    next_version,
    register,
    set_production_version,
    write_audit_log,
    write_drift_report,
)
from retrain import load_dataset, retrain


@dataclass
class PipelineResult:
    """Structured outcome of one pipeline run. Persisted as audit log."""
    batch_id: str
    started_at: str
    completed_at: str
    drift_detected: bool
    drift_reason: str
    n_features_drifted: int
    retrained: bool
    challenger_version: str | None
    promoted: bool
    promotion_reasons: list[str]
    champion_f1: float | None
    challenger_f1: float | None
    f1_delta: float | None
    error: str | None = None


def _build_current_dataframe(reference_date: pd.Timestamp) -> pd.DataFrame:
    """Build the 'current' feature set for drift detection from real DB state."""
    df = load_dataset("real", reference_date)
    return df


def run_pipeline(
    inject_drift_preset: str | None = None,
    reference_date: pd.Timestamp | None = None,
    skip_retraining: bool = False,
) -> PipelineResult:
    """
    Single end-to-end pipeline run.

    Args:
        inject_drift_preset: if set, simulate a drift batch and union it into the
                             current dataset (for demo scenarios). None = real data only.
        reference_date: 'today' for feature computation. Defaults to 2026-04-29 to
                        match the v0 baseline.
        skip_retraining: if True, run drift check only; no retrain regardless of result.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    batch_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    if reference_date is None:
        reference_date = pd.Timestamp("2026-04-29")

    sim_batch_id: str | None = None

    try:
        # 1. (Optional) inject simulated drift batch into the DB.
        if inject_drift_preset is not None:
            sim_batch_id = f"{batch_id}_sim"
            sim_cfg = SimulatorConfig(
                n_customers=300, mode="drift", drift_preset=inject_drift_preset,
                seed=2026, reference_date=reference_date, batch_id=sim_batch_id,
            )
            cust, orders = simulate(sim_cfg)
            persist(cust, orders)

        # 2. Build the 'current' dataset for drift detection.
        if sim_batch_id is not None:
            df_current = load_dataset("union", reference_date, simulator_batch_id=sim_batch_id)
        else:
            df_current = load_dataset("real", reference_date)

        # 3. Drift detection vs frozen reference.
        ref = load_reference()
        drift = detect_drift(
            df_current[FEATURE_COLUMNS], ref,
            drift_feature_fraction=settings.DRIFT_FEATURE_FRACTION,
            severe_psi_threshold=settings.SEVERE_PSI_THRESHOLD,
            severe_ks_statistic=settings.SEVERE_KS_STATISTIC,
        )
        drift_dict = drift.to_dict()
        drift_dict["batch_id"] = batch_id
        write_drift_report(batch_id, drift_dict)
        if drift.system_drift:
            alert_drift_detected(batch_id, drift.n_drifted, "fraction" if drift.fraction_rule_tripped else "severity")

        drift_reason = (
            "fraction" if drift.fraction_rule_tripped
            else ("severity" if drift.severity_rule_tripped else "none")
        )

        # 4. Decide whether to retrain.
        if not drift.system_drift or skip_retraining:
            result = PipelineResult(
                batch_id=batch_id, started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                drift_detected=drift.system_drift,
                drift_reason=drift_reason,
                n_features_drifted=drift.n_drifted,
                retrained=False, challenger_version=None,
                promoted=False, promotion_reasons=["No drift, retraining skipped"],
                champion_f1=None, challenger_f1=None, f1_delta=None,
            )
            write_audit_log(batch_id, asdict(result))
            return result

        # 5. Retrain on the current dataset.
        chal = retrain(df_current, seed=42, threshold=0.4)

        # 6. Build holdout for fair comparison: hold out 30% of REAL customers
        #    (not the union — holdout must come from the same distribution as
        #    Day 2's training data so champion isn't unfairly evaluated on drift).
        real_df = load_dataset("real", reference_date)
        holdout = real_df.sample(frac=0.3, random_state=99).reset_index(drop=True)

        # 7. Evaluate champion + challenger on the same holdout.
        prod_version = get_production_version()
        if prod_version is None:
            # No champion yet — challenger auto-promotes.
            chal_eval = evaluate(chal.model_state_dict, chal.scaler, holdout, threshold=0.4)
            new_version = next_version()
            register(new_version, chal)
            set_production_version(new_version, reason="initial promotion (no prior champion)")
            result = PipelineResult(
                batch_id=batch_id, started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                drift_detected=True, drift_reason=drift_reason,
                n_features_drifted=drift.n_drifted,
                retrained=True, challenger_version=new_version,
                promoted=True,
                promotion_reasons=["First model — no champion to compare against"],
                champion_f1=None, challenger_f1=chal_eval.overall_f1, f1_delta=None,
            )
            write_audit_log(batch_id, asdict(result))
            return result

        champ_state = load_model_state(prod_version)
        champ_scaler = load_scaler(prod_version)
        champ_eval = evaluate(champ_state, champ_scaler, holdout, threshold=0.4)
        chal_eval = evaluate(chal.model_state_dict, chal.scaler, holdout, threshold=0.4)

        decision = decide_promotion(
            champ_eval, chal_eval,
            min_f1_improvement=settings.MIN_F1_IMPROVEMENT,
            max_segment_regression=settings.MAX_SEGMENT_REGRESSION,
        )

        # 8. If promoted, register + flip the production pointer.
        new_version: str | None = None
        if decision.promote:
            new_version = next_version()
            register(new_version, chal)
            set_production_version(new_version, reason=" | ".join(decision.reasons))
            alert_promotion(batch_id, new_version, chal_eval.overall_f1, decision.f1_delta)
        else:
            alert_promotion_rejected(batch_id, decision.reasons)

        result = PipelineResult(
            batch_id=batch_id, started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            drift_detected=True, drift_reason=drift_reason,
            n_features_drifted=drift.n_drifted,
            retrained=True, challenger_version=new_version,
            promoted=decision.promote,
            promotion_reasons=decision.reasons,
            champion_f1=champ_eval.overall_f1,
            challenger_f1=chal_eval.overall_f1,
            f1_delta=decision.f1_delta,
        )
        write_audit_log(batch_id, asdict(result))
        return result

    except Exception as e:
        result = PipelineResult(
            batch_id=batch_id, started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            drift_detected=False, drift_reason="error",
            n_features_drifted=0,
            retrained=False, challenger_version=None,
            promoted=False, promotion_reasons=["pipeline error"],
            champion_f1=None, challenger_f1=None, f1_delta=None,
            error=f"{type(e).__name__}: {e}",
        )
        try:
            write_audit_log(batch_id, asdict(result))
        except Exception:
            pass
        alert_pipeline_error(batch_id, f"{type(e).__name__}: {e}")
        raise
