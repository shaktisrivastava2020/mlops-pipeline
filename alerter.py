"""Slack webhook alerter — fire-and-forget notifications for pipeline events."""
from __future__ import annotations
from typing import Any
import requests

from config import settings


def _send(text: str, blocks: list[dict] | None = None) -> bool:
    """Send a message to Slack. Returns True on success, False otherwise.
    Always swallows errors — alerts must never break the pipeline."""
    if not settings.SLACK_WEBHOOK_URL:
        return False
    payload: dict[str, Any] = {"text": text}
    if blocks:
        payload["blocks"] = blocks
    try:
        r = requests.post(settings.SLACK_WEBHOOK_URL, json=payload, timeout=5)
        return 200 <= r.status_code < 300
    except Exception:
        return False


def alert_drift_detected(batch_id: str, n_drifted: int, reason: str) -> bool:
    return _send(
        f":warning: Drift detected in batch `{batch_id}` — {n_drifted} features drifted ({reason})"
    )


def alert_promotion(batch_id: str, version: str, f1: float, delta: float) -> bool:
    return _send(
        f":white_check_mark: Promoted `{version}` from batch `{batch_id}` — "
        f"F1 {f1:.4f} (Δ {delta:+.4f})"
    )


def alert_promotion_rejected(batch_id: str, reasons: list[str]) -> bool:
    body = "\n".join(f"• {r}" for r in reasons)
    return _send(f":no_entry: Promotion rejected for batch `{batch_id}`\n{body}")


def alert_pipeline_error(batch_id: str, error: str) -> bool:
    return _send(f":rotating_light: Pipeline error in batch `{batch_id}` — {error}")
