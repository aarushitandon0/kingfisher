"""Anomaly detection. Two signals reported separately: burstiness spike (routing-
derived) and forecast residual outside P10-P90. Scored against
data/reference/incidents.csv."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def score_anomalies(city: str) -> None:
    raise NotImplementedError("models.anomaly - Day 5")
