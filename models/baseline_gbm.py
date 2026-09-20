"""LightGBM quantile baseline (alpha 0.1/0.5/0.9), one model per horizon bucket,
pooled across reaches with reach_id as a categorical. SHAP attribution feeds the
alert engine. Build this first - it is the system that ships."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def train(city: str) -> None:
    raise NotImplementedError("models.baseline_gbm - Day 3")
