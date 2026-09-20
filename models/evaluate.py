"""Walk-forward evaluation: MAE/RMSE/CRPS by horizon against seasonal-naive and
climatology, reliability diagram, precision/recall vs incidents, lead-time
distribution, skill split by observability, Coimbra-to-Pune transfer.

Every metric is reported, favourable or not."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def evaluate(city: str) -> None:
    raise NotImplementedError("models.evaluate - Day 3")
