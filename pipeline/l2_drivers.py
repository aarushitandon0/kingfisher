"""L2 - drivers: Open-Meteo archive (training) and forecast (live), aggregated over
each reach's upstream catchment polygon, then feature-engineered into API,
antecedent dry days, first-flush index and the temperature/low-flow term.
Writes drivers_daily."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def build_drivers(city: str) -> None:
    raise NotImplementedError("pipeline.l2_drivers - Day 2")
