"""L2 - static catchment attributes: CLMS imperviousness and riparian zones (EU),
ESA WorldCover/GHSL elsewhere, VIIRS ALAN, OSM road density, GHS-POP.
Writes catchment_attributes. These are the scenario levers."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def build_static_attributes(city: str) -> None:
    raise NotImplementedError("pipeline.l2_static - Day 2")
