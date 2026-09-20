"""L0 - spatial backbone: OSM waterways to reaches, MERIT Hydro upstream catchments,
reach topology graph. Writes reaches + reach_topology.

Do not derive flow direction from a raw DEM; MERIT Hydro ships it precomputed."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def build_network(city: str) -> None:
    raise NotImplementedError("pipeline.l0_network - Day 1")
