"""L1 - observables: Sentinel Hub Statistical API per reach polygon, MNDWI/NDCI/
turbidity/NDVI, plus water_pixel_count. Writes observations.

Never download granules. Cache every response keyed by
(reach_id, date_range, evalscript_hash). Missing reach-dates are NULL with a
quality_flag, never zero."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def fetch_observations(city: str) -> None:
    raise NotImplementedError("pipeline.l1_satellite - Day 2")
