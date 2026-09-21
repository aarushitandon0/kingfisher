"""Exposure pathways - pure functions only. No I/O, no DB, no network.

For every reach: what people-facing features sit within a buffer of its centreline
(default 250 m, city config `exposure.buffer_m`), how many, and how close the nearest one
is; plus the GHS-POP population inside the same buffer. Pathways and proximity only - no
health outcome prediction and no safe/unsafe declaration (CLAUDE.md #8).

Feature types and the OSM tags that define them (DATA_SOURCES.md 1.7):

  school         amenity=school
  kindergarten   amenity=kindergarten
  playground     leisure=playground
  park           leisure=park | leisure=garden
  footway        highway=footway (includes mapped sidewalks - OSM does not separate a
                 riverside path from a pavement reliably)
  cycleway       highway=cycleway
  water_access   leisure=swimming_area | leisure=slipway | waterway=access_point |
                 ford=<anything but "no">
  healthcare     amenity=hospital | amenity=clinic | healthcare=hospital | healthcare=clinic

`count` is the number of distinct OSM elements of that type intersecting the buffer.
0 is a real count of what OSM maps, not a fill; OSM completeness is a stated limitation.
`nearest_distance_m` is the distance from the centreline to the nearest counted element
(0 if it touches the reach) and is NULL when count is 0 - nothing is searched beyond the
buffer. Population is feature_type "population": `count` is the rounded GHS-POP sum over
cells whose centre falls in the buffer, NULL (flag in `source`) if the buffer leaves the
grid or covers a nodata cell.

Callers load the data (pipeline/exposure_features.py) and write `exposure_features`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

FEATURE_TYPES: tuple[str, ...] = (
    "school",
    "kindergarten",
    "playground",
    "park",
    "footway",
    "cycleway",
    "water_access",
    "healthcare",
)
POPULATION = "population"
OSM_SOURCE = "OpenStreetMap (ODbL)"
DEFAULT_BUFFER_M = 250.0

# Tags that make a closed way an area rather than a ring-shaped line.
_AREA_KEYS = ("amenity", "leisure", "landuse", "building", "healthcare")


def classify_osm_tags(tags: Mapping[str, str]) -> str | None:
    """OSM tags -> one exposure feature type, or None. First match wins, in the order of
    FEATURE_TYPES, so an element is never counted under two types."""
    amenity = tags.get("amenity")
    leisure = tags.get("leisure")
    highway = tags.get("highway")
    healthcare = tags.get("healthcare")
    if amenity == "school":
        return "school"
    if amenity == "kindergarten":
        return "kindergarten"
    if leisure == "playground":
        return "playground"
    if leisure in ("park", "garden"):
        return "park"
    if highway == "footway":
        return "footway"
    if highway == "cycleway":
        return "cycleway"
    if (
        leisure in ("swimming_area", "slipway")
        or tags.get("waterway") == "access_point"
        or tags.get("ford", "no") != "no"
    ):
        return "water_access"
    if amenity in ("hospital", "clinic") or healthcare in ("hospital", "clinic"):
        return "healthcare"
    return None


def element_geometry(element: Mapping[str, Any]) -> BaseGeometry | None:
    """One Overpass element (`out geom`) -> WGS84 geometry. None if it has no usable
    coordinates (counted by the caller as dropped)."""
    if element.get("type") == "node":
        if "lat" not in element or "lon" not in element:
            return None
        return Point(float(element["lon"]), float(element["lat"]))
    coords = [(float(p["lon"]), float(p["lat"])) for p in element.get("geometry") or [] if p]
    if len(coords) < 2:
        return Point(coords[0]) if coords else None
    tags = element.get("tags", {})
    closed = coords[0] == coords[-1] and len(coords) >= 4
    if closed and any(k in tags for k in _AREA_KEYS):
        poly = Polygon(coords)
        return poly if poly.is_valid else poly.buffer(0)
    return LineString(coords)


def osm_exposure_features(elements: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Overpass elements -> DataFrame(osm_type, osm_id, feature_type, geometry[WGS84]).
    Elements that are not exposure features are skipped; ones without geometry are
    returned in `.attrs["dropped_no_geometry"]`."""
    rows: list[dict[str, Any]] = []
    dropped = 0
    for el in elements:
        ftype = classify_osm_tags(el.get("tags") or {})
        if ftype is None:
            continue
        geom = element_geometry(el)
        if geom is None or geom.is_empty:
            dropped += 1
            continue
        rows.append(
            {
                "osm_type": el.get("type"),
                "osm_id": el.get("id"),
                "feature_type": ftype,
                "geometry": geom,
            }
        )
    out = pd.DataFrame(rows, columns=["osm_type", "osm_id", "feature_type", "geometry"])
    out.attrs["dropped_no_geometry"] = dropped
    return out


@dataclass(frozen=True)
class PopulationGrid:
    """A preloaded population raster: values[row, col] (NaN = nodata), the affine
    transform (a, b, c, d, e, f) as in rasterio / GDAL, and its CRS."""

    values: npt.NDArray[np.float64]
    transform: tuple[float, float, float, float, float, float]
    crs: str
    source: str = "GHS-POP"

    def bounds(self) -> tuple[float, float, float, float]:
        a, _, c, _, e, f = self.transform
        rows, cols = self.values.shape
        xs = (c, c + a * cols)
        ys = (f, f + e * rows)
        return min(xs), min(ys), max(xs), max(ys)


def population_in(area: BaseGeometry, grid: PopulationGrid) -> tuple[int | None, str]:
    """Sum of cells whose centre lies in `area` (area in the grid's CRS).
    Returns (population or None, provenance/flag)."""
    minx, miny, maxx, maxy = grid.bounds()
    if not box(minx, miny, maxx, maxy).contains(area):
        return None, f"{grid.source}: OUT_OF_GRID"
    a, _, c, _, e, f = grid.transform
    x0, y0, x1, y1 = area.bounds
    col_lo = max(0, int(np.floor((x0 - c) / a)))
    col_hi = min(grid.values.shape[1], int(np.ceil((x1 - c) / a)) + 1)
    row_lo = max(0, int(np.floor((y1 - f) / e)))
    row_hi = min(grid.values.shape[0], int(np.ceil((y0 - f) / e)) + 1)
    total = 0.0
    for r in range(row_lo, row_hi):
        cy = f + e * (r + 0.5)
        for col in range(col_lo, col_hi):
            cx = c + a * (col + 0.5)
            if area.contains(Point(cx, cy)):
                v = grid.values[r, col]
                if np.isnan(v):
                    return None, f"{grid.source}: NODATA_IN_BUFFER"
                total += float(v)
    return int(round(total)), grid.source


def compute_exposure(
    reaches: pd.DataFrame,
    features: pd.DataFrame,
    *,
    buffer_m: float = DEFAULT_BUFFER_M,
    population: PopulationGrid | None = None,
    to_population_crs: Any = None,
) -> pd.DataFrame:
    """Exposure rows for every reach x feature type (+ population).

    reaches   reach_id, geometry - centrelines in a METRIC CRS
    features  feature_type, osm_id, geometry - same metric CRS (osm_exposure_features
              output, reprojected by the caller)
    population, to_population_crs
              optional grid, and a pure (x, y) -> (x, y) function taking the metric CRS
              to the grid's CRS (e.g. a pyproj Transformer's .transform)
    Returns reach_id, feature_type, count, nearest_distance_m, geom (the nearest counted
    feature, metric CRS, None when count is 0), buffer_m, source.
    """
    if buffer_m <= 0:
        raise ValueError(f"buffer_m must be positive, got {buffer_m}")
    if population is not None and to_population_crs is None:
        raise ValueError("population grid given without a metric -> grid CRS transform")
    unknown = set(features["feature_type"]) - set(FEATURE_TYPES)
    if unknown:
        raise ValueError(f"unknown feature types {sorted(unknown)}")

    by_type: dict[str, tuple[list[BaseGeometry], STRtree | None]] = {}
    for ftype in FEATURE_TYPES:
        geoms = list(features.loc[features["feature_type"] == ftype, "geometry"])
        by_type[ftype] = (geoms, STRtree(geoms) if geoms else None)

    rows: list[dict[str, Any]] = []
    for rid, line in zip(reaches["reach_id"], reaches["geometry"], strict=True):
        area = line.buffer(buffer_m)
        for ftype in FEATURE_TYPES:
            geoms, tree = by_type[ftype]
            hits = (
                [] if tree is None else [int(i) for i in tree.query(area, predicate="intersects")]
            )
            if hits:
                dists = [(line.distance(geoms[i]), i) for i in hits]
                d, i = min(dists)
                nearest, geom = float(d), geoms[i]
            else:
                nearest, geom = None, None
            rows.append(
                {
                    "reach_id": rid,
                    "feature_type": ftype,
                    "count": len(hits),
                    "nearest_distance_m": nearest,
                    "geom": geom,
                    "buffer_m": buffer_m,
                    "source": OSM_SOURCE,
                }
            )
        if population is not None:
            pop, source = population_in(shapely_transform(to_population_crs, area), population)
            rows.append(
                {
                    "reach_id": rid,
                    "feature_type": POPULATION,
                    "count": pop,
                    "nearest_distance_m": None,
                    "geom": None,
                    "buffer_m": buffer_m,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def summarise_exposure(rows: pd.DataFrame, reach_id: str) -> dict[str, Any]:
    """One reach's exposure rows -> the dict an alert carries. Counts and distances
    only; nothing here grades or interprets them."""
    sub = rows[rows["reach_id"] == reach_id]
    if sub.empty:
        raise LookupError(f"no exposure rows for {reach_id}")
    out: dict[str, Any] = {"buffer_m": float(sub["buffer_m"].iloc[0]), "features": {}}
    for r in sub.itertuples(index=False):
        count = None if pd.isna(r.count) else int(r.count)
        near = (
            None
            if r.nearest_distance_m is None or pd.isna(r.nearest_distance_m)
            else round(float(r.nearest_distance_m), 1)
        )
        if r.feature_type == POPULATION:
            out["population"] = count
            out["population_source"] = r.source
        else:
            out["features"][r.feature_type] = {"count": count, "nearest_distance_m": near}
    return out
