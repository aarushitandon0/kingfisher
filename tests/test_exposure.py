"""engine.exposure - pure, hand-checkable geometry in a metric CRS."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from engine.exposure import (
    FEATURE_TYPES,
    PopulationGrid,
    classify_osm_tags,
    compute_exposure,
    element_geometry,
    osm_exposure_features,
    summarise_exposure,
)

REACH = pd.DataFrame({"reach_id": ["R1"], "geometry": [LineString([(0, 0), (1000, 0)])]})


def feats(*rows: tuple[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_type": [r[0] for r in rows],
            "osm_id": list(range(len(rows))),
            "geometry": [r[1] for r in rows],
        }
    )


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"amenity": "school"}, "school"),
        ({"amenity": "kindergarten"}, "kindergarten"),
        ({"leisure": "playground"}, "playground"),
        ({"leisure": "garden"}, "park"),
        ({"highway": "footway"}, "footway"),
        ({"highway": "cycleway"}, "cycleway"),
        ({"ford": "yes", "highway": "track"}, "water_access"),
        ({"ford": "no"}, None),
        ({"leisure": "slipway"}, "water_access"),
        ({"healthcare": "clinic"}, "healthcare"),
        ({"amenity": "hospital", "healthcare": "hospital"}, "healthcare"),
        ({"highway": "residential"}, None),
    ],
)
def test_osm_tag_classification(tags: dict[str, str], expected: str | None) -> None:
    assert classify_osm_tags(tags) == expected


def test_closed_area_way_is_a_polygon_and_a_path_is_a_line() -> None:
    ring = [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 1}, {"lat": 1, "lon": 1}, {"lat": 0, "lon": 0}]
    assert isinstance(
        element_geometry({"type": "way", "geometry": ring, "tags": {"amenity": "school"}}), Polygon
    )
    assert isinstance(
        element_geometry({"type": "way", "geometry": ring, "tags": {"highway": "footway"}}),
        LineString,
    )
    out = osm_exposure_features(
        [{"type": "way", "id": 1, "geometry": [], "tags": {"amenity": "school"}}]
    )
    assert out.empty and out.attrs["dropped_no_geometry"] == 1


def test_counts_and_nearest_distance_within_buffer() -> None:
    f = feats(
        ("school", Point(500, 180)),  # 180 m from the reach: in
        ("school", Point(500, 240)),  # 240 m: in
        ("school", Point(500, 260)),  # 260 m: out of a 250 m buffer
        ("footway", LineString([(-50, 10), (1050, 10)])),  # runs alongside: 10 m
    )
    out = compute_exposure(REACH, f, buffer_m=250).set_index("feature_type")
    assert out.loc["school", "count"] == 2
    assert out.loc["school", "nearest_distance_m"] == pytest.approx(180)
    assert out.loc["footway", "count"] == 1
    assert out.loc["footway", "nearest_distance_m"] == pytest.approx(10)


def test_every_feature_type_gets_a_row_and_zero_count_has_null_distance() -> None:
    out = compute_exposure(REACH, feats(("park", Point(0, 50))), buffer_m=250)
    assert list(out["feature_type"]) == list(FEATURE_TYPES)
    empty = out[out["feature_type"] == "school"].iloc[0]
    assert empty["count"] == 0 and pd.isna(empty["nearest_distance_m"]) and empty["geom"] is None


def test_buffer_is_configurable() -> None:
    f = feats(("playground", Point(500, 400)))
    small = compute_exposure(REACH, f, buffer_m=250).set_index("feature_type")
    big = compute_exposure(REACH, f, buffer_m=500).set_index("feature_type")
    assert small.loc["playground", "count"] == 0 and big.loc["playground", "count"] == 1


def _grid(values: np.ndarray) -> PopulationGrid:
    # 100 m cells, origin at (-500, 500), y decreasing - same orientation as a GeoTIFF
    return PopulationGrid(
        values=values, transform=(100.0, 0.0, -500.0, 0.0, -100.0, 500.0), crs="LOCAL"
    )


def test_population_sums_cells_whose_centre_is_in_the_buffer() -> None:
    grid = _grid(np.ones((10, 30)))
    out = compute_exposure(
        REACH, feats(), buffer_m=100, population=grid, to_population_crs=lambda x, y: (x, y)
    )
    pop = out[out["feature_type"] == "population"].iloc[0]
    # cell centres at y = +-50 lie inside the 100 m buffer along x in [-50, 1050]:
    # x centres -50..1050 in steps of 100 = 12 columns x 2 rows = 24 (ends are rounded caps)
    assert pop["count"] == 24


def test_population_is_null_when_buffer_touches_nodata_or_leaves_the_grid() -> None:
    values = np.ones((10, 30))
    values[5, 10] = np.nan
    out = compute_exposure(
        REACH,
        feats(),
        buffer_m=100,
        population=_grid(values),
        to_population_crs=lambda x, y: (x, y),
    )
    pop = out[out["feature_type"] == "population"].iloc[0]
    assert pd.isna(pop["count"]) and "NODATA" in pop["source"]
    tiny = _grid(np.ones((2, 2)))
    out = compute_exposure(
        REACH, feats(), buffer_m=100, population=tiny, to_population_crs=lambda x, y: (x, y)
    )
    pop = out[out["feature_type"] == "population"].iloc[0]
    assert pd.isna(pop["count"]) and "OUT_OF_GRID" in pop["source"]


def test_summary_is_counts_and_distances_only() -> None:
    rows = compute_exposure(REACH, feats(("school", Point(500, 180))), buffer_m=250)
    s = summarise_exposure(rows, "R1")
    assert s["features"]["school"] == {"count": 1, "nearest_distance_m": 180.0}
    assert s["buffer_m"] == 250
    assert not any(k in str(s).lower() for k in ("risk", "safe", "disease"))


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError):
        compute_exposure(REACH, feats(), buffer_m=0)
    with pytest.raises(ValueError, match="unknown"):
        compute_exposure(REACH, feats(("bakery", Point(0, 0))))
    assert math.isfinite(250.0)
