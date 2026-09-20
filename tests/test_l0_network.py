"""L0 backbone tests - the parts that must be right before anything is built on them.

No network, no database, no rasters: every function under test here is pure. The
fixtures are hand-built toy networks whose correct answers can be read off by eye.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LineString

from pipeline.l0_network import (
    HydroGrid,
    Link,
    OsmWay,
    Reach,
    _piece_count,
    build_topology,
    compute_strahler,
    filter_by_strahler,
    length_distribution,
    merge_pass_through_links,
    segment_links,
    snap_to_stream,
    split_at_confluences,
    validate_catchment_monotonicity,
)
from scripts.check_observability import (
    ascii_histogram,
    dates_from_payload,
    summarise_reach,
)

# A Y: two headwaters (1->3, 2->3) joining at node 3, one trunk (3->4).
#
#   n1 \
#        n3 ---> n4
#   n2 /
Y_WAYS = [
    OsmWay(1, "stream", "West Fork", (1, 3), ((0.0, 0.0), (0.01, 0.0))),
    OsmWay(2, "stream", "East Fork", (2, 3), ((0.0, 0.01), (0.01, 0.0))),
    OsmWay(3, "stream", "Trunk", (3, 4), ((0.01, 0.0), (0.02, 0.0))),
]

CFG = {
    "reach_id_prefix": "TST",
    "bbox": {"min_lon": -1.0, "min_lat": -1.0, "max_lon": 1.0, "max_lat": 1.0},
    "crs": {"storage": "EPSG:4326", "metric": "EPSG:3763"},
    "network": {
        "waterway_tags": ["river", "stream", "canal"],
        "reach_length_m": {"target": 350, "min": 200, "max": 500},
    },
}


def _links(ways: list[OsmWay]) -> list[Link]:
    return merge_pass_through_links(split_at_confluences(ways))


# ---------------------------------------------------------------------------
# network construction
# ---------------------------------------------------------------------------
def test_ways_split_at_a_shared_node() -> None:
    """A vertex shared by two ways is a confluence and must become a network node."""
    way_a = OsmWay(1, "stream", None, (1, 2, 3), ((0.0, 0.0), (0.01, 0.0), (0.02, 0.0)))
    way_b = OsmWay(2, "stream", None, (9, 2), ((0.01, 0.01), (0.01, 0.0)))
    links = split_at_confluences([way_a, way_b])
    assert len(links) == 3  # way_a cut in two at node 2, plus way_b
    assert {(link.u_node, link.v_node) for link in links} == {(1, 2), (2, 3), (9, 2)}


def test_pass_through_way_boundaries_are_merged() -> None:
    """Two ways meeting end-to-end are one channel, not two reaches' worth of slivers."""
    way_a = OsmWay(1, "stream", "Ribeira", (1, 2), ((0.0, 0.0), (0.01, 0.0)))
    way_b = OsmWay(2, "stream", "Ribeira", (2, 3), ((0.01, 0.0), (0.02, 0.0)))
    merged = merge_pass_through_links(split_at_confluences([way_a, way_b]))
    assert len(merged) == 1
    assert merged[0].u_node == 1
    assert merged[0].v_node == 3
    assert merged[0].way_ids == [1, 2]


def test_merge_stops_at_a_confluence() -> None:
    """A node with two inflows is a junction; the channel above and below stays split."""
    merged = _links(Y_WAYS)
    assert len(merged) == 3


def test_merge_does_not_cross_a_channel_class_change() -> None:
    stream = OsmWay(1, "stream", None, (1, 2), ((0.0, 0.0), (0.01, 0.0)))
    canal = OsmWay(2, "canal", None, (2, 3), ((0.01, 0.0), (0.02, 0.0)))
    assert len(merge_pass_through_links(split_at_confluences([stream, canal]))) == 2


# ---------------------------------------------------------------------------
# Strahler
# ---------------------------------------------------------------------------
def test_strahler_two_equal_tributaries_increment_the_trunk() -> None:
    links = _links(Y_WAYS)
    compute_strahler(links)
    by_nodes = {(link.u_node, link.v_node): link.strahler for link in links}
    assert by_nodes[(1, 3)] == 1
    assert by_nodes[(2, 3)] == 1
    assert by_nodes[(3, 4)] == 2


def test_strahler_unequal_tributaries_do_not_increment() -> None:
    """order-2 joined by order-1 stays order-2 - that is the whole Strahler rule."""
    ways = [
        *Y_WAYS,
        OsmWay(4, "stream", None, (5, 4), ((0.02, 0.01), (0.02, 0.0))),
        OsmWay(5, "stream", None, (4, 6), ((0.02, 0.0), (0.03, 0.0))),
    ]
    links = _links(ways)
    compute_strahler(links)
    by_nodes = {(link.u_node, link.v_node): link.strahler for link in links}
    assert by_nodes[(3, 4)] == 2
    assert by_nodes[(5, 4)] == 1
    assert by_nodes[(4, 6)] == 2


def test_strahler_survives_a_cycle() -> None:
    """A mis-digitised loop must not hang the ordering or silently vanish."""
    ways = [
        OsmWay(1, "stream", None, (1, 2), ((0.0, 0.0), (0.01, 0.0))),
        OsmWay(2, "stream", None, (2, 3), ((0.01, 0.0), (0.02, 0.0))),
        OsmWay(3, "stream", None, (3, 1), ((0.02, 0.0), (0.0, 0.0))),
    ]
    links = _links(ways)
    summary = compute_strahler(links)
    assert summary["cycle_edges_removed"] >= 1
    assert all(link.strahler is not None for link in links)


def test_filter_by_strahler_refuses_to_empty_the_network() -> None:
    links = _links(Y_WAYS)
    compute_strahler(links)
    assert len(filter_by_strahler(links, 2)) == 1
    with pytest.raises(RuntimeError):
        filter_by_strahler(links, 9)


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("length_m", "expected"),
    [
        (120.0, 1),  # shorter than min: kept whole, flagged, never merged away
        (480.0, 1),
        (700.0, 2),
        (1400.0, 4),
        (5000.0, 14),
    ],
)
def test_piece_count_respects_the_configured_bounds(length_m: float, expected: int) -> None:
    assert _piece_count(length_m, 350, 200, 500) == expected


@pytest.mark.parametrize("length_m", [600.0, 900.0, 1250.0, 2000.0, 4321.0])
def test_pieces_of_a_long_link_stay_inside_min_and_max(length_m: float) -> None:
    n = _piece_count(length_m, 350, 200, 500)
    assert 200 <= length_m / n <= 500


def test_segmentation_assigns_contiguous_prefixed_ids_ordered_by_strahler() -> None:
    links = _links(Y_WAYS)
    compute_strahler(links)
    reaches = segment_links(links, CFG, "test")
    assert [r.reach_id for r in reaches] == [f"TST-{i:04d}" for i in range(1, len(reaches) + 1)]
    # The trunk (order 2) sorts first, so TST-0001 is never an arbitrary headwater ditch.
    assert reaches[0].strahler_order == max(r.strahler_order for r in reaches)


def test_segmentation_is_deterministic() -> None:
    first = segment_links(_links(Y_WAYS), CFG, "test")
    second = segment_links(_links(Y_WAYS), CFG, "test")
    assert [(r.reach_id, round(r.length_m, 6)) for r in first] == [
        (r.reach_id, round(r.length_m, 6)) for r in second
    ]


def test_segmentation_drops_reaches_outside_the_bbox() -> None:
    cfg = {**CFG, "bbox": {"min_lon": 5.0, "min_lat": 5.0, "max_lon": 6.0, "max_lat": 6.0}}
    with pytest.raises(RuntimeError, match="no reaches inside the bbox"):
        segment_links(_links(Y_WAYS), cfg, "test")


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------
def _reach(reach_id: str, link: tuple[int, int], piece: int, n: int, **kw) -> Reach:
    return Reach(
        reach_id=reach_id,
        city="test",
        name=None,
        geom=LineString([(0.0, 0.0), (0.001, 0.0)]),
        length_m=300.0,
        strahler_order=1,
        waterway="stream",
        osm_way_id=link[0],
        link_key=link,
        piece=piece,
        n_pieces=n,
        u_node=kw.get("u_node"),
        v_node=kw.get("v_node"),
    )


def test_topology_chains_pieces_of_one_link_in_flow_order() -> None:
    reaches = [
        _reach("A-0001", (1, 0), 0, 3, u_node=10),
        _reach("A-0002", (1, 0), 1, 3),
        _reach("A-0003", (1, 0), 2, 3, v_node=11),
    ]
    assert build_topology(reaches) == [("A-0001", "A-0002"), ("A-0002", "A-0003")]


def test_topology_joins_two_tributaries_to_one_trunk_at_a_confluence() -> None:
    reaches = [
        _reach("A-0001", (1, 0), 0, 1, u_node=1, v_node=3),
        _reach("A-0002", (2, 0), 0, 1, u_node=2, v_node=3),
        _reach("A-0003", (3, 0), 0, 1, u_node=3, v_node=4),
    ]
    assert build_topology(reaches) == [("A-0001", "A-0003"), ("A-0002", "A-0003")]


def test_topology_has_no_self_edges() -> None:
    reaches = [_reach("A-0001", (1, 0), 0, 1, u_node=7, v_node=7)]
    assert build_topology(reaches) == []


def test_topology_of_the_real_y_network_is_a_single_confluence() -> None:
    links = _links(Y_WAYS)
    compute_strahler(links)
    reaches = segment_links(links, CFG, "test")
    edges = build_topology(reaches)
    downstream = {u: d for u, d in edges}
    assert len(downstream) == len(edges)  # no reach drains two ways at once here
    trunk = reaches[0].reach_id
    assert sum(1 for _, d in edges if d == trunk) == 2


# ---------------------------------------------------------------------------
# catchment validation
# ---------------------------------------------------------------------------
def _with_area(reach_id: str, area: float) -> Reach:
    reach = _reach(reach_id, (1, 0), 0, 1)
    reach.catchment_area_km2 = area
    reach.snap_distance_m = 12.0
    return reach


def test_monotonicity_passes_when_area_grows_downstream() -> None:
    reaches = [_with_area("A-0001", 2.0), _with_area("A-0002", 5.0)]
    assert validate_catchment_monotonicity(reaches, [("A-0001", "A-0002")]) == []


def test_monotonicity_flags_a_shrinking_catchment_as_a_bad_snap() -> None:
    reaches = [_with_area("A-0001", 1788.0), _with_area("A-0002", 0.03)]
    violations = validate_catchment_monotonicity(reaches, [("A-0001", "A-0002")])
    assert len(violations) == 1
    assert violations[0]["shrinkage_km2"] == pytest.approx(1787.97, abs=0.01)
    assert "MONOTONICITY_VIOLATION" in reaches[0].flags
    assert "MONOTONICITY_VIOLATION" in reaches[1].flags


def test_monotonicity_tolerates_identical_areas_from_a_shared_snap_cell() -> None:
    reaches = [_with_area("A-0001", 4.0), _with_area("A-0002", 3.99)]
    assert validate_catchment_monotonicity(reaches, [("A-0001", "A-0002")]) == []


def test_monotonicity_skips_edges_where_a_catchment_is_null() -> None:
    """A NULL catchment is not a violation and must never be treated as zero area."""
    upstream = _with_area("A-0001", 9.0)
    downstream = _reach("A-0002", (1, 0), 0, 1)  # catchment_area_km2 stays None
    assert validate_catchment_monotonicity([upstream, downstream], [("A-0001", "A-0002")]) == []
    assert "MONOTONICITY_VIOLATION" not in downstream.flags


def test_length_distribution_reports_the_spread() -> None:
    dist = length_distribution([100.0, 200.0, 300.0, 400.0, 500.0])
    assert dist["length_n"] == 5
    assert dist["length_median_m"] == 300.0
    assert dist["length_min_m"] == 100.0
    assert dist["length_max_m"] == 500.0
    assert math.isclose(dist["length_total_km"], 1.5)


# ---------------------------------------------------------------------------
# observability gate (pure response handling)
# ---------------------------------------------------------------------------
def _interval(day: str, strict_mean: float, mndwi_mean: float, samples: int, nodata: int) -> dict:
    return {
        "interval": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"},
        "outputs": {
            "water": {
                "bands": {
                    "B0": {
                        "stats": {
                            "mean": strict_mean,
                            "sampleCount": samples,
                            "noDataCount": nodata,
                        }
                    },
                    "B1": {
                        "stats": {
                            "mean": mndwi_mean,
                            "sampleCount": samples,
                            "noDataCount": nodata,
                        }
                    },
                }
            }
        },
    }


def test_cloudy_date_is_null_not_zero() -> None:
    """The invariant that matters most: missing is NULL + a flag, never 0."""
    payload = {"data": [{"interval": {"from": "2023-03-01T00:00:00Z"}, "outputs": {}}]}
    rows = dates_from_payload(payload)
    assert rows[0]["water_pixels"] is None
    assert rows[0]["quality_flag"] == "CLOUD"


def test_fully_masked_date_is_cloud_not_no_water() -> None:
    payload = {"data": [_interval("2023-03-01", 0.0, 0.0, 40, 40)]}
    rows = dates_from_payload(payload)
    assert rows[0]["water_pixels"] is None
    assert rows[0]["quality_flag"] == "CLOUD"


def test_clear_date_with_no_water_is_zero_and_flagged() -> None:
    payload = {"data": [_interval("2023-03-01", 0.0, 0.1, 40, 0)]}
    rows = dates_from_payload(payload)
    assert rows[0]["water_pixels"] == 0
    assert rows[0]["quality_flag"] == "NO_WATER_PIXELS"
    assert rows[0]["water_pixels_mndwi_only"] == 4


def test_water_pixel_count_is_mean_times_usable_pixels() -> None:
    payload = {"data": [_interval("2023-03-01", 0.25, 0.5, 100, 20)]}
    assert dates_from_payload(payload)[0]["water_pixels"] == 20


def test_failed_interval_is_reported_not_dropped() -> None:
    payload = {"data": [], "failedIntervals": [{"interval": {"from": "2023-05-05T00:00:00Z"}}]}
    rows = dates_from_payload(payload)
    assert rows[0]["quality_flag"] == "OUT_OF_RANGE"
    assert rows[0]["water_pixels"] is None


REACH_STUB = {"reach_id": "CMB-0001", "name": "Rio Mondego", "strahler_order": 5, "length_m": 300.0}


def test_summarise_ignores_cloudy_dates_when_taking_the_median() -> None:
    rows = [
        {"date": "2023-01-01", "quality_flag": "CLOUD", "water_pixels": None},
        {"date": "2023-01-06", "quality_flag": "OK", "water_pixels": 40},
        {"date": "2023-01-11", "quality_flag": "OK", "water_pixels": 60},
    ]
    summary = summarise_reach(REACH_STUB, rows, threshold=5)
    assert summary["median_water_pixels"] == 50
    assert summary["dates_usable"] == 2
    assert summary["dates_cloud"] == 1
    assert summary["observable"] is True


def test_reach_with_no_usable_date_is_unobservable_with_a_reason() -> None:
    rows = [{"date": "2023-01-01", "quality_flag": "CLOUD", "water_pixels": None}]
    summary = summarise_reach(REACH_STUB, rows, threshold=5)
    assert summary["median_water_pixels"] is None
    assert summary["observable"] is False
    assert summary["reason"] == "NO_USABLE_DATE"


def test_threshold_is_inclusive() -> None:
    rows = [{"date": "2023-01-01", "quality_flag": "OK", "water_pixels": 5}]
    assert summarise_reach(REACH_STUB, rows, threshold=5)["observable"] is True
    rows = [{"date": "2023-01-01", "quality_flag": "OK", "water_pixels": 4}]
    assert summarise_reach(REACH_STUB, rows, threshold=5)["observable"] is False


def test_ascii_histogram_counts_every_reach_once() -> None:
    values = [0.0, 0.0, 1.0, 3.0, 7.0, 12.0, 30.0, 80.0]
    rendered = ascii_histogram(values)
    counted = sum(int(line.rsplit(" ", 1)[1]) for line in rendered.splitlines())
    assert counted == len(values)


# ---------------------------------------------------------------------------
# outlet snapping (synthetic grid - no rasters, no pysheds)
# ---------------------------------------------------------------------------
def _toy_hydro(acc_values: np.ndarray, acc_units: str = "cells") -> HydroGrid:
    """A 1-degree-per-100-cells grid with a hand-written accumulation surface."""
    from affine import Affine

    return HydroGrid(
        grid=None,
        fdir=None,
        acc=acc_values,
        transform=Affine(0.01, 0.0, 0.0, 0.0, -0.01, 1.0),
        source="TEST",
        dirmap=(64, 128, 1, 2, 4, 8, 16, 32),
        cell_size_deg=0.01,
        acc_units=acc_units,
        cell_area_km2=1.0,
    )


def test_snap_prefers_the_nearest_channel_cell_over_a_bigger_river_nearby() -> None:
    """The bug this rule exists for: a tributary outlet taking the trunk's catchment."""
    acc = np.zeros((10, 10))
    acc[5, 5] = 50.0  # the tributary channel, right under the outlet
    acc[5, 7] = 100000.0  # the trunk, two cells away
    hydro = _toy_hydro(acc)
    lon, lat = hydro.transform @ (5.5, 5.5)
    snap = snap_to_stream(hydro, lon, lat, search_cells=3, stream_threshold_km2=10.0)
    assert (snap["row"], snap["col"]) == (5, 5)
    assert snap["method"] == "NEAREST_STREAM"


def test_snap_falls_back_to_max_accumulation_when_no_channel_is_in_range() -> None:
    acc = np.zeros((10, 10))
    acc[5, 6] = 3.0
    hydro = _toy_hydro(acc)
    lon, lat = hydro.transform @ (5.5, 5.5)
    snap = snap_to_stream(hydro, lon, lat, search_cells=3, stream_threshold_km2=10.0)
    assert (snap["row"], snap["col"]) == (5, 6)
    assert snap["method"] == "MAX_ACCUMULATION"


def test_snap_ties_are_broken_by_accumulation() -> None:
    acc = np.zeros((10, 10))
    acc[5, 4] = 20.0
    acc[5, 6] = 60.0  # same distance from the outlet, bigger channel
    hydro = _toy_hydro(acc)
    lon, lat = hydro.transform @ (5.5, 5.5)
    snap = snap_to_stream(hydro, lon, lat, search_cells=3, stream_threshold_km2=10.0)
    assert (snap["row"], snap["col"]) == (5, 6)


def test_snap_outside_the_window_returns_none_not_a_zero_catchment() -> None:
    hydro = _toy_hydro(np.zeros((10, 10)))
    assert snap_to_stream(hydro, 50.0, 50.0, search_cells=3) is None


def test_stream_mask_threshold_respects_accumulation_units() -> None:
    acc = np.array([[0.0, 5.0], [20.0, 100.0]])
    in_cells = _toy_hydro(acc, acc_units="cells")  # cell_area_km2 = 1.0
    assert in_cells.stream_mask(20.0).tolist() == [[False, False], [True, True]]
    in_km2 = _toy_hydro(acc, acc_units="km2")
    assert in_km2.stream_mask(20.0).tolist() == [[False, False], [True, True]]
