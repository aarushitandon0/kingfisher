"""The L1 run plan: which requests, in which order, and what a riparian window becomes."""

from __future__ import annotations

from datetime import date

from shapely.geometry import LineString

from pipeline.l1_satellite import (
    RIPARIAN,
    WATER,
    RunPlan,
    plan_tasks,
    riparian_chunk,
    summarise_riparian_window,
)
from tests.test_l1_satellite import S

PLAN = RunPlan(
    water_years=(2024, 2025, 2026, 2023, 2022),
    water_observable_only=True,
    riparian_years=(2024, 2025, 2026, 2023),
    riparian_window=("07-01", "07-31"),
    probe_years=(2016, 2017),
)
LINE = LineString([(0, 0), (1, 1)])
REACHES = [
    {"reach_id": "A", "observable": True, "line": LINE},
    {"reach_id": "B", "observable": False, "line": LINE},
    {"reach_id": "C", "observable": None, "line": LINE},
    {"reach_id": "D", "observable": True, "line": LINE},
]


def test_water_only_for_observable_reaches_in_year_order() -> None:
    tasks, skipped = plan_tasks(REACHES, PLAN, date(2026, 9, 21))
    water = [t for t in tasks if t.kind == WATER]
    assert {t.reach_id for t in water} == {"A", "D"}  # unassessed (None) is not observable
    assert [t.start.year for t in water][::2] == [2024, 2025, 2026, 2023, 2022]
    assert skipped["WATER_UNOBSERVABLE_REACH"] == 2
    # all water before any riparian
    kinds = [t.kind for t in tasks]
    assert kinds.index(RIPARIAN) == len(water)


def test_current_year_water_chunk_ends_today() -> None:
    tasks, _ = plan_tasks(REACHES, PLAN, date(2026, 9, 21))
    y2026 = [t for t in tasks if t.kind == WATER and t.start.year == 2026]
    assert all(t.end == date(2026, 9, 21) for t in y2026)


def test_riparian_is_one_midsummer_window_per_reach_per_year_for_every_reach() -> None:
    tasks, _ = plan_tasks(REACHES, PLAN, date(2026, 9, 21))
    rip = [t for t in tasks if t.kind == RIPARIAN]
    assert len(rip) == len(REACHES) * len(PLAN.riparian_years)
    assert {(t.start.month, t.start.day, t.end.month, t.end.day) for t in rip} == {(7, 1, 7, 31)}
    assert PLAN.window_days() == 31


def test_unfinished_riparian_window_is_not_requested() -> None:
    assert riparian_chunk(2026, ("07-01", "07-31"), date(2026, 7, 15)) is None
    _, skipped = plan_tasks(REACHES, PLAN, date(2026, 7, 15))
    assert skipped["RIPARIAN_WINDOW_NOT_COMPLETE"] == len(REACHES)


def _rip(
    day: str, *, cloud: int = 0, usable: int = 300, land: int = 200, ndvi: float = 0.5
) -> dict:
    return {
        "date": day,
        "r_data_pixels": 300,
        "r_cloud_pixels": cloud,
        "r_usable_pixels": usable,
        "r_land_pixels": land,
        "r_ndvi": ndvi,
    }


def test_riparian_window_is_median_of_clear_acquisitions_only() -> None:
    rows = [
        _rip("2024-07-03", ndvi=0.40),
        _rip("2024-07-08", ndvi=0.60),
        _rip("2024-07-13", ndvi=0.50),
        _rip("2024-07-18", cloud=200, ndvi=0.05),  # cloudy - must not drag the median
    ]
    out = summarise_riparian_window("A", date(2024, 7, 1), date(2024, 7, 31), rows, S)
    assert out["flag"] == "OK" and out["riparian_ndvi_median"] == 0.5
    assert out["n_acquisitions"] == 4 and out["n_clear"] == 3
    assert out["flag_counts"] == {"OK": 3, "CLOUD": 1}


def test_riparian_window_without_a_clear_acquisition_is_null_with_a_flag() -> None:
    rows = [_rip("2024-07-03", cloud=250), _rip("2024-07-08", cloud=299)]
    out = summarise_riparian_window("A", date(2024, 7, 1), date(2024, 7, 31), rows, S)
    assert out["riparian_ndvi_median"] is None and out["flag"] == "NO_CLEAR_ACQUISITION"
    empty = summarise_riparian_window("A", date(2024, 7, 1), date(2024, 7, 31), [], S)
    assert empty["riparian_ndvi_median"] is None and empty["flag"] == "NO_ACQUISITION"
