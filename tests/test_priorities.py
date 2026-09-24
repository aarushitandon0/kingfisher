"""Prioritisation (engine/priorities.py) - MASTERSPEC 10. Pure, hand-checkable fixtures."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.priorities import exposure_score, rank_reaches

W = {"school": 3.0, "footway": 1.0}


def fc(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["reach_id", "variable", "p10", "p90", "exceedance_prob"])


def ex(rows: list[tuple[str, str, float | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["reach_id", "feature_type", "count"])


def test_value_is_uncertainty_times_risk_times_exposure() -> None:
    f = fc([("A", "turbidity_proxy", 0, 4, 0.8), ("B", "turbidity_proxy", 0, 2, 0.8)])
    e = ex([("A", "school", 1), ("B", "school", 1), ("B", "footway", 3)])
    p = rank_reaches(["A", "B"], f, e, W, 1000)
    by = {r.reach_id: r for r in p.ranked}
    # A: width 4 = max -> u 1; E 3; B: u 0.5; E 6 = max -> weight 1
    assert by["A"].information_value == pytest.approx(1.0 * 0.8 * math.log1p(3) / math.log1p(6))
    assert by["B"].information_value == pytest.approx(0.5 * 0.8 * 1.0)
    assert [r.rank for r in p.ranked] == [1, 2]
    assert p.ranked[0].information_value >= p.ranked[1].information_value


def test_unscorable_reaches_are_listed_not_zeroed() -> None:
    f = fc([("A", "ndci", 0, 1, 0.5), ("D", "ndci", 0, 1, float("nan")), ("X", "ndci", 0, 1, 0.9)])
    e = ex([("A", "school", 1), ("D", "school", 1)])
    p = rank_reaches(["A", "D", "X", "N"], f, e, W, 1000)
    assert [r.reach_id for r in p.ranked] == ["A"]
    assert {(n.reach_id, n.reason) for n in p.not_ranked} == {
        ("D", "NO_THRESHOLD"),
        ("X", "NO_EXPOSURE"),
        ("N", "NO_FORECAST"),
    }


def test_missing_population_is_flagged_not_filled() -> None:
    e, flags = exposure_score(ex([("A", "population", None), ("A", "school", 2)]), W, 1000)
    assert e == 6.0 and flags == ("POPULATION_UNAVAILABLE",)
    e2, _ = exposure_score(ex([("A", "population", 2500)]), W, 1000)
    assert e2 == 2.5


def test_best_variable_is_reported() -> None:
    f = fc([("A", "ndci", 0, 1, 0.2), ("A", "turbidity_proxy", 0, 1, 0.9)])
    p = rank_reaches(["A"], f, ex([("A", "school", 1)]), W, 1000)
    assert p.ranked[0].components.variable == "turbidity_proxy"


def test_bad_config_is_refused() -> None:
    with pytest.raises(ValueError):
        rank_reaches([], fc([]), ex([]), W, 0)
    with pytest.raises(ValueError):
        rank_reaches([], fc([]), ex([]), {"school": -1}, 1000)
