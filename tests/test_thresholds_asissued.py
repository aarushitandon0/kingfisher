"""Per-reach seasonal thresholds (engine/thresholds.py) and the as-issued weather splice
(pipeline/asissued_weather.py) against hand-computed fixtures."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.config import load_config
from engine.thresholds import lookup, month_to_season, seasonal_thresholds
from pipeline.asissued_weather import asissued_features, run_daily
from pipeline.l2_drivers import FeatureSettings, antecedent_dry_days, api_k

CFG = load_config("thresholds")
SEASONS = CFG["seasons"]


def _obs(
    rid: str, dates: list[str], values: list[float], var: str = "turbidity_proxy"
) -> pd.DataFrame:
    return pd.DataFrame(
        {"reach_id": rid, "date": pd.to_datetime(dates), "variable": var, "value": values}
    )


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------
def test_config_is_per_reach_seasonal_percentile_with_display_names() -> None:
    for v in CFG["variables"].values():
        assert v["derivation"] == "per_reach_seasonal_percentile"
        assert v["display_name"]
    assert CFG["variables"]["turbidity_proxy"]["display_name"] == "turbidity index"
    assert sorted(month_to_season(SEASONS)) == list(range(1, 13))


def test_threshold_is_the_reachs_own_seasonal_percentile() -> None:
    summer = [f"2022-07-{d:02d}" for d in range(1, 11)]
    obs = pd.concat(
        [
            _obs("A", summer, list(range(1, 11))),  # A: 1..10 in JJA
            _obs("B", summer, [100 + v for v in range(1, 11)]),  # B: 101..110 in JJA
        ]
    )
    t = seasonal_thresholds(obs, date(2023, 12, 31), CFG).set_index(["reach_id", "season"])
    assert t.loc[("A", "JJA"), "threshold"] == pytest.approx(np.quantile(range(1, 11), 0.9))
    assert t.loc[("B", "JJA"), "threshold"] == pytest.approx(100 + np.quantile(range(1, 11), 0.9))
    assert t.loc[("A", "JJA"), "clim_exceed_freq"] == pytest.approx(0.1)


def test_no_threshold_below_min_obs_and_never_from_the_future() -> None:
    few = _obs("A", [f"2022-01-{d:02d}" for d in range(1, 5)], [1, 2, 3, 4])  # 4 < min_obs
    later = _obs("A", [f"2024-07-{d:02d}" for d in range(1, 21)], [9.0] * 20)  # after fit_end
    t = seasonal_thresholds(pd.concat([few, later]), date(2023, 12, 31), CFG)
    assert t.empty
    targets = pd.DataFrame(
        {"reach_id": ["A"], "variable": ["turbidity_proxy"], "target_date": [date(2024, 7, 5)]}
    )
    assert np.isnan(lookup(targets, t, SEASONS)["threshold"].iloc[0])


def test_lookup_uses_the_target_dates_season() -> None:
    obs = pd.concat(
        [
            _obs("A", [f"2022-08-{d:02d}" for d in range(1, 11)], [10.0] * 10),
            _obs("A", [f"2022-09-{d:02d}" for d in range(1, 11)], [20.0] * 10),
        ]
    )
    t = seasonal_thresholds(obs, date(2023, 12, 31), CFG)
    targets = pd.DataFrame(
        {
            "reach_id": ["A", "A"],
            "variable": ["turbidity_proxy"] * 2,
            "target_date": [date(2024, 8, 31), date(2024, 9, 1)],
        }
    )
    assert list(lookup(targets, t, SEASONS)["threshold"]) == [10.0, 20.0]


# ---------------------------------------------------------------------------
# as-issued splice
# ---------------------------------------------------------------------------
FS = FeatureSettings(0.9, (7, 14, 30), 1.0, 0.5, 5, date(2023, 12, 31))


def _archive(precip: list[float], end: str) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=len(precip), freq="D")
    s = pd.Series(precip, index=idx, dtype="float64")
    return pd.DataFrame({"precip_mm": s, "antecedent_dry_days": antecedent_dry_days(s, 1.0)})


def _run_hourly(issued: str, days: int, precip_per_day: list[float]) -> pd.DataFrame:
    idx = pd.date_range(issued, periods=days * 24, freq="h", tz="UTC")
    p = np.repeat(np.array(precip_per_day) / 24, 24)
    frame = pd.DataFrame({"precip": p, "temp": 15.0}, index=idx)
    frame.iloc[0, 0] = np.nan  # the run never knows hour 00 of the issue day
    return frame


def test_issue_day_hour_00_comes_from_the_archive_not_left_null() -> None:
    run = run_daily(_run_hourly("2024-06-10", 3, [2.4, 0.0, 0.0]), date(2024, 6, 10), 0.1, 1.0)
    assert run.loc["2024-06-10", "precip_mm"] == pytest.approx(0.1 + 2.4 * 23 / 24)


def test_day_missing_hours_is_null_not_a_partial_sum() -> None:
    hourly = _run_hourly("2024-06-10", 3, [0.0, 4.8, 4.8]).iloc[:-5]  # last day truncated
    run = run_daily(hourly, date(2024, 6, 10), 0.0, 1.0)
    assert np.isnan(run.loc["2024-06-12", "precip_mm"])
    assert run.loc["2024-06-11", "precip_mm"] == pytest.approx(4.8)


def test_asissued_features_match_hand_computation() -> None:
    # archive: ... 10 dry days ending 2024-06-09; run: t=06-10 dry, 06-11 5 mm, 06-12 dry
    archive = _archive([5.0] + [0.0] * 10, "2024-06-09")
    archive.loc[pd.Timestamp("2024-06-10")] = [np.nan, 10.0]  # ADD at t known from archive
    run = run_daily(_run_hourly("2024-06-10", 4, [0.0, 5.0, 0.0, 0.0]), date(2024, 6, 10), 0.0, 1.0)
    f = asissued_features(archive, run, date(2024, 6, 10), [1, 2], FS)
    # h=1 (06-11): dry run of 1 run day (06-10) + archive's 10 = 11; pmax 5/24
    assert f[1]["precip_mm"] == pytest.approx(5.0)
    assert f[1]["first_flush_index"] == pytest.approx(11 * 5.0 / 24)
    # h=2 (06-12): 06-11 was wet -> 0 dry days
    assert f[2]["first_flush_index"] == pytest.approx(0.0)
    # api_7 at 06-12 = sum_{i=1..7} p(t+2-i) 0.9^i with p(06-11)=5 at i=1, rest dry archive
    assert f[2]["api_7"] == pytest.approx(5.0 * 0.9)
    assert f[2]["precip_fut_cum_mm"] == pytest.approx(5.0)


def test_asissued_api7_equals_archive_formula_when_the_run_is_perfect() -> None:
    """Feed the run the archive's own future: the splice must reproduce api_7 exactly."""
    rng = np.random.default_rng(3)
    rain = list(np.round(rng.gamma(0.6, 3.0, 40), 1))
    archive = _archive(rain, "2024-06-19")
    issued = date(2024, 6, 10)
    future = archive.loc["2024-06-10":"2024-06-19", "precip_mm"].tolist()
    hourly = _run_hourly("2024-06-10", 10, future)
    run = run_daily(hourly, issued, future[0] / 24, 1.0)
    feats = asissued_features(archive, run, issued, [3, 6], FS)
    api = api_k(archive["precip_mm"], 7, 0.9)
    for h in (3, 6):
        assert feats[h]["api_7"] == pytest.approx(
            api.loc[pd.Timestamp(issued) + pd.Timedelta(days=h)]
        )
