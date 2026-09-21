"""L2 driver-feature tests against HAND-COMPUTED fixtures.

api_k, antecedent_dry_days and first_flush_index are load-bearing: a silent bug here
silently ruins the model. Every expected value below is worked out in the comment next
to it, from the definitions in pipeline/l2_drivers.py and MASTERSPEC 6.2:

    api_k(t)            = sum_{i=1..k} P(t-i) * 0.9^i        (day t itself excluded)
    antecedent_dry_days = consecutive days before t with P < 1.0 mm
    first_flush_index   = antecedent_dry_days(t) * precip_max_hourly(t)
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pipeline.l2_drivers import (
    FeatureSettings,
    antecedent_dry_days,
    api_k,
    daily_aggregate,
    doy_harmonics,
    engineer_features,
    first_flush_index,
    max_api_training,
    snap,
    temp_low_flow_index,
    upstream_state_lag1,
)

NAN = math.nan
DAYS = pd.date_range("2023-03-01", periods=8, freq="D")

#                 d0   d1   d2   d3   d4   d5   d6   d7
PRECIP = [0.0, 2.0, 0.0, 0.0, 5.0, 0.5, 0.0, 10.0]
PMAX = [0.0, 1.5, 0.0, 0.0, 3.0, 0.4, 0.0, 6.0]


def _series(values: list[float], index: pd.DatetimeIndex = DAYS) -> pd.Series:
    return pd.Series(values, index=index, dtype=float)


def _assert_series(actual: pd.Series, expected: list[float]) -> None:
    np.testing.assert_allclose(actual.to_numpy(dtype=float), np.array(expected), equal_nan=True)


# ---------------------------------------------------------------------------
# api_k
# ---------------------------------------------------------------------------
def test_api_3_hand_computed() -> None:
    expected = [
        NAN,  # t0: needs P(-1..-3) - no history
        NAN,  # t1
        NAN,  # t2
        1.62,  # t3: P2*.9 + P1*.81 + P0*.729 = 0 + 2*.81 + 0            = 1.62
        1.458,  # t4: P3*.9 + P2*.81 + P1*.729 = 0 + 0 + 2*.729          = 1.458
        4.5,  # t5: P4*.9 + P3*.81 + P2*.729 = 5*.9                      = 4.5
        4.5,  # t6: P5*.9 + P4*.81 + P3*.729 = .45 + 4.05 + 0            = 4.5
        4.05,  # t7: P6*.9 + P5*.81 + P4*.729 = 0 + .405 + 3.645          = 4.05
    ]
    _assert_series(api_k(_series(PRECIP), 3, 0.9), expected)


def test_api_excludes_same_day_rain() -> None:
    # Only today is wet: api must be 0, because api is ANTECEDENT wetness.
    precip = _series([0, 0, 0, 0, 0, 0, 0, 50.0])
    assert api_k(precip, 3, 0.9).iloc[-1] == 0.0


def test_api_7_single_pulse_decays_geometrically() -> None:
    # 10 mm on d0 and nothing after. api_7 at t = 7 sees d0 at lag 7 -> 10 * 0.9^7.
    index = pd.date_range("2023-01-01", periods=10, freq="D")
    precip = _series([10.0] + [0.0] * 9, index)
    api7 = api_k(precip, 7, 0.9)
    assert api7.iloc[7] == pytest.approx(10 * 0.9**7)  # 4.782969
    assert api7.iloc[8] == 0.0  # d0 has left the 7-day window
    assert api7.iloc[:7].isna().all()


def test_api_null_in_window_is_null_not_dry() -> None:
    precip = _series([1.0, 1.0, NAN, 1.0, 1.0, 1.0, 1.0, 1.0])
    api3 = api_k(precip, 3, 0.9)
    # t3, t4, t5 have d2 in their window -> NULL. t6 window is d3..d5 -> 0.9+0.81+0.729.
    assert api3.iloc[3:6].isna().all()
    assert api3.iloc[6] == pytest.approx(2.439)


def test_api_rejects_non_contiguous_dates() -> None:
    gappy = pd.Series(
        [1.0, 2.0, 3.0], index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"])
    )
    with pytest.raises(ValueError, match="contiguous"):
        api_k(gappy, 2, 0.9)


# ---------------------------------------------------------------------------
# antecedent_dry_days
# ---------------------------------------------------------------------------
def test_antecedent_dry_days_hand_computed() -> None:
    expected = [
        NAN,  # t0: nothing is known before the series
        NAN,  # t1: d0 dry, but how long it had been dry before d0 is unknown
        0,  # t2: d1 = 2.0 wet
        1,  # t3: d2 dry
        2,  # t4: d2, d3 dry
        0,  # t5: d4 = 5.0 wet
        1,  # t6: d5 = 0.5 < 1.0 counts as dry
        2,  # t7: d5, d6 dry
    ]
    _assert_series(antecedent_dry_days(_series(PRECIP), 1.0), expected)


def test_exactly_threshold_is_wet() -> None:
    precip = _series([5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    add = antecedent_dry_days(precip, 1.0)
    assert add.iloc[2] == 0  # d1 = 1.0 is NOT < 1.0 -> wet
    assert add.iloc[7] == 5  # d2..d6


def test_null_day_makes_dry_run_unknown_until_next_wet_day() -> None:
    #                 d0   d1   d2   d3   d4   d5   d6   d7
    precip = _series([5.0, 0.0, NAN, 0.0, 0.0, 5.0, 0.0, 0.0])
    expected = [
        NAN,  # t0
        0,  # t1: d0 wet
        1,  # t2: d1 dry
        NAN,  # t3: d2 unknown
        NAN,  # t4: d3 dry, but the run through d2 is unknown
        NAN,  # t5: same
        0,  # t6: d5 wet resets - known again
        1,  # t7
    ]
    _assert_series(antecedent_dry_days(precip, 1.0), expected)


# ---------------------------------------------------------------------------
# first_flush_index
# ---------------------------------------------------------------------------
def test_first_flush_index_hand_computed() -> None:
    add = antecedent_dry_days(_series(PRECIP), 1.0)
    expected = [
        NAN,  # t0: add unknown
        NAN,  # t1: add unknown (even though pmax = 1.5)
        0.0,  # t2: 0 * 0.0
        0.0,  # t3: 1 * 0.0
        6.0,  # t4: 2 dry days * 3.0 mm/h
        0.0,  # t5: 0 * 0.4
        0.0,  # t6: 1 * 0.0
        12.0,  # t7: 2 dry days * 6.0 mm/h  <- the first flush
    ]
    _assert_series(first_flush_index(add, _series(PMAX)), expected)


def test_first_flush_index_null_intensity_is_null() -> None:
    add = _series([3.0] * 8)
    pmax = _series([1.0, NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ffi = first_flush_index(add, pmax)
    assert math.isnan(ffi.iloc[1]) and ffi.iloc[0] == 3.0


# ---------------------------------------------------------------------------
# hourly -> daily
# ---------------------------------------------------------------------------
def _hourly(precip: list[float], temp: list[float] | None = None) -> pd.DataFrame:
    index = pd.date_range("2023-03-01", periods=len(precip), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "precip": precip,
            "temp": temp if temp is not None else [10.0] * len(precip),
            "soil_moisture": [0.3] * len(precip),
            "et0": [0.1] * len(precip),
        },
        index=index,
    )


def test_daily_aggregate_intensity_duration_and_sum() -> None:
    day1 = [0.0] * 24
    day1[5], day1[6], day1[7] = 0.5, 2.0, 4.0  # 0.5 is NOT > 0.5
    day2 = [0.2] * 24
    daily = daily_aggregate(_hourly(day1 + day2), 0.5)
    assert daily["precip_mm"].tolist() == pytest.approx([6.5, 4.8])
    assert daily["precip_max_hourly"].tolist() == pytest.approx([4.0, 0.2])
    assert daily["precip_duration_h"].tolist() == [2, 0]
    assert daily["et0"].tolist() == pytest.approx([2.4, 2.4])


def test_daily_aggregate_missing_hour_is_null_day_not_less_rain() -> None:
    day1 = [1.0] * 24
    day1[12] = NAN
    daily = daily_aggregate(_hourly(day1 + [1.0] * 24), 0.5)
    assert math.isnan(daily["precip_mm"].iloc[0])
    assert math.isnan(daily["precip_max_hourly"].iloc[0])
    assert math.isnan(daily["precip_duration_h"].iloc[0])
    assert daily["temp_mean_c"].iloc[0] == 10.0  # temperature itself was complete
    assert daily["precip_mm"].iloc[1] == 24.0


def test_daily_aggregate_partial_day_is_null() -> None:
    daily = daily_aggregate(_hourly([1.0] * 30), 0.5)  # 24 h + 6 h
    assert daily["precip_mm"].iloc[0] == 24.0
    assert math.isnan(daily["precip_mm"].iloc[1])


# ---------------------------------------------------------------------------
# temp_low_flow_index, harmonics, snapping
# ---------------------------------------------------------------------------
def test_temp_low_flow_index_hand_computed_and_clipped() -> None:
    temp = _series([20.0, 20.0, 20.0, NAN, 20, 20, 20, 20])
    api14 = _series([0.0, 5.0, 12.0, 5.0, 10, 10, 10, 10])
    tlf = temp_low_flow_index(temp, api14, max_api=10.0)
    assert tlf.iloc[0] == 20.0  # 20 * (1 - 0/10)
    assert tlf.iloc[1] == 10.0  # 20 * (1 - 5/10)
    assert tlf.iloc[2] == 0.0  # wetter than training max -> clipped to 0, not -4
    assert math.isnan(tlf.iloc[3])


def test_max_api_uses_training_period_only() -> None:
    index = pd.date_range("2023-12-30", periods=4, freq="D")
    api14 = pd.Series([3.0, 4.0, 99.0, 99.0], index=index)
    assert max_api_training(api14, date(2023, 12, 31)) == 4.0


def test_doy_harmonics_unit_circle() -> None:
    s, c = doy_harmonics(pd.date_range("2023-01-01", periods=365, freq="D"))
    np.testing.assert_allclose(s**2 + c**2, 1.0)


def test_snap_to_grid() -> None:
    assert snap(-8.4432, 0.1) == -8.4
    assert snap(40.2551, 0.1) == 40.3
    assert snap(40.2449, 0.1) == 40.2


def test_engineer_features_reindexes_gaps_to_explicit_nulls() -> None:
    fs = FeatureSettings(0.9, (7, 14, 30), 1.0, 0.5, 5, date(2023, 12, 31))
    index = pd.date_range("2023-01-01", "2024-02-01", freq="D")
    daily = pd.DataFrame(
        {
            "precip_mm": 1.0,
            "precip_max_hourly": 0.5,
            "precip_duration_h": 1,
            "temp_mean_c": 12.0,
            "temp_max_c": 15.0,
            "soil_moisture": 0.3,
            "et0": 2.0,
        },
        index=index,
    ).drop(index[100])  # one missing day
    out = engineer_features(daily, fs)
    assert len(out) == len(index)
    assert math.isnan(out.loc[index[100], "precip_mm"])
    assert out.loc[index[101] : index[107], "api_7"].isna().all()
    assert not math.isnan(out.loc[index[108], "api_7"])


# ---------------------------------------------------------------------------
# upstream_state_lag1
# ---------------------------------------------------------------------------
def test_upstream_state_lag1_area_weighted_and_lagged() -> None:
    # A (area 1) and B (area 3) both flow into C. D is a headwater.
    dates = pd.date_range("2023-06-01", periods=8, freq="D")
    obs = pd.DataFrame(
        [
            ("A", date(2023, 6, 2), "OK", 10.0, 0.1),
            ("B", date(2023, 6, 2), "OK", 30.0, 0.3),
            ("B", date(2023, 6, 3), "CLOUD", None, None),  # not OK - ignored
            ("A", date(2023, 6, 5), "OK", 20.0, 0.2),
        ],
        columns=["reach_id", "obs_date", "quality_flag", "turbidity_proxy", "ndci"],
    )
    out = upstream_state_lag1(
        obs,
        [("A", "C"), ("B", "C")],
        {"A": 1.0, "B": 3.0, "C": 5.0, "D": 1.0},
        ["C", "D"],
        dates,
        max_age_days=3,
    ).set_index(["reach_id", "date"])
    c = out.loc["C"]
    # 06-02: the 06-02 observations are not yet t-1 -> nothing
    assert c.loc["2023-06-02", "upstream_state_flag"] == "NO_RECENT_UPSTREAM_OBS"
    # 06-03 .. 06-05: obs of 06-02 served (age 1..3): (10*1 + 30*3) / 4 = 25
    for day in ("2023-06-03", "2023-06-04", "2023-06-05"):
        assert c.loc[day, "upstream_state_lag1_turbidity"] == pytest.approx(25.0)
    # 06-06: A's 06-05 obs (age 1); B's 06-02 obs is age 4 > 3 -> only A: 20
    assert c.loc["2023-06-06", "upstream_state_lag1_turbidity"] == pytest.approx(20.0)
    assert c.loc["2023-06-06", "upstream_state_lag1_ndci"] == pytest.approx(0.2)
    assert c.loc["2023-06-06", "upstream_state_flag"] == "OK"
    # 06-08: A's 06-05 obs is age 3 -> still 20
    assert c.loc["2023-06-08", "upstream_state_lag1_turbidity"] == pytest.approx(20.0)
    d = out.loc["D"]
    assert (d["upstream_state_flag"] == "NO_UPSTREAM_REACH").all()
    assert d["upstream_state_lag1_turbidity"].isna().all()
