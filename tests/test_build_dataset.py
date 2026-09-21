"""Leakage tests (CLAUDE.md Testing #2): no future information enters any training fold.

The central test is a perturbation test. Build the frame; build it again after replacing
every observation AND every weather value dated after a fold boundary with different
numbers (and adding and deleting observation dates); the rows of every earlier fold must
come out bit-identical. That catches every leak path at once - a target shifted the wrong
way, an as-of join that looks forward, an embargo that misses a column, a normaliser
fitted on the whole series (temp_low_flow_index's max_api), a centred rolling window -
without having to enumerate them.

The frame is built through the real L2 feature code (engineer_features,
upstream_state_lag1), not a stand-in, so the drivers are covered too. And the check is
itself checked: `test_perturbation_check_catches_a_deliberate_leak` plants a leak and
asserts the check fails, so a vacuous pass is impossible.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pipeline.build_dataset import (
    FrameSettings,
    assign_fold,
    build_frame,
    embargo,
    split_frame,
    summarise,
    target_column,
    walk_forward,
)
from pipeline.l2_drivers import FeatureSettings, engineer_features, upstream_state_lag1

TRAIN_END = date(2023, 12, 31)
VAL_END = date(2024, 12, 31)
START, END = date(2022, 9, 1), date(2025, 6, 30)
REACHES = ["TST-A", "TST-B", "TST-C"]  # A -> C, B -> C
TOPOLOGY = [("TST-A", "TST-C"), ("TST-B", "TST-C")]

FS = FrameSettings(
    variables=("turbidity_proxy", "ndci"),
    horizons=tuple(range(1, 11)),
    asof_max_age_days=60,
    future_driver_columns=(
        "precip_mm",
        "precip_max_hourly",
        "first_flush_index",
        "api_7",
        "temp_mean_c",
    ),
    train_end=TRAIN_END,
    val_start=date(2024, 1, 1),
    val_end=VAL_END,
    test_start=date(2025, 1, 1),
    test_end=None,
)
DFS = FeatureSettings(0.9, (7, 14, 30), 1.0, 0.5, 5, TRAIN_END)


# ---------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------
def _weather(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="D", name="date")
    wet = rng.random(len(idx)) < 0.35
    precip = np.where(wet, rng.gamma(1.5, 6.0, len(idx)), 0.0)
    doy = idx.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "precip_mm": precip,
            "precip_max_hourly": precip * rng.uniform(0.1, 0.5, len(idx)),
            "precip_duration_h": (precip / 2).round(),
            "temp_mean_c": 15
            + 8 * np.sin(2 * np.pi * (doy - 110) / 365)
            + rng.normal(0, 1, len(idx)),
            "temp_max_c": 20 + 8 * np.sin(2 * np.pi * (doy - 110) / 365),
            "soil_moisture": 0.3,
            "et0": 2.5,
        },
        index=idx,
    )


def _observations(seed: int = 1) -> pd.DataFrame:
    """Every 5th day per reach: OK most of the time, sometimes CLOUD (values NULL)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, rid in enumerate(REACHES):
        for day in pd.date_range(START + pd.Timedelta(days=i), END, freq="5D"):
            flag = "OK" if rng.random() < 0.6 else "CLOUD"
            ok = flag == "OK"
            rows.append(
                {
                    "reach_id": rid,
                    "obs_date": day.date(),
                    "source": "S2",
                    "quality_flag": flag,
                    "turbidity_proxy": float(rng.gamma(2, 3)) if ok else np.nan,
                    "ndci": float(rng.normal(-0.05, 0.1)) if ok else np.nan,
                }
            )
    return pd.DataFrame(rows)


STATIC = pd.DataFrame(
    {
        "reach_id": REACHES,
        "imperviousness_pct": [10.0, 40.0, 25.0],
        "riparian_ndvi_mean": [0.5, 0.3, 0.4],
        "riparian_width_m": [30.0, 10.0, 20.0],
        "road_density_km_km2": [2.0, 8.0, 5.0],
        "alan_radiance": [np.nan, np.nan, np.nan],
        "population": [100, 5000, 2000],
        "urban_fraction": [0.1, 0.4, 0.25],
    }
)
REACH_ATTRS = pd.DataFrame(
    {
        "reach_id": REACHES,
        "catchment_area_km2": [2.0, 3.0, 6.0],
        "strahler_order": [2, 2, 3],
        "observable": [True, True, False],
        "median_water_pixels": [12.0, 8.0, 1.0],
    }
)


def _drivers(weather: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """The real L2 path: engineer_features per cell, broadcast, upstream state per reach."""
    features = engineer_features(weather, DFS)
    upstream = upstream_state_lag1(
        obs,
        TOPOLOGY,
        {"TST-A": 2.0, "TST-B": 3.0, "TST-C": 6.0},
        REACHES,
        pd.DatetimeIndex(features.index),
        DFS.upstream_max_age_days,
    )
    frames = [features.rename_axis("date").reset_index().assign(reach_id=r) for r in REACHES]
    drivers = pd.concat(frames, ignore_index=True).merge(upstream, on=["reach_id", "date"])
    drivers["source"] = "ARCHIVE"
    return drivers


def _build(weather: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    return build_frame(_drivers(weather, obs), obs, STATIC, REACH_ATTRS, FS)


def _perturb_after(
    weather: pd.DataFrame, obs: pd.DataFrame, boundary: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace everything dated after `boundary` with different numbers."""
    w = weather.copy()
    after = w.index > pd.Timestamp(boundary)
    w.loc[after, "precip_mm"] = w.loc[after, "precip_mm"] * 3 + 20
    w.loc[after, "precip_max_hourly"] = w.loc[after, "precip_max_hourly"] * 5 + 7
    w.loc[after, "temp_mean_c"] += 9
    o = obs.copy()
    later = pd.to_datetime(o["obs_date"]) > pd.Timestamp(boundary)
    o.loc[later, "turbidity_proxy"] = o.loc[later, "turbidity_proxy"] * 100 + 999
    o.loc[later, "ndci"] = 0.9
    o.loc[later & (o["quality_flag"] == "CLOUD"), ["quality_flag", "turbidity_proxy", "ndci"]] = [
        "OK",
        555.0,
        0.5,
    ]
    o = o.drop(o[later].index[::7])  # some later observations vanish
    extra = pd.DataFrame(
        {
            "reach_id": REACHES,
            "obs_date": [boundary + pd.Timedelta(days=1).to_pytimedelta()] * 3,
            "source": "S2",
            "quality_flag": "OK",
            "turbidity_proxy": 7777.0,
            "ndci": 0.7,
        }
    )
    o = pd.concat([o, extra], ignore_index=True).drop_duplicates(
        ["reach_id", "obs_date"], keep="last"
    )
    return w, o


def assert_folds_invariant(
    build: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    boundary: date,
    folds: tuple[str, ...],
) -> None:
    weather, obs = _weather(), _observations()
    base = split_frame(build(weather, obs), FS)
    pw, po = _perturb_after(weather, obs, boundary)
    perturbed = split_frame(build(pw, po), FS)
    for fold in folds:
        assert not base[fold].empty
        pd.testing.assert_frame_equal(
            base[fold].reset_index(drop=True), perturbed[fold].reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# THE leakage tests
# ---------------------------------------------------------------------------
def test_train_fold_is_blind_to_everything_after_train_end() -> None:
    assert_folds_invariant(_build, TRAIN_END, ("train",))


def test_val_fold_is_blind_to_everything_after_val_end() -> None:
    assert_folds_invariant(_build, VAL_END, ("train", "val"))


def test_walk_forward_training_windows_are_blind_to_their_future() -> None:
    weather, obs = _weather(), _observations()
    base = walk_forward(_build(weather, obs), FS)
    for fold in base:
        pw, po = _perturb_after(weather, obs, fold["train_end"])
        perturbed = {f["name"]: f for f in walk_forward(_build(pw, po), FS)}
        pd.testing.assert_frame_equal(
            fold["train"].reset_index(drop=True),
            perturbed[fold["name"]]["train"].reset_index(drop=True),
        )


def test_perturbation_check_catches_a_deliberate_leak() -> None:
    """Break it on purpose: a feature that peeks 3 days ahead at the observation. If the
    invariance check did not fail here, the tests above would prove nothing."""

    def leaky_build(weather: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
        frame = _build(weather, obs)
        frame = frame.copy()
        frame["peek"] = frame.groupby("reach_id", observed=True)["turbidity_proxy_asof"].shift(-3)
        return frame

    with pytest.raises(AssertionError):
        assert_folds_invariant(leaky_build, TRAIN_END, ("train",))


def test_full_series_normaliser_would_be_caught() -> None:
    """temp_low_flow_index's max_api fitted on ALL dates instead of training dates is a
    leak through the weather; the check must see it."""

    def leaky_features_build(weather: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
        frame = _build(weather, obs)
        api14 = frame["api_14"].astype("float64")
        frame["temp_low_flow_index"] = (
            frame["temp_mean_c"] * (1 - api14 / api14.max()).clip(0, 1)
        ).astype("float32")
        return frame

    with pytest.raises(AssertionError):
        assert_folds_invariant(leaky_features_build, TRAIN_END, ("train",))


# ---------------------------------------------------------------------------
# structural checks
# ---------------------------------------------------------------------------
def test_no_fold_carries_a_target_dated_after_its_end() -> None:
    frame = _build(_weather(), _observations())
    last = frame["date"].max().date()
    ends = {"train": TRAIN_END, "val": VAL_END, "test": last}
    for name, rows in split_frame(frame, FS).items():
        d = pd.to_datetime(rows["date"])
        for h in FS.horizons:
            for col in [target_column(v, h) for v in FS.variables] + [
                f"{c}_fut_h{h}" for c in FS.future_driver_columns
            ]:
                referenced = d[rows[col].notna()] + pd.Timedelta(days=h)
                if len(referenced):
                    assert referenced.max() <= pd.Timestamp(ends[name]), (name, col)


def test_folds_are_disjoint_chronological_and_cover_the_frame() -> None:
    frame = _build(_weather(), _observations())
    parts = split_frame(frame, FS)
    assert parts["train"]["date"].max() < parts["val"]["date"].min()
    assert parts["val"]["date"].max() < parts["test"]["date"].min()
    assert parts["train"]["date"].max().date() == TRAIN_END
    assert parts["val"]["date"].min().date() == date(2024, 1, 1)
    assert sum(len(p) for p in parts.values()) == len(frame)


def test_targets_are_exactly_the_ok_observation_h_days_ahead() -> None:
    obs = _observations()
    frame = _build(_weather(), obs).set_index(["reach_id", "date"])
    ok = obs[obs["quality_flag"] == "OK"].set_index(["reach_id", "obs_date"])
    rid = "TST-A"
    for (r, d), row in ok.loc[[rid]].head(20).iterrows():
        for h in (1, 4, 10):
            issue = pd.Timestamp(d) - pd.Timedelta(days=h)
            if (r, issue) in frame.index:
                got = frame.loc[(r, issue), target_column("turbidity_proxy", h)]
                assert got == pytest.approx(row["turbidity_proxy"], rel=1e-6)


def test_cloud_observations_never_become_targets_or_zeros() -> None:
    obs = _observations()
    frame = _build(_weather(), obs).set_index(["reach_id", "date"])
    cloud = obs[obs["quality_flag"] == "CLOUD"]
    checked = 0
    for _, row in cloud.head(30).iterrows():
        issue = pd.Timestamp(row["obs_date"]) - pd.Timedelta(days=1)
        key = (row["reach_id"], issue)
        if key in frame.index:
            assert np.isnan(frame.loc[key, target_column("turbidity_proxy", 1)])
            checked += 1
    assert checked > 0
    t = frame[[c for c in frame.columns if c.startswith("target_turbidity")]]
    assert not (t == 0).any().any()


def test_asof_is_labelled_with_age_and_expires() -> None:
    weather, obs = _weather(), _observations()
    # TST-A: keep only one OK observation, early; everything else gone.
    first = obs[(obs["reach_id"] == "TST-A") & (obs["quality_flag"] == "OK")].iloc[[0]]
    obs = pd.concat([obs[obs["reach_id"] != "TST-A"], first], ignore_index=True)
    frame = _build(weather, obs)
    a = frame[frame["reach_id"] == "TST-A"].set_index("date")
    d0 = pd.Timestamp(first["obs_date"].iloc[0])
    assert a.loc[d0, "obs_asof_age_days"] == 0
    assert a.loc[d0 + pd.Timedelta(days=60), "obs_asof_age_days"] == 60
    assert a.loc[d0 + pd.Timedelta(days=60), "turbidity_proxy_asof"] == pytest.approx(
        first["turbidity_proxy"].iloc[0], rel=1e-6
    )
    assert np.isnan(a.loc[d0 + pd.Timedelta(days=61), "turbidity_proxy_asof"])
    assert np.isnan(a.loc[d0 + pd.Timedelta(days=61), "obs_asof_age_days"])
    assert a.loc[: d0 - pd.Timedelta(days=1), "turbidity_proxy_asof"].isna().all()


def test_forecast_driver_rows_are_refused() -> None:
    drivers = _drivers(_weather(), _observations())
    drivers.loc[drivers.index[-5:], "source"] = "FORECAST"
    with pytest.raises(ValueError, match="FORECAST"):
        build_frame(drivers, _observations(), STATIC, REACH_ATTRS, FS)


def test_gappy_driver_series_is_refused() -> None:
    drivers = _drivers(_weather(), _observations())
    drivers = drivers.drop(drivers.index[100])
    with pytest.raises(ValueError, match="contiguous"):
        build_frame(drivers, _observations(), STATIC, REACH_ATTRS, FS)


def test_assign_fold_boundaries() -> None:
    dates = pd.Series(pd.to_datetime(["2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01"]))
    assert assign_fold(dates, FS, date(2025, 6, 30)).tolist() == ["train", "val", "val", "test"]


def test_embargo_masks_only_columns_reaching_past_the_end() -> None:
    frame = _build(_weather(), _observations())
    row = frame[(frame["reach_id"] == "TST-A") & (frame["date"] == pd.Timestamp("2023-12-28"))]
    masked = embargo(row, TRAIN_END, FS)
    assert not np.isnan(masked["precip_mm_fut_h3"].iloc[0])  # 12-31: inside
    assert np.isnan(masked["precip_mm_fut_h4"].iloc[0])  # 01-01: beyond
    assert masked["precip_mm"].iloc[0] == row["precip_mm"].iloc[0]  # t itself untouched


def test_summary_reports_the_observable_split_and_missingness() -> None:
    text = summarise(_build(_weather(), _observations()), FS)
    assert "observable   reaches    2" in text
    assert "driver-only  reaches    1" in text
    for fold in ("train", "val", "test"):
        assert f"  {fold} " in text
    assert "MISSING FRACTION PER COLUMN" in text
    assert "alan_radiance" in text and "100.0%" in text
