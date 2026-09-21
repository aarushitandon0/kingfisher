"""EA-LSTM export and inputs (P5.1): targets are never filled, the model never sees reach
identity or an observation of the state, the spatial holdout is never trained on, and
the hindcast -> forecast-row mapping keeps the walk-forward embargo.

The on-disk checks run against data/processed/nh when it exists and skip otherwise.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from models import ealstm
from pipeline.export_neuralhydrology import (
    QUALITY_CODES,
    ExportError,
    TargetSpec,
    check_simulator_inputs,
    check_target_nans,
    load_nh_config,
    reach_timeseries,
    select_holdout,
)

TARGETS = (
    TargetSpec("turbidity_proxy", "turbidity_log1p", "log1p"),
    TargetSpec("ndci", "ndci", "identity"),
)
DYN = ("precip_mm", "api_7")


def _rows(n: int = 10) -> pd.DataFrame:
    d = pd.date_range("2024-01-01", periods=n, freq="D")
    age = np.full(n, np.nan)
    turb = np.full(n, np.nan)
    ndci = np.full(n, np.nan)
    for i in (2, 6):  # OK observations on day 2 and day 6
        age[i], turb[i], ndci[i] = 0, 5.0 + i, 0.1 * i
    return pd.DataFrame(
        {
            "reach_id": "CMB-0001",
            "date": d.date,
            "precip_mm": np.arange(n, dtype=float),
            "api_7": np.ones(n),
            "obs_asof_age_days": age,
            "turbidity_proxy_asof": turb,
            "ndci_asof": ndci,
        }
    )


def _flags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-07", "2024-01-09"]),
            "quality_flag": ["OK", "CLOUD", "OK", "NO_WATER_PIXELS"],
        }
    )


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def test_targets_nan_exactly_where_not_ok() -> None:
    s = reach_timeseries(_rows(), _flags(), DYN, TARGETS)
    counts = check_target_nans(s, TARGETS, "CMB-0001")
    # 10 days, 2 OK, 2 non-OK flagged, 6 with no row at all
    assert counts == {
        "days": 10,
        "ok": 2,
        "non_ok": 2,
        "no_row": 6,
        "nan_turbidity_log1p": 8,
        "nan_ndci": 8,
    }
    assert s["turbidity_log1p"].iloc[2] == pytest.approx(np.log1p(7.0))
    assert s["quality_code"].iloc[3] == QUALITY_CODES["CLOUD"]
    assert not (s[["turbidity_log1p", "ndci"]] == 0).any().any()


def test_a_filled_target_is_caught() -> None:
    s = reach_timeseries(_rows(), _flags(), DYN, TARGETS)
    s.loc[s.index[3], "ndci"] = 0.0  # the CLOUD day, "filled"
    with pytest.raises(ExportError, match="filled"):
        check_target_nans(s, TARGETS, "CMB-0001")


def test_a_forward_fill_is_caught() -> None:
    s = reach_timeseries(_rows(), _flags(), DYN, TARGETS)
    s["turbidity_log1p"] = s["turbidity_log1p"].ffill()
    with pytest.raises(ExportError):
        check_target_nans(s, TARGETS, "CMB-0001")


def test_unobserved_reach_exports_all_nan_targets() -> None:
    rows = _rows()
    rows[["obs_asof_age_days", "turbidity_proxy_asof", "ndci_asof"]] = np.nan
    s = reach_timeseries(rows, None, DYN, TARGETS)
    assert s["turbidity_log1p"].isna().all() and s["ndci"].isna().all()
    assert check_target_nans(s, TARGETS, "X")["no_row"] == 10


# ---------------------------------------------------------------------------
# inputs: no reach identity, no observation of the state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    ["reach_id", "upstream_state_lag1_turbidity", "turbidity_proxy_asof", "obs_asof_age_days"],
)
def test_forbidden_inputs_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        check_simulator_inputs([*DYN, bad], ["imperviousness_pct"], {})


def test_basin_id_encoding_rejected() -> None:
    with pytest.raises(ValueError, match="use_basin_id_encoding"):
        check_simulator_inputs(DYN, ["imperviousness_pct"], {"use_basin_id_encoding": True})


@pytest.mark.parametrize("key", ["autoregressive_inputs", "lagged_features", "evolving_attributes"])
def test_autoregression_rejected(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        check_simulator_inputs(DYN, ["imperviousness_pct"], {key: ["ndci"]})


@pytest.mark.parametrize("variable", ["turbidity_proxy", "ndci"])
def test_shipped_configs_are_pure_simulators(variable: str) -> None:
    cfg = load_nh_config(variable)
    es = ealstm.EALSTMSettings.from_config()
    ealstm.check_run_config(cfg, es.target(variable))  # raises on any deviation
    assert "reach_id" not in cfg["dynamic_inputs"] + cfg["static_attributes"]
    assert not any("lag" in f or "asof" in f for f in cfg["dynamic_inputs"])


def test_run_config_deviation_raises() -> None:
    cfg = dict(load_nh_config("ndci"))
    cfg["hidden_size"] = 128
    with pytest.raises(ValueError, match="hidden_size"):
        ealstm.check_run_config(cfg, ealstm.EALSTMSettings.from_config().target("ndci"))


# ---------------------------------------------------------------------------
# spatial holdout
# ---------------------------------------------------------------------------
def test_holdout_deterministic_and_about_20_percent() -> None:
    ids = [f"CMB-{i:04d}" for i in range(51)]
    a = select_holdout(ids, 0.2, "salt")
    assert a == select_holdout(list(reversed(ids)), 0.2, "salt")
    assert len(a) == 10
    assert a != select_holdout(ids, 0.2, "other-salt")


NH_DIR = ealstm.EALSTMSettings.from_config().nh_dir
needs_export = pytest.mark.skipif(
    not (NH_DIR / "export_manifest.json").exists(), reason="no NH export on disk"
)


@needs_export
def test_export_on_disk_holdout_never_trained() -> None:
    m = json.loads((NH_DIR / "export_manifest.json").read_text())
    train = set((NH_DIR / "train_reaches.txt").read_text().split())
    hold = set((NH_DIR / "spatial_holdout.txt").read_text().split())
    alls = set((NH_DIR / "all_reaches.txt").read_text().split())
    assert train and hold
    assert not train & hold
    assert train | hold <= alls
    assert all(m["per_reach"][r]["observable"] for r in hold)
    for v in ("turbidity_proxy", "ndci"):
        cfg = load_nh_config(v)
        assert "train_reaches" in cfg["train_basin_file"]
        assert "train_reaches" in cfg["validation_basin_file"]  # early stopping too


@needs_export
def test_export_on_disk_nan_invariant_holds() -> None:
    import xarray as xr

    m = json.loads((NH_DIR / "export_manifest.json").read_text())
    reaches = sorted(m["per_reach"])
    observable = [r for r in reaches if m["per_reach"][r]["ok"] > 0][:5]
    blind = [r for r in reaches if m["per_reach"][r]["ok"] == 0][:3]
    for rid in [*observable, *blind]:
        with xr.open_dataset(NH_DIR / "time_series" / f"{rid}.nc") as ds:
            s = ds.to_dataframe()
        counts = check_target_nans(s, TARGETS, rid)
        assert counts["ok"] == m["per_reach"][rid]["ok"]
        if rid in blind:
            assert s["turbidity_log1p"].isna().all()


@needs_export
def test_export_on_disk_statics_used_have_no_nulls_on_simulated_reaches() -> None:
    m = json.loads((NH_DIR / "export_manifest.json").read_text())
    a = pd.read_csv(NH_DIR / "attributes" / "attributes.csv", dtype={"reach_id": str})
    a = a.set_index("reach_id")
    alls = (NH_DIR / "all_reaches.txt").read_text().split()
    assert not a.loc[alls, m["static_attributes_used"]].isna().any().any()


# ---------------------------------------------------------------------------
# windows + forecast rows
# ---------------------------------------------------------------------------
def test_window_with_missing_driver_is_invalid_not_filled() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    X = np.ones((20, 2))
    X[5, 1] = np.nan
    w, ok = ealstm.build_windows(
        X, dates, [date(2024, 1, 4), date(2024, 1, 8), date(2024, 1, 20)], 4
    )
    assert ok.tolist() == [True, False, True]  # the 2nd window contains day 6 (NaN)
    assert not np.isnan(w).any()
    _, ok2 = ealstm.build_windows(X, dates, [date(2024, 1, 2)], 4)  # before the series
    assert not ok2[0]


def _sim() -> pd.DataFrame:
    d = pd.to_datetime(["2024-12-29", "2025-01-02", "2025-01-15"])
    return pd.DataFrame(
        {
            "variable": "ndci",
            "reach_id": "CMB-0001",
            "date": d.date,
            "observed": [0.1, 0.2, 0.3],
            "reach_observable": True,
            "split": "train",
            "model_version": "v",
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.1,
            "p50": 0.2,
            "p75": 0.3,
            "p90": 0.4,
            "p95": 0.5,
        }
    )


def test_forecast_rows_respect_fold_embargo() -> None:
    folds = {
        "val": (date(2024, 1, 1), date(2024, 12, 31)),
        "test": (date(2025, 1, 1), date(2026, 9, 20)),
    }
    rows = ealstm.forecast_rows(_sim(), range(1, 11), folds)
    # no val row may point past 2024-12-31; every row's fold is its ISSUE date's
    val = rows[rows["fold"] == "val"]
    assert (pd.to_datetime(val["target_date"]) <= "2024-12-31").all()
    assert (pd.to_datetime(val["issued_date"]) <= "2024-12-31").all()
    test = rows[rows["fold"] == "test"]
    assert (pd.to_datetime(test["issued_date"]) >= "2025-01-01").all()
    # 2025-01-02 target: h=1 issued 2025-01-01 (test); h>=2 issued in 2024 but the target
    # is past the val fold's end -> embargoed, dropped
    t2 = rows[rows["target_date"] == date(2025, 1, 2)]
    assert t2["horizon"].tolist() == [1] and t2["fold"].tolist() == ["test"]
    assert set(rows["weather"]) == {ealstm.WEATHER_HINDCAST}
    # issued = target - horizon, always
    lag = pd.to_datetime(rows["target_date"]) - pd.to_datetime(rows["issued_date"])
    assert (lag.dt.days == rows["horizon"]).all()
