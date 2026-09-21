"""P5.2 - residual assimilation and calibration.

Leakage: no observation dated after the issue date may move a forecast. Proved by
perturbing every observation after each issue date and asserting the assimilated
forecasts do not change - and checked against a deliberately leaky variant so the test
cannot pass vacuously. Calibration: every fit refuses a test-fold row.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from models import assimilation as A
from models import calibration as C

TR = {"ndci": "identity", "turbidity_proxy": "log1p"}
BUCKETS = {"h01_03": [1, 2, 3], "h04_07": [4, 5, 6, 7], "h08_10": [8, 9, 10]}
VAL_END = date(2024, 12, 31)
Q = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
OFFS = (-1.6, -1.3, -0.7, 0.0, 0.7, 1.3, 1.6)


def _rows(
    reach: str = "CMB-0001",
    variable: str = "ndci",
    issued: list[date] | None = None,
    horizons: range = range(1, 11),
    center: float = 0.0,
    observable: bool | None = True,
    fold: str = "val",
) -> pd.DataFrame:
    issued = issued or [date(2024, 3, 1) + timedelta(days=i) for i in range(30)]
    recs = []
    for t in issued:
        for h in horizons:
            r = {
                "fold": fold,
                "variable": variable,
                "reach_id": reach,
                "issued_date": t,
                "horizon": h,
                "target_date": t + timedelta(days=h),
                "target": np.nan,
                "reach_observable": observable,
            }
            for c, o in zip(Q, OFFS, strict=True):
                r[c] = center + o
            recs.append(r)
    return pd.DataFrame(recs)


def _observe(rows: pd.DataFrame, values: dict[date, float]) -> pd.DataFrame:
    out = rows.copy()
    out["target"] = out["target_date"].map(values).astype("float64")
    return out


# ---------------------------------------------------------------------------
# residuals + the correction
# ---------------------------------------------------------------------------
def test_residual_uses_shortest_horizon_in_transformed_space() -> None:
    rows = _rows(variable="turbidity_proxy", center=np.log1p(9.0))
    rows[list(Q)] = np.expm1(rows[list(Q)])  # original units
    rows = _observe(rows, {date(2024, 3, 10): 19.0})
    res = A.residuals(rows, TR)
    r = res[res["obs_date"] == pd.Timestamp("2024-03-10")].iloc[0]
    assert r["source_horizon"] == 1
    assert r["r0"] == pytest.approx(np.log1p(19.0) - np.log1p(9.0))


def test_shift_decays_with_age_and_moves_every_quantile() -> None:
    rows = _observe(_rows(), {date(2024, 3, 5): 2.0})  # p50 is 0 -> r0 = 2
    res = A.residuals(rows, TR)
    out = A.assimilate(rows, res, 5.0, max_obs_age_days=21, transforms=TR)
    t = date(2024, 3, 7)
    for h in (1, 5, 10):
        row = out[(out["issued_date"] == t) & (out["horizon"] == h)].iloc[0]
        age = (t + timedelta(days=h) - date(2024, 3, 5)).days
        expect = 2.0 * np.exp(-age / 5.0)
        assert row["assimilated"] and row["assim_age_days"] == age
        assert [row[c] for c in Q] == pytest.approx([o + expect for o in OFFS])


def test_log1p_shift_is_multiplicative_in_original_units() -> None:
    rows = _rows(variable="turbidity_proxy", center=np.log1p(9.0))
    rows[list(Q)] = np.expm1(rows[list(Q)])
    rows = _observe(rows, {date(2024, 3, 5): 19.0})
    out = A.assimilate(rows, A.residuals(rows, TR), 1e9, max_obs_age_days=21, transforms=TR)
    row = out[(out["issued_date"] == date(2024, 3, 6)) & (out["horizon"] == 1)].iloc[0]
    assert row["p50"] == pytest.approx(19.0)  # tau -> inf: full correction
    assert row["p90"] > row["p50"] > row["p10"] > 0


def test_observation_on_the_issue_date_is_used_after_it_is_not() -> None:
    rows = _observe(_rows(), {date(2024, 3, 10): 3.0})
    out = A.assimilate(rows, A.residuals(rows, TR), 7.0, max_obs_age_days=21, transforms=TR)
    on = out[out["issued_date"] == date(2024, 3, 10)]
    before = out[out["issued_date"] == date(2024, 3, 9)]
    assert on["assimilated"].all()
    assert (before["assim_reason"] == A.NO_OBS).all()
    assert before["p50"].eq(0.0).all()


def test_stale_observation_gives_no_correction() -> None:
    rows = _observe(_rows(), {date(2024, 3, 2): 3.0})
    out = A.assimilate(rows, A.residuals(rows, TR), 7.0, max_obs_age_days=21, transforms=TR)
    t_ok, t_stale = date(2024, 3, 23), date(2024, 3, 24)  # 21 vs 22 days old at issue
    assert out[out["issued_date"] == t_ok]["assimilated"].all()
    s = out[out["issued_date"] == t_stale]
    assert (~s["assimilated"]).all() and (s["assim_reason"] == A.STALE).all()
    assert s["p50"].eq(0.0).all()


@pytest.mark.parametrize("flag", [False, None])
def test_unobservable_reach_is_never_corrected(flag: bool | None) -> None:
    rows = _observe(_rows(observable=flag), {date(2024, 3, 5): 3.0})
    out = A.assimilate(rows, A.residuals(rows, TR), 7.0, max_obs_age_days=21, transforms=TR)
    assert (~out["assimilated"]).all()
    assert (out["assim_reason"] == A.UNOBSERVABLE).all()
    assert out["p50"].eq(0.0).all()


def test_residual_from_another_fold_is_not_used() -> None:
    val = _observe(_rows(issued=[date(2024, 12, 20)], fold="val"), {date(2024, 12, 25): 3.0})
    test = _rows(issued=[date(2025, 1, 1)], fold="test")
    both = pd.concat([val, test], ignore_index=True)
    out = A.assimilate(both, A.residuals(both, TR), 7.0, max_obs_age_days=21, transforms=TR)
    t = out[out["fold"] == "test"]
    assert (~t["assimilated"]).all() and (t["assim_reason"] == A.NO_OBS).all()


def _random_table(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    issued = [date(2024, 2, 1) + timedelta(days=i) for i in range(80)]
    parts = []
    for reach in ("CMB-0001", "CMB-0002"):
        obs_days = sorted(rng.choice(np.arange(0, 90), 25, replace=False))
        vals = {
            date(2024, 2, 1) + timedelta(days=int(d)): float(rng.normal(0, 2)) for d in obs_days
        }
        parts.append(_observe(_rows(reach=reach, issued=issued, center=float(rng.normal())), vals))
    return pd.concat(parts, ignore_index=True)


def _perturb_after(rows: pd.DataFrame, cut: date, rng: np.random.Generator) -> pd.DataFrame:
    out = rows.copy()
    late = pd.to_datetime(out["target_date"]) > pd.Timestamp(cut)
    obs = late & out["target"].notna()
    out.loc[obs, "target"] = out.loc[obs, "target"] + rng.normal(50, 10, int(obs.sum()))
    return out


def _leaky_assimilate(rows: pd.DataFrame, res: pd.DataFrame, **kw: object) -> pd.DataFrame:
    """A deliberately broken assimilation that looks FORWARD for the observation."""
    shifted = rows.copy()
    shifted["issued_date"] = pd.to_datetime(shifted["issued_date"]) + pd.Timedelta(days=5)
    shifted["issued_date"] = shifted["issued_date"].dt.date
    out = A.assimilate(shifted, res, **kw)  # type: ignore[arg-type]
    out["issued_date"] = rows["issued_date"].to_numpy()
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_observation_after_issue_date_is_ever_used(seed: int) -> None:
    rng = np.random.default_rng(seed)
    table = _random_table(seed)
    kw = {"max_obs_age_days": 21, "transforms": TR}
    for cut in (date(2024, 2, 20), date(2024, 3, 10), date(2024, 4, 1)):
        base = A.assimilate(table, A.residuals(table, TR), 4.0, **kw)
        pert = _perturb_after(table, cut, rng)
        again = A.assimilate(pert, A.residuals(pert, TR), 4.0, **kw)
        early = (pd.to_datetime(base["issued_date"]) <= pd.Timestamp(cut)).to_numpy()
        assert np.allclose(base.loc[early, list(Q)], again.loc[early, list(Q)], equal_nan=True)
        # every correction that was applied used an observation on/before issue
        used = base["assimilated"].to_numpy()
        assert (
            pd.to_datetime(base.loc[used, "assim_obs_date"])
            <= pd.to_datetime(base.loc[used, "issued_date"])
        ).all()


def test_leakage_test_catches_a_forward_looking_assimilation() -> None:
    rng = np.random.default_rng(9)
    table = _random_table(9)
    cut = date(2024, 3, 10)
    kw = {"max_obs_age_days": 21, "transforms": TR}
    base = _leaky_assimilate(table, A.residuals(table, TR), tau=4.0, **kw)
    pert = _perturb_after(table, cut, rng)
    again = _leaky_assimilate(pert, A.residuals(pert, TR), tau=4.0, **kw)
    early = (pd.to_datetime(base["issued_date"]) <= pd.Timestamp(cut)).to_numpy()
    assert not np.allclose(base.loc[early, list(Q)], again.loc[early, list(Q)], equal_nan=True)


def test_fit_tau_recovers_true_decay() -> None:
    """Truth: the state's anomaly decays with e-folding time 7 days after each
    observation; the model's P50 knows nothing about it. fit_tau should pick ~7."""
    rng = np.random.default_rng(0)
    issued = [date(2024, 1, 1) + timedelta(days=i) for i in range(300)]
    rows = _rows(issued=issued)
    obs_days = [date(2024, 1, 1) + timedelta(days=int(d)) for d in range(0, 310, 6)]
    anomaly = {d: float(rng.normal(0, 3)) for d in obs_days}  # a fresh anomaly at each obs
    vals = {}
    for d in [date(2024, 1, 1) + timedelta(days=i) for i in range(320)]:
        prev = max((o for o in obs_days if o <= d), default=None)
        if prev is None:
            continue
        vals[d] = anomaly[prev] * np.exp(-(d - prev).days / 7.0)
    obs_only = {d: v for d, v in vals.items() if d in anomaly}
    scored = rows.copy()
    scored["target"] = scored["target_date"].map(vals).astype(float)
    res = A.residuals(_observe(rows, obs_only), TR)
    grid = (1.0, 3.0, 7.0, 14.0, 30.0)
    fit = A.fit_tau(
        scored, res, grid, max_obs_age_days=21, transforms=TR, val_end=date(2025, 1, 31)
    )
    assert fit["ndci"]["tau_days"] == 7.0
    assert fit["ndci"]["crps_best"] < fit["ndci"]["crps_no_assimilation"]


def test_fit_tau_refuses_test_rows() -> None:
    rows = _observe(_rows(fold="test", issued=[date(2025, 2, 1)]), {date(2025, 2, 3): 1.0})
    with pytest.raises(C.TestFoldTouched):
        A.fit_tau(
            rows, A.residuals(rows, TR), (1.0,), max_obs_age_days=21, transforms=TR, val_end=VAL_END
        )


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def _calib_rows(n: int, scale: float, seed: int, fold: str = "val") -> pd.DataFrame:
    """Intervals built for sd=1 noise; the truth has sd=scale."""
    rng = np.random.default_rng(seed)
    t0 = date(2024, 1, 1) if fold == "val" else date(2025, 1, 1)
    recs = []
    for i in range(n):
        h = int(rng.integers(1, 11))
        t = t0 + timedelta(days=int(rng.integers(0, 300)))
        r = {
            "fold": fold,
            "variable": "ndci",
            "reach_id": f"CMB-{i % 7:04d}",
            "issued_date": t,
            "horizon": h,
            "target_date": t + timedelta(days=h),
            "target": float(rng.normal(0, scale)),
            "reach_observable": True,
        }
        for c, z in zip(Q, (-1.645, -1.2816, -0.674, 0, 0.674, 1.2816, 1.645), strict=True):
            r[c] = z
        recs.append(r)
    df = pd.DataFrame(recs)
    return df[pd.to_datetime(df["target_date"]) <= pd.Timestamp(VAL_END)] if fold == "val" else df


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_conformal_restores_nominal_coverage(scale: float) -> None:
    val, test = _calib_rows(3000, scale, 1), _calib_rows(3000, scale, 2, fold="test")
    fit = C.fit_conformal(val, transforms=TR, buckets=BUCKETS, val_end=VAL_END)
    out = C.apply_conformal(test, fit, transforms=TR, buckets=BUCKETS)
    cov = np.mean((out["target"] >= out["p10"]) & (out["target"] <= out["p90"]))
    raw = np.mean((test["target"] >= test["p10"]) & (test["target"] <= test["p90"]))
    assert abs(raw - 0.8) > 0.15  # the raw intervals really are miscalibrated
    assert cov == pytest.approx(0.8, abs=0.03)
    assert (np.diff(out[list(Q)].to_numpy(), axis=1) >= 0).all()
    q = fit.cells[("ndci", "h01_03", "observable")].q_adjust
    assert (q > 0) if scale > 1 else (q < 0)


def test_conformal_fit_refuses_test_fold() -> None:
    val = _calib_rows(200, 1.0, 1)
    test = _calib_rows(50, 1.0, 2, fold="test")
    with pytest.raises(C.TestFoldTouched):
        C.fit_conformal(pd.concat([val, test]), transforms=TR, buckets=BUCKETS, val_end=VAL_END)
    sneaky = val.copy()
    sneaky.loc[sneaky.index[0], "target_date"] = date(2025, 1, 3)  # labelled val, dated test
    with pytest.raises(C.TestFoldTouched):
        C.fit_conformal(sneaky, transforms=TR, buckets=BUCKETS, val_end=VAL_END)


def test_small_or_empty_cells_are_not_adjusted_and_say_why() -> None:
    val = _calib_rows(60, 2.0, 1)
    fit = C.fit_conformal(val, transforms=TR, buckets=BUCKETS, val_end=VAL_END, min_rows=30)
    driver_only = fit.cells[("ndci", "h01_03", "driver_only")]
    assert driver_only.q_adjust is None and driver_only.reason == "no calibration rows"
    small = [c for c in fit.cells.values() if c.obs_class == "observable" and c.n < 30]
    assert small and all(c.q_adjust is None and "fewer than" in (c.reason or "") for c in small)


def test_isotonic_only_with_enough_events() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 400)
    few = rng.uniform(0, 1, 400) < 0.03
    mp, rec = C.fit_isotonic(p, few, min_events=20)
    assert mp is None and rec["status"] == C.INSUFFICIENT_EVENTS
    many = rng.uniform(0, 1, 400) < p**2  # overconfident probabilities
    mp, rec = C.fit_isotonic(p, many, min_events=20)
    assert mp is not None and rec["status"].startswith("calibrated")
    grid = np.linspace(0, 1, 50)
    assert (np.diff(mp(grid)) >= 0).all()
    assert mp(np.array([0.5]))[0] == pytest.approx(0.25, abs=0.1)
    assert np.isnan(mp(np.array([np.nan]))[0])


def test_isotonic_fit_refuses_test_fold() -> None:
    rows = _calib_rows(100, 1.0, 1, fold="test").assign(threshold=0.5)
    with pytest.raises(C.TestFoldTouched):
        C.fit_isotonic_by_class(rows, np.full(len(rows), 0.5), min_events=1, val_end=VAL_END)
