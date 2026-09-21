"""P5.3 - the weather-explained anomaly detector.

The headline pair: a synthetic spike on a dry day must be UNEXPLAINED; the SAME spike the
day after a large synthetic storm must be WEATHER_EXPLAINED. The forecast comes from a
toy driver -> state model (state rises with antecedent rain), turned into quantiles and
into a CDF by engine.probability, exactly as the detector does for LightGBM. For the
EA-LSTM path the same check runs through the exact CMAL CDF.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from engine.probability import DEFAULT_LEVELS
from models import anomaly as A
from models import cmal

S = A.AnomalySettings(
    pit_threshold=0.975,
    p90_margin_fraction=0.25,
    match_window_days=3,
    match_downstream_reaches=3,
    min_incidents=10,
    proxy_percentile=0.95,
    bootstrap_resamples=300,
    bootstrap_seed=1,
)
TR = {"ndci": "identity", "turbidity_proxy": "log1p"}


def toy_model(api_7_mm: float) -> tuple[float, float]:
    """Toy driver -> state simulator, turbidity in log1p space: wetter antecedent
    conditions -> higher, more uncertain state."""
    return np.log1p(8.0) + 0.04 * api_7_mm, 0.25 + 0.004 * api_7_mm


def _forecast_row(api: float, observed: float, threshold: float = 20.0) -> dict:
    mu, sd = toy_model(api)
    q_t = mu + sd * norm.ppf(DEFAULT_LEVELS)
    q = np.expm1(q_t)
    pit = A.pit_from_quantiles(q[None, :], DEFAULT_LEVELS, np.array([observed]))[0]
    return {
        "variable": "turbidity_proxy",
        "reach_id": "CMB-0001",
        "date": date(2024, 6, 1),
        "observed": observed,
        "pit": pit,
        "p10": q[1],
        "p50": q[3],
        "p90": q[5],
        "threshold": threshold,
    }


SPIKE = 45.0  # turbidity index; the reach's seasonal threshold is 20


def test_dry_day_spike_is_unexplained() -> None:
    out = A.classify(pd.DataFrame([_forecast_row(api=0.0, observed=SPIKE)]), S, TR)
    assert out["label"].iloc[0] == A.UNEXPLAINED
    assert out["anomaly_score"].iloc[0] > -np.log(1 - 0.975)


def test_same_spike_after_storm_is_weather_explained() -> None:
    out = A.classify(pd.DataFrame([_forecast_row(api=40.0, observed=SPIKE)]), S, TR)
    assert out["high"].iloc[0]  # still a high reading...
    assert out["label"].iloc[0] == A.WEATHER_EXPLAINED  # ...but the weather explains it


def test_unremarkable_reading_is_normal() -> None:
    out = A.classify(pd.DataFrame([_forecast_row(api=0.0, observed=9.0)]), S, TR)
    assert out["label"].iloc[0] == A.NORMAL


def test_extreme_pit_without_margin_is_not_an_anomaly() -> None:
    """PIT > 0.975 alone is not enough: the value must clear P90 by the margin."""
    row = _forecast_row(api=0.0, observed=SPIKE)
    row["pit"] = 0.99
    row["observed"] = row["p90"] * 1.001  # barely above P90
    out = A.classify(pd.DataFrame([row]), S, TR)
    assert out["label"].iloc[0] != A.UNEXPLAINED


def test_no_forecast_gives_no_label_never_normal() -> None:
    row = _forecast_row(api=0.0, observed=SPIKE)
    row["pit"] = np.nan
    out = A.classify(pd.DataFrame([row]), S, TR)
    assert out["label"].iloc[0] is None


def test_storm_vs_dry_through_exact_cmal_cdf() -> None:
    """The EA-LSTM path: PIT from the mixture CDF (a single-component ALD mixture
    centred on the toy model's median)."""
    rows = []
    for api in (0.0, 40.0):
        mu, sd = toy_model(api)
        mix = cmal.Mixture(
            np.array([[mu]]), np.array([[sd * 0.7]]), np.array([[0.5]]), np.array([[1.0]])
        )
        q = np.expm1(cmal.quantiles(mix, (0.1, 0.5, 0.9))[0])
        rows.append(
            {
                "variable": "turbidity_proxy",
                "reach_id": "CMB-0001",
                "date": date(2024, 6, 1),
                "observed": SPIKE,
                "pit": float(cmal.cdf(mix, np.log1p([SPIKE]))[0]),
                "p10": q[0],
                "p50": q[1],
                "p90": q[2],
                "threshold": 20.0,
            }
        )
    out = A.classify(pd.DataFrame(rows), S, TR)
    assert out["label"].tolist() == [A.UNEXPLAINED, A.WEATHER_EXPLAINED]


def test_score_is_uncapped() -> None:
    assert np.isinf(A.score(np.array([1.0]))[0])
    assert A.score(np.array([0.5]))[0] == pytest.approx(np.log(2))


# ---------------------------------------------------------------------------
# localisation - a label, never an attribution
# ---------------------------------------------------------------------------
def _lab(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": "ndci", "reach_id": r, "date": date(2024, 6, 1), "label": lab, "pit": p}
            for r, lab, p in rows
        ]
    )


def test_localised_when_upstream_is_quiet_same_day() -> None:
    lab = _lab([("B", A.UNEXPLAINED, 0.999), ("A", A.NORMAL, 0.4)])
    loc = A.localise(lab, {"B": ["A"]}, 0.975)
    assert loc.iloc[0] is True and loc.iloc[1] is None


def test_not_localised_when_upstream_also_unexplained() -> None:
    lab = _lab([("B", A.UNEXPLAINED, 0.999), ("A", A.UNEXPLAINED, 0.999)])
    assert A.localise(lab, {"B": ["A"]}, 0.975).iloc[0] is False


def test_unknown_when_no_upstream_observed_that_day() -> None:
    lab = _lab([("B", A.UNEXPLAINED, 0.999)])
    assert A.localise(lab, {"B": ["A"]}, 0.975).iloc[0] is None


def test_output_never_names_a_source() -> None:
    lab = A.classify(pd.DataFrame([_forecast_row(0.0, SPIKE)]), S, TR)
    banned = {"source", "polluter", "cause", "culprit", "discharger"}
    assert not banned & {c.lower() for c in lab.columns}


# ---------------------------------------------------------------------------
# matching + scoring
# ---------------------------------------------------------------------------
DOWN = {"R1": ["R2"], "R2": ["R3"], "R3": ["R4"], "R4": ["R5"]}


def _df(rows: list[tuple[str, date]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["reach_id", "date"])


def test_match_window_and_downstream_hops() -> None:
    d0 = date(2024, 6, 10)
    events = _df([("R1", d0)])
    dets = _df(
        [
            ("R1", d0 + timedelta(days=3)),  # same reach, 3 days: match
            ("R4", d0 - timedelta(days=1)),  # 3 hops downstream: match
            ("R5", d0),  # 4 hops: no
            ("R1", d0 + timedelta(days=4)),  # 4 days: no
        ]
    )
    d_ok, e_ok = A.match(dets, events, DOWN, window_days=3, k_downstream=3)
    assert d_ok.tolist() == [True, True, False, False]
    assert e_ok.tolist() == [True]
    m = A.prf(d_ok, e_ok)
    assert m["precision"] == 0.5 and m["recall"] == 1.0
    assert m["f1"] == pytest.approx(2 * 0.5 / 1.5)


def test_upstream_detection_does_not_match() -> None:
    d0 = date(2024, 6, 10)
    d_ok, _ = A.match(_df([("R1", d0)]), _df([("R2", d0)]), DOWN, window_days=3, k_downstream=3)
    assert not d_ok[0]


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    det_reach = rng.choice([f"R{i}" for i in range(20)], 200)
    det_ok = rng.random(200) < 0.6
    ev_reach = rng.choice([f"R{i}" for i in range(20)], 100)
    ev_ok = rng.random(100) < 0.4
    ci = A.bootstrap_prf(det_reach, det_ok, ev_reach, ev_ok, n=500, seed=1)
    p = A.prf(det_ok, ev_ok)
    assert ci["precision"][0] <= p["precision"] <= ci["precision"][1]
    assert ci["recall"][0] <= p["recall"] <= ci["recall"][1]


def test_proxy_events_use_only_the_fitting_period() -> None:
    from core.config import load_config

    thr = load_config("thresholds")
    rng = np.random.default_rng(3)
    days = pd.date_range("2022-01-01", "2024-12-31", freq="5D")
    obs = pd.DataFrame(
        {
            "reach_id": "R1",
            "variable": "ndci",
            "date": days.date,
            "observed": rng.normal(0, 1, len(days)),
        }
    )
    ev = A.proxy_events(obs, date(2023, 12, 31), 0.95, thr)
    assert (pd.to_datetime(ev["date"]) > "2023-12-31").all()
    # moving post-fit values does not move the thresholds that judge them
    obs2 = obs.copy()
    late = pd.to_datetime(obs2["date"]) > "2023-12-31"
    obs2.loc[late, "observed"] += 100
    ev2 = A.proxy_events(obs2, date(2023, 12, 31), 0.95, thr)
    assert len(ev2) >= len(ev) and len(ev2) == int(late.sum())
