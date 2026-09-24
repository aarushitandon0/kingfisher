"""engine/alert_run.py - the live alert run, without a server or a database."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

from engine.alert_run import (
    RunInputs,
    issued_at,
    reach_state,
    run_alerts,
    select_residual_forecasts,
)
from engine.alerts import Alert, GuardrailConfig, HistoryEntry, Severity

ISSUED = date(2026, 9, 20)
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
OBS, DRV, LOW = "CMB-0001", "CMB-0002", "CMB-0003"  # observable, driver-only, observable
QCOLS = ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]
CFG = GuardrailConfig()


def _forecasts(quantiles: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for rid, q in quantiles.items():
        for h in range(1, 4):
            rows.append(
                {
                    "reach_id": rid,
                    "issued_date": ISSUED,
                    "target_date": ISSUED + timedelta(days=h),
                    "horizon": h,
                    "variable": "turbidity_proxy",
                    **dict(zip(QCOLS, q, strict=True)),
                    "model_version": "gbm-test+coimbra.A.production.0",
                    "variant": "A",
                    "weather": "LIVE",
                    "future_drivers_missing": 0,
                }
            )
    return pd.DataFrame(rows)


def _inputs(**over: object) -> RunInputs:
    high = [20, 22, 25, 30, 35, 40, 45]  # every quantile above the threshold of 10
    low = [1, 2, 3, 4, 5, 6, 7]
    obs = pd.DataFrame(
        [
            {
                "reach_id": r,
                "date": ISSUED - timedelta(days=d),
                "variable": "turbidity_proxy",
                "value": 5.0,
            }
            for r in (OBS, LOW)
            for d in (0, 5, 10)
        ]
    )
    shap = pd.DataFrame(
        [
            {
                "reach_id": OBS,
                "variable": "turbidity_proxy",
                "target_date": ISSUED + timedelta(days=h),
                "shap__reach_id": 99.0,  # the reach's offset - never a driver
                "shap__horizon": 50.0,
                "shap__precip_mm_fut": 3.0 * h,
                "shap__api_7": -1.0,
                "value__horizon": float(h),
                "value__precip_mm_fut": 12.0,
                "value__api_7": 0.5,
            }
            for h in range(1, 4)
        ]
    )
    base = RunInputs(
        issued=ISSUED,
        forecasts=_forecasts({OBS: high, DRV: high, LOW: low}),
        thresholds=pd.DataFrame(
            [
                {
                    "reach_id": r,
                    "variable": "turbidity_proxy",
                    "season": "SON",
                    "threshold": 10.0,
                    "n_obs": 20,
                    "clim_exceed_freq": 0.1,
                    "percentile": 0.9,
                    "fit_end": ISSUED,
                }
                for r in (OBS, LOW)
            ]
        ),
        seasons=SEASONS,
        reaches=pd.DataFrame({"reach_id": [OBS, DRV, LOW], "observable": [True, False, True]}),
        upstream={DRV: [OBS]},
        observations=obs,
        past_forecasts=pd.DataFrame(
            columns=[
                "reach_id",
                "variable",
                "issued_date",
                "target_date",
                "horizon",
                "fit",
                "weather",
                "p10",
                "p50",
                "p90",
                "model_version",
            ]
        ),
        history=(),
        shap=shap,
        exposure=pd.DataFrame(
            [
                {
                    "reach_id": OBS,
                    "feature_type": "school",
                    "count": 2,
                    "nearest_distance_m": 80.0,
                    "buffer_m": 250.0,
                    "source": "osm",
                }
            ]
        ),
        threshold_derivations={"turbidity_proxy": "fixture derivation"},
    )
    return replace(base, **over)  # type: ignore[arg-type]


def _by_reach(result):  # type: ignore[no-untyped-def]
    return {o.reach_id: o for o in result.outcomes}


def test_run_issues_alert_with_peak_attribution_and_exposure() -> None:
    out = _by_reach(run_alerts(_inputs(), CFG))[OBS]
    assert isinstance(out, Alert) and out.severity is Severity.ALERT
    assert out.exceedance_prob is not None and out.exceedance_prob > 0.95
    features = [d.feature for d in out.attribution]
    assert "reach_id" not in features and "horizon" not in features
    # attribution is taken at the peak target date: precip contribution 3 * h
    assert out.attribution[0].feature == "precip_mm_fut"
    assert out.attribution[0].contribution == 3.0 * (out.peak_target_date - ISSUED).days
    assert out.exposure is not None and out.exposure["features"]["school"]["count"] == 2
    assert out.basis["forecast_weather"] == "LIVE"
    assert out.threshold_derivation == "fixture derivation"


def test_driver_only_reach_without_threshold_is_insufficient_evidence_not_an_alert() -> None:
    out = _by_reach(run_alerts(_inputs(), CFG))[DRV]
    assert isinstance(out, Alert)
    assert out.severity is Severity.INSUFFICIENT_EVIDENCE
    assert out.exceedance_prob is None and out.suppressed_reason.startswith("NO_THRESHOLD")


def test_driver_only_reach_takes_its_evidence_from_direct_upstream_reaches() -> None:
    inp = _inputs()
    s = reach_state(DRV, "turbidity_proxy", False, inp, ())
    assert s.evidence_source == "upstream" and s.last_usable_observation == ISSUED
    assert s.usable_observations_30d == 3
    orphan = reach_state(DRV, "turbidity_proxy", False, replace(inp, upstream={}), ())
    assert orphan.last_usable_observation is None


def test_suppressed_candidates_are_counted_not_written_as_alerts() -> None:
    result = run_alerts(_inputs(), CFG)
    assert LOW not in {a.reach_id for a in result.alerts}
    assert result.suppressed == {"confidence": 1}
    assert result.counts == {"ALERT": 1, "WATCH": 0, "INSUFFICIENT_EVIDENCE": 1}


def test_future_observations_are_not_evidence() -> None:
    inp = _inputs()
    future = pd.DataFrame(
        [
            {
                "reach_id": OBS,
                "date": ISSUED + timedelta(days=1),
                "variable": "turbidity_proxy",
                "value": 1.0,
            }
        ]
    )
    inp = replace(inp, observations=pd.concat([inp.observations, future], ignore_index=True))
    assert reach_state(OBS, "turbidity_proxy", True, inp, ()).last_usable_observation == ISSUED


def test_cooldown_uses_earlier_runs_only() -> None:
    same_day = HistoryEntry(OBS, issued_at(ISSUED), Severity.ALERT)
    rerun = _by_reach(run_alerts(_inputs(history=(same_day,)), CFG))[OBS]
    assert isinstance(rerun, Alert)  # a re-run of the same issue date does not cool itself
    yesterday = HistoryEntry(OBS, issued_at(ISSUED - timedelta(days=1)), Severity.ALERT)
    cooled = _by_reach(run_alerts(_inputs(history=(yesterday,)), CFG))[OBS]
    assert not isinstance(cooled, Alert) and cooled.guardrail == "cooldown"


def test_residuals_prefer_production_then_shortest_lead_and_never_the_future() -> None:
    def row(fit: str, h: int, target: date, weather: str = "ORACLE") -> dict[str, object]:
        return {
            "reach_id": OBS,
            "variable": "turbidity_proxy",
            "issued_date": target - timedelta(days=h),
            "target_date": target,
            "horizon": h,
            "fit": fit,
            "weather": weather,
            "p10": 1.0,
            "p50": 2.0,
            "p90": 3.0,
            "model_version": fit,
        }

    t = ISSUED - timedelta(days=5)
    past = pd.DataFrame(
        [
            row("wf-test", 1, t),
            row("production", 3, t, "LIVE"),
            row("wf-test", 2, t),
            row("wf-test", 1, ISSUED),
            row("wf-test", 1, ISSUED - timedelta(days=90)),
        ]
    )
    sel = select_residual_forecasts(past, ISSUED, 60)
    assert len(sel) == 1 and sel.iloc[0]["fit"] == "production"
    wf = select_residual_forecasts(past[past["fit"] != "production"], ISSUED, 60)
    assert len(wf) == 1 and wf.iloc[0]["horizon"] == 1


def test_drift_guardrail_fires_on_the_run() -> None:
    days = [ISSUED - timedelta(days=d) for d in (3, 6, 9, 12)]
    obs = pd.DataFrame(
        [{"reach_id": OBS, "date": d, "variable": "turbidity_proxy", "value": 50.0} for d in days]
    )
    past = pd.DataFrame(
        [
            {
                "reach_id": OBS,
                "variable": "turbidity_proxy",
                "issued_date": d - timedelta(days=1),
                "target_date": d,
                "horizon": 1,
                "fit": "wf-test",
                "weather": "ORACLE",
                "p10": 1.0,
                "p50": 2.0,
                "p90": 3.0,
                "model_version": "wf",
            }
            for d in days
        ]
    )
    out = _by_reach(run_alerts(_inputs(observations=obs, past_forecasts=past), CFG))[OBS]
    assert not isinstance(out, Alert) and out.guardrail == "drift"


def test_run_refuses_mixed_forecasts() -> None:
    inp = _inputs()
    fc = inp.forecasts.copy()
    fc.loc[0, "model_version"] = "other"
    with pytest.raises(ValueError, match="model_version"):
        run_alerts(replace(inp, forecasts=fc), CFG)
    with pytest.raises(ValueError, match="no production forecast"):
        run_alerts(replace(inp, forecasts=fc.iloc[0:0]), CFG)
