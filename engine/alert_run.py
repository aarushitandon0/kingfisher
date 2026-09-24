"""A live alert run - pure: DataFrames in, engine.alerts outcomes out. No I/O, no DB.

models/alert_run.py loads the inputs and persists the result; everything that decides an
outcome is here, so the run is testable without a server (tests/test_alert_run.py).

For every (reach, variable) of the production forecast:

  forecasts      the window's daily quantile forecasts, each with its own seasonal
                 threshold (a window can straddle a season boundary)
  ReachState     evidence for the staleness guardrail. An observable reach's evidence is
                 its own OK observations of that variable. A driver-only reach's is the
                 OK observations of its DIRECT upstream reaches - the ones its
                 upstream_state_lag1 feature is built from (pipeline.l2_drivers) - and
                 evidence_source says so.
  residuals      drift guardrail: past forecasts for dates in the drift window that were
                 later observed. Preference, per target date: a production forecast over
                 a walk-forward one (the production model is in-sample on its own
                 training period, so until it has issued out-of-sample forecasts the
                 walk-forward test fold is the only honest residual), then the shortest
                 lead, then the weather it would have had live (LIVE, ASISSUED, ORACLE).
                 Only target dates strictly before the issue date - no future data.
  history        earlier runs' ALERT/WATCH rows (cooldown)
  attribution    TreeSHAP of the P90 model at the alert's PEAK target date - the upper
                 tail is what crosses the threshold. reach_id and horizon are never
                 presented as drivers (engine.alerts.build_attribution drops them).
  exposure       engine.exposure.summarise_exposure; None if the reach has no rows

compose_alert decides. Suppressed outcomes are not alert rows (they have no severity);
the run counts them per guardrail so they are reported, not lost.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from engine.alerts import (
    Alert,
    ForecastDistribution,
    GuardrailConfig,
    HistoryEntry,
    ReachState,
    Residual,
    Suppressed,
    build_attribution,
    compose_alert,
)
from engine.exposure import summarise_exposure
from engine.probability import DEFAULT_LEVELS
from engine.thresholds import lookup as threshold_lookup

QUANTILE_COLUMNS = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
ATTRIBUTION_QUANTILE = "p90"
FIT_PREFERENCE = {"production": 0}  # anything else (wf-*) ranks after
WEATHER_PREFERENCE = {"LIVE": 0, "ASISSUED": 1, "ORACLE": 2}


@dataclass(frozen=True)
class RunInputs:
    issued: date
    forecasts: pd.DataFrame  # api.repository.FORECAST_COLUMNS, one issue date, one model
    thresholds: pd.DataFrame  # engine.thresholds table, fitted to <= issued
    seasons: Mapping[str, list[int]]
    reaches: pd.DataFrame  # reach_id, observable
    upstream: Mapping[str, Sequence[str]]  # reach -> its direct upstream reaches
    observations: pd.DataFrame  # OK only: reach_id, date, variable, value
    past_forecasts: pd.DataFrame  # reach_id, variable, issued_date, target_date, horizon,
    #                               fit, weather, p10, p50, p90, model_version
    history: Sequence[HistoryEntry]
    shap: pd.DataFrame  # production rows at ATTRIBUTION_QUANTILE: reach_id, variable,
    #                     target_date, shap__<f>, value__<f>
    exposure: pd.DataFrame  # exposure_features rows
    threshold_derivations: Mapping[str, str]  # variable -> text shown on the alert


@dataclass(frozen=True)
class RunResult:
    issued: date
    model_version: str
    weather: str
    outcomes: tuple[Alert | Suppressed, ...]

    @property
    def alerts(self) -> list[Alert]:
        return [o for o in self.outcomes if isinstance(o, Alert)]

    @property
    def counts(self) -> dict[str, int]:
        c = Counter(str(a.severity) for a in self.alerts)
        return {s: c.get(s, 0) for s in ("ALERT", "WATCH", "INSUFFICIENT_EVIDENCE")}

    @property
    def suppressed(self) -> dict[str, int]:
        return dict(Counter(o.guardrail for o in self.outcomes if isinstance(o, Suppressed)))


def issued_at(d: date) -> datetime:
    """Runs are issued at 00 UTC of the issue date - the ECMWF run the forecast used."""
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _obs_dates(obs: pd.DataFrame, reach_ids: Sequence[str], variable: str) -> pd.Series:
    sub = obs[(obs["variable"] == variable) & obs["reach_id"].isin(list(reach_ids))]
    return pd.to_datetime(sub["date"]).dt.date


def reach_state(
    reach_id: str,
    variable: str,
    observable: bool | None,
    inp: RunInputs,
    residuals: tuple[Residual, ...],
) -> ReachState:
    if observable is True:
        sources, evidence = [reach_id], "reach"
    else:
        sources, evidence = list(inp.upstream.get(reach_id, ())), "upstream"
    dates = _obs_dates(inp.observations, sources, variable)
    dates = dates[dates <= inp.issued]
    last = max(dates) if len(dates) else None
    recent = int(((dates > inp.issued - timedelta(days=30)) & (dates <= inp.issued)).sum())
    return ReachState(
        reach_id=reach_id,
        observable=observable,
        last_usable_observation=last,
        evidence_source=evidence,
        usable_observations_30d=recent,
        recent_residuals=residuals if observable is True else (),
    )


def select_residual_forecasts(past: pd.DataFrame, issued: date, window_days: int) -> pd.DataFrame:
    """One forecast per (reach, variable, target date) in (issued - window, issued)."""
    if past.empty:
        return past
    t = pd.to_datetime(past["target_date"]).dt.date
    p = past[(t < issued) & (t > issued - timedelta(days=window_days))].copy()
    if p.empty:
        return p
    if (pd.to_datetime(p["issued_date"]).dt.date >= issued).any():
        raise ValueError("a residual forecast was issued on/after the run it judges")
    p["_fit"] = p["fit"].map(FIT_PREFERENCE).fillna(1)
    p["_wx"] = p["weather"].map(WEATHER_PREFERENCE).fillna(len(WEATHER_PREFERENCE))
    p = p.sort_values(["_fit", "horizon", "_wx"], kind="stable")
    p = p.drop_duplicates(["reach_id", "variable", "target_date"], keep="first")
    return p.drop(columns=["_fit", "_wx"])


def residuals_for(
    selected: pd.DataFrame, obs: pd.DataFrame
) -> dict[tuple[str, str], tuple[Residual, ...]]:
    if selected.empty:
        return {}
    o = obs.assign(target_date=pd.to_datetime(obs["date"]).dt.date)[
        ["reach_id", "variable", "target_date", "value"]
    ]
    s = selected.assign(target_date=pd.to_datetime(selected["target_date"]).dt.date)
    m = s.merge(o, on=["reach_id", "variable", "target_date"], how="inner")
    m = m.dropna(subset=["value", "p10", "p50", "p90"])
    out: dict[tuple[str, str], list[Residual]] = {}
    for r in m.itertuples(index=False):
        out.setdefault((str(r.reach_id), str(r.variable)), []).append(
            Residual(
                obs_date=r.target_date,
                observed=float(r.value),
                p10=float(r.p10),
                p50=float(r.p50),
                p90=float(r.p90),
            )
        )
    return {k: tuple(sorted(v, key=lambda x: x.obs_date)) for k, v in out.items()}


def residual_source(selected: pd.DataFrame) -> dict[str, int]:
    """fit/weather -> residual forecasts used, for the alert's basis."""
    if selected.empty:
        return {}
    return {f"{f}/{w}": int(n) for (f, w), n in selected.groupby(["fit", "weather"]).size().items()}


def attribution_for(
    shap: pd.DataFrame, reach_id: str, variable: str, target: date, top_k: int
) -> tuple[Any, ...]:
    if shap.empty:
        return ()
    row = shap[
        (shap["reach_id"].astype(str) == reach_id)
        & (shap["variable"] == variable)
        & (pd.to_datetime(shap["target_date"]).dt.date == target)
    ]
    if row.empty:
        return ()
    if len(row) > 1:
        raise ValueError(f"{len(row)} SHAP rows for {reach_id}/{variable}/{target}")
    r = row.iloc[0]
    names = [c.removeprefix("shap__") for c in shap.columns if c.startswith("shap__")]
    values: list[float | None] = []
    for f in names:
        v = r.get(f"value__{f}")
        try:
            fv = float(v)
            values.append(None if math.isnan(fv) else fv)
        except (TypeError, ValueError):
            values.append(None)
    return tuple(
        build_attribution([float(r[f"shap__{f}"]) for f in names], names, values, top_k=top_k)
    )


def run_alerts(inp: RunInputs, cfg: GuardrailConfig, *, top_k_drivers: int = 5) -> RunResult:
    fc = inp.forecasts
    if fc.empty:
        raise ValueError("no production forecast to alert on")
    for col in ("issued_date", "model_version", "weather"):
        if fc[col].nunique() != 1:
            raise ValueError(f"forecasts span {fc[col].nunique()} values of {col}")
    if pd.to_datetime(fc["issued_date"]).dt.date.iloc[0] != inp.issued:
        raise ValueError("forecasts are not from the run's issue date")
    model_version = str(fc["model_version"].iloc[0])
    at = issued_at(inp.issued)

    thr = threshold_lookup(
        fc[["reach_id", "variable", "target_date"]], inp.thresholds, dict(inp.seasons)
    )
    fc = fc.assign(_thr=thr["threshold"].to_numpy(dtype="float64"))
    selected = select_residual_forecasts(inp.past_forecasts, inp.issued, cfg.drift_window_days)
    residuals = residuals_for(selected, inp.observations)
    res_source = residual_source(selected)
    observable = {
        str(r.reach_id): (None if pd.isna(r.observable) else bool(r.observable))
        for r in inp.reaches.itertuples(index=False)
    }

    outcomes: list[Alert | Suppressed] = []
    for (rid, var), g in fc.groupby(["reach_id", "variable"], sort=True):
        rid, var = str(rid), str(var)
        if rid not in observable:
            raise KeyError(f"forecast for unknown reach {rid}")
        g = g.sort_values("target_date")
        dists = [
            ForecastDistribution(
                quantiles=tuple(float(x) for x in row),
                levels=DEFAULT_LEVELS,
                target_date=pd.Timestamp(d).date(),
            )
            for row, d in zip(
                g[list(QUANTILE_COLUMNS)].to_numpy(dtype="float64"), g["target_date"], strict=True
            )
        ]
        thresholds = [None if np.isnan(t) else float(t) for t in g["_thr"]]
        state = reach_state(rid, var, observable[rid], inp, residuals.get((rid, var), ()))
        try:
            exposure = summarise_exposure(inp.exposure, rid)
        except LookupError:
            exposure = None
        out = compose_alert(
            reach_id=rid,
            variable=var,
            issued_at=at,
            forecasts=dists,
            thresholds=thresholds,
            threshold_derivation=inp.threshold_derivations[var],
            reach_state=state,
            history=[h for h in inp.history if h.issued_at < at],
            exposure=exposure,
            model_version=model_version,
            config=cfg,
        )
        if isinstance(out, Alert):
            extra = {
                "forecast_weather": str(fc["weather"].iloc[0]),
                "drift_residuals": len(state.recent_residuals),
                "drift_residual_source": res_source,
            }
            attribution = out.attribution
            if out.peak_target_date is not None:
                attribution = attribution_for(
                    inp.shap, rid, var, out.peak_target_date, top_k_drivers
                )
                extra["attribution"] = (
                    f"TreeSHAP of the {ATTRIBUTION_QUANTILE.upper()} model at the peak "
                    f"target date {out.peak_target_date}, in the target's units"
                )
            out = replace(out, attribution=attribution, basis={**out.basis, **extra})
        outcomes.append(out)
    return RunResult(
        issued=inp.issued,
        model_version=model_version,
        weather=str(fc["weather"].iloc[0]),
        outcomes=tuple(outcomes),
    )
