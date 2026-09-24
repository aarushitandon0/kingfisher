"""Reach routes (MASTERSPEC 12):

GET /api/reaches?city=coimbra              GeoJSON FeatureCollection + current status
GET /api/reaches/{id}                      detail, history, catchment, exposure
GET /api/reaches/{id}/forecast?horizon=10  P05-P95 series with threshold and P(exceed)
GET /api/reaches/{id}/catchment            upstream catchment polygon only (map hover)
GET /api/reaches/{id}/attribution          stored TreeSHAP of the production forecast
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import ShapSource, get_repository, get_shap_source
from api.repository import Repository
from api.schemas import (
    AttributionResponse,
    AttributionSeries,
    CatchmentFeature,
    DriverContributionOut,
    ExposureFeatureOut,
    ExposureOut,
    ForecastDay,
    ForecastResponse,
    ForecastSeries,
    Geometry,
    ObservationOut,
    ReachCollection,
    ReachDetail,
    ReachDetailProperties,
    ReachFeature,
    ReachProperties,
    ThresholdOut,
    observability,
)
from api.services import (
    FORECAST_VARIANT,
    QUANTILES,
    city_configs,
    nan_to_none,
    threshold_derivation,
    threshold_table,
    thresholds_config,
    with_exceedance,
)
from engine.alert_run import ATTRIBUTION_QUANTILE, attribution_for
from engine.exposure import summarise_exposure

router = APIRouter(tags=["reaches"])

MAX_HORIZON = 10
CATCHMENT_SIMPLIFY_DEG = 0.0003  # ~30 m: display only, areas come from the stored value
ATTRIBUTION_TOP_K = 8


def _peak(scored: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """reach -> variable -> peak daily P(exceed) over the window (None if no finite one)."""
    out: dict[str, dict[str, float | None]] = {}
    if scored.empty:
        return out
    peak = scored.groupby(["reach_id", "variable"], sort=False)["exceedance_prob"].max()
    for (rid, var), p in peak.items():  # max skips NaN; all-NaN -> NaN -> None
        out.setdefault(str(rid), {})[str(var)] = None if pd.isna(p) else float(p)
    return out


def _latest_alerts(repo: Repository, city: str) -> tuple[date | None, dict[str, dict[str, Any]]]:
    """The latest alert run's most severe alert per reach."""
    run = repo.alert_run_date(city)
    if run is None:
        return None, {}
    rank = {"ALERT": 0, "WATCH": 1, "INSUFFICIENT_EVIDENCE": 2}
    best: dict[str, dict[str, Any]] = {}
    for a in repo.alerts(city, run):
        cur = best.get(a["reach_id"])
        if cur is None or rank[a["severity"]] < rank[cur["severity"]]:
            best[a["reach_id"]] = a
    return run, best


def _properties(
    r: dict[str, Any],
    peak: dict[str, float | None] | None,
    has_forecast: bool,
    alert: dict[str, Any] | None,
) -> dict[str, Any]:
    pt = (peak or {}).get("turbidity_proxy")
    pn = (peak or {}).get("ndci")
    finite = [p for p in (pt, pn) if p is not None]
    if not has_forecast:
        status = "NO_FORECAST"
    elif not finite:
        status = "NO_THRESHOLD"
    else:
        status = "OK"
    return {
        "reach_id": r["reach_id"],
        "city": r["city"],
        "name": r["name"],
        "strahler_order": r["strahler_order"],
        "length_m": float(r["length_m"]),
        "catchment_area_km2": nan_to_none(r["catchment_area_km2"]),
        "observable": r["observable"],
        "observability": observability(r["observable"]),
        "median_water_pixels": nan_to_none(r["median_water_pixels"]),
        "exceedance_status": status,
        "p_exceed_turbidity_proxy": pt,
        "p_exceed_ndci": pn,
        "p_exceed_max": max(finite) if finite else None,
        "alert_severity": alert["severity"] if alert else None,
        "alert_id": alert["alert_id"] if alert else None,
    }


@router.get("/reaches", response_model=ReachCollection)
def list_reaches(
    city: str = Query(..., description="city id, e.g. coimbra"),
    repo: Repository = Depends(get_repository),
) -> ReachCollection:
    if city not in city_configs():
        raise HTTPException(404, f"unknown city {city!r}")
    rows = repo.reaches(city)
    if not rows:
        raise HTTPException(404, f"no reaches for {city} - run `make l0 city={city}`")
    fc = repo.latest_forecasts(city, FORECAST_VARIANT)
    issued = fc["issued_date"].iloc[0] if not fc.empty else None
    scored = with_exceedance(fc, threshold_table(repo, city, issued)) if issued else fc
    peaks = _peak(scored)
    forecasted = set(fc["reach_id"]) if not fc.empty else set()
    run, alerts = _latest_alerts(repo, city)
    return ReachCollection(
        city=city,
        forecast_issued_date=issued,
        forecast_model_version=str(fc["model_version"].iloc[0]) if not fc.empty else None,
        alert_run="OK" if run else "NO_ALERT_RUN",
        alert_run_issued_date=run,
        features=[
            ReachFeature(
                id=r["reach_id"],
                geometry=Geometry(**r["geometry"]),
                properties=ReachProperties(
                    **_properties(
                        r,
                        peaks.get(r["reach_id"]),
                        r["reach_id"] in forecasted,
                        alerts.get(r["reach_id"]),
                    )
                ),
            )
            for r in rows
        ],
    )


def _get_reach(repo: Repository, reach_id: str) -> tuple[dict[str, Any], str]:
    r = repo.reach(reach_id)
    if r is None:
        raise HTTPException(404, f"unknown reach {reach_id!r}")
    return r, str(r["city"])


@router.get("/reaches/{reach_id}", response_model=ReachDetail)
def reach_detail(
    reach_id: str,
    history_days: int = Query(
        365, ge=1, le=20000, description="days of history before the last observation"
    ),
    repo: Repository = Depends(get_repository),
) -> ReachDetail:
    r, city = _get_reach(repo, reach_id)
    fc = repo.latest_forecasts(city, FORECAST_VARIANT, reach_id)
    issued = fc["issued_date"].iloc[0] if not fc.empty else None
    last_obs = repo.last_observation_date(reach_id)
    fit_end = issued or last_obs
    table = (
        threshold_table(repo, city, fit_end)
        if fit_end
        else pd.DataFrame(columns=["reach_id", "variable", "season", "threshold"])
    )
    scored = with_exceedance(fc, table) if issued else fc
    _, alerts = _latest_alerts(repo, city)

    mine = table[table["reach_id"] == reach_id] if not table.empty else table
    thresholds = [
        ThresholdOut(
            variable=t.variable,
            season=t.season,
            threshold=float(t.threshold),
            n_obs=int(t.n_obs),
            percentile=float(t.percentile),
            fit_end=t.fit_end,
        )
        for t in mine.itertuples(index=False)
    ]
    start = last_obs - timedelta(days=history_days - 1) if last_obs else None
    history = (
        [
            ObservationOut(**{k: nan_to_none(v) for k, v in o.items()})
            for o in repo.observations(reach_id, start, last_obs)
        ]
        if last_obs
        else []
    )

    exp_rows = repo.exposure([reach_id])
    exposure = None
    if not exp_rows.empty:
        s = summarise_exposure(exp_rows, reach_id)
        exposure = ExposureOut(
            buffer_m=s.get("buffer_m"),
            population=s.get("population"),
            population_source=s.get("population_source"),
            features={k: ExposureFeatureOut(**v) for k, v in s["features"].items()},
        )
    attrs = repo.catchment_attributes(reach_id)
    base = _properties(r, _peak(scored).get(reach_id), not fc.empty, alerts.get(reach_id))
    return ReachDetail(
        id=reach_id,
        geometry=Geometry(**r["geometry"]),
        properties=ReachDetailProperties(
            **base,
            upstream_ids=r["upstream_ids"],
            downstream_ids=r["downstream_ids"],
            catchment_attributes=(
                {k: nan_to_none(v) for k, v in attrs.items()} if attrs is not None else None
            ),
            thresholds=thresholds,
            threshold_derivation=threshold_derivation(),
            exposure=exposure,
            history_start=start,
            history_end=last_obs,
            history=history,
        ),
        catchment=CatchmentFeature(
            id=f"{reach_id}-catchment",
            geometry=Geometry(**r["catchment_geometry"]) if r["catchment_geometry"] else None,
            properties={
                "reach_id": reach_id,
                "catchment_area_km2": nan_to_none(r["catchment_area_km2"]),
                "unit": "upstream contributing catchment",
            },
        ),
    )


@router.get("/reaches/{reach_id}/forecast", response_model=ForecastResponse)
def reach_forecast(
    reach_id: str,
    horizon: int = Query(MAX_HORIZON, ge=1, le=MAX_HORIZON),
    repo: Repository = Depends(get_repository),
) -> ForecastResponse:
    r, city = _get_reach(repo, reach_id)
    fc = repo.latest_forecasts(city, FORECAST_VARIANT, reach_id)
    if fc.empty:
        raise HTTPException(
            404, f"no production forecast on record for {reach_id} - run `make evaluate`"
        )
    issued = fc["issued_date"].iloc[0]
    fc = fc[fc["horizon"] <= horizon]
    scored = with_exceedance(fc, threshold_table(repo, city, issued))
    vcfg = thresholds_config()["variables"]
    notes: list[str] = []
    if r["observable"] is not True:
        notes.append(
            "Driver-predicted reach: not optically observable at 10 m, so it has no seasonal "
            "threshold and no exceedance probability; alerts here are capped at WATCH."
        )
    missing = scored["future_drivers_missing"].dropna()
    if len(missing) and (missing > 0).any():
        notes.append(
            f"{int((missing > 0).sum())} of {len(scored)} forecast rows had no weather for the "
            "target day (future driver features NULL) - issued without future weather, so "
            "weaker than a forecast driven by a weather forecast."
        )
    series = []
    for var, g in scored.groupby("variable", sort=True):
        series.append(
            ForecastSeries(
                variable=var,
                display_name=str(vcfg.get(var, {}).get("display_name", var)),
                units=str(vcfg.get(var, {}).get("units", "")),
                threshold_derivation=threshold_derivation(str(var)) if var in vcfg else "",
                days=[
                    ForecastDay(
                        target_date=d.target_date,
                        horizon=int(d.horizon),
                        **{q: nan_to_none(getattr(d, q)) for q in QUANTILES},
                        threshold=nan_to_none(d.threshold),
                        exceedance_prob=nan_to_none(d.exceedance_prob),
                        future_drivers_missing=nan_to_none(d.future_drivers_missing),
                    )
                    for d in g.sort_values("target_date").itertuples(index=False)
                ],
            )
        )
    return ForecastResponse(
        reach_id=reach_id,
        issued_date=issued,
        model_version=str(fc["model_version"].iloc[0]),
        variant=str(fc["variant"].iloc[0]),
        weather=str(fc["weather"].iloc[0]),
        horizon=horizon,
        observable=r["observable"],
        observability=observability(r["observable"]),
        notes=notes,
        series=series,
    )


@router.get("/reaches/{reach_id}/catchment", response_model=CatchmentFeature)
def reach_catchment(reach_id: str, repo: Repository = Depends(get_repository)) -> CatchmentFeature:
    c = repo.catchment(reach_id, CATCHMENT_SIMPLIFY_DEG)
    if c is None:
        raise HTTPException(404, f"unknown reach {reach_id!r}")
    return CatchmentFeature(
        id=f"{reach_id}-catchment",
        geometry=Geometry(**c["geometry"]) if c["geometry"] else None,
        properties={
            "reach_id": reach_id,
            "catchment_area_km2": nan_to_none(c["catchment_area_km2"]),
            "unit": "upstream contributing catchment",
            "display_simplified_deg": CATCHMENT_SIMPLIFY_DEG,
        },
    )


@router.get("/reaches/{reach_id}/attribution", response_model=AttributionResponse)
def reach_attribution(
    reach_id: str,
    repo: Repository = Depends(get_repository),
    shap_source: ShapSource = Depends(get_shap_source),
) -> AttributionResponse:
    """Driver attribution for the reach's current production forecast: the TreeSHAP rows
    models/baseline_gbm.py stored for it (nothing is recomputed), at the day the alert run
    would attribute - the peak P(exceed) - or, with no threshold, the peak P90."""
    r, city = _get_reach(repo, reach_id)
    fc = repo.latest_forecasts(city, FORECAST_VARIANT, reach_id)
    if fc.empty:
        raise HTTPException(
            404, f"no production forecast on record for {reach_id} - run `make evaluate`"
        )
    issued = fc["issued_date"].iloc[0]
    version = str(fc["model_version"].iloc[0])
    try:
        shap = shap_source(city, version, ATTRIBUTION_QUANTILE)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    if shap.empty:
        raise HTTPException(
            503,
            f"no {ATTRIBUTION_QUANTILE} SHAP rows stored for {version} - the forecast in the "
            "DB and the SHAP on disk are from different runs; re-run `make forecasts-latest`",
        )
    scored = with_exceedance(fc, threshold_table(repo, city, issued))
    series = []
    for var, g in scored.groupby("variable", sort=True):
        if g["exceedance_prob"].notna().any():
            row = g.loc[g["exceedance_prob"].idxmax()]
            selection = "PEAK_EXCEEDANCE"
        elif g["p90"].notna().any():
            row = g.loc[g["p90"].idxmax()]
            selection = "PEAK_P90"
        else:
            continue
        target = pd.to_datetime(row["target_date"]).date()
        contribs = attribution_for(shap, reach_id, str(var), target, ATTRIBUTION_TOP_K)
        series.append(
            AttributionSeries(
                variable=var,
                selection=selection,
                target_date=target,
                horizon=int(row["horizon"]),
                quantile=ATTRIBUTION_QUANTILE,
                contributions=[
                    DriverContributionOut(
                        feature=d.feature, contribution=d.contribution, value=d.value, kind=d.kind
                    )
                    for d in contribs
                ],
            )
        )
    return AttributionResponse(
        reach_id=reach_id,
        issued_date=issued,
        model_version=version,
        basis=(
            f"TreeSHAP of the {ATTRIBUTION_QUANTILE.upper()} LightGBM model, stored with the "
            "production forecast; contributions in the target's units, largest magnitude "
            "first; reach identity and lead time are not drivers and are left out"
        ),
        series=series,
    )
