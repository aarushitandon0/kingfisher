"""GET /api/timeline?city= - everything the hydrograph rail needs to re-render the whole map
at any date, in one columnar payload.

  observations  every OK Sentinel-2 value of the city (flagged dates are absent, so a gap
                stays a gap - nothing is filled)
  forecast      the production forecast of the latest issue date, P10/P50/P90 with the
                reach's seasonal threshold and the daily P(exceed) from engine.probability
  thresholds    the per-reach seasonal thresholds (engine.thresholds) as of that issue date

Every number is a stored value or computed by engine/ (via api.services). Columnar rather
than one object per row: ~15k observations and ~7k forecast rows for Coimbra.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_repository
from api.repository import Repository
from api.schemas import TimelineForecast, TimelineObservations, TimelineResponse, TimelineThresholds
from api.services import (
    FORECAST_VARIANT,
    city_configs,
    nan_to_none,
    threshold_table,
    thresholds_config,
    with_exceedance,
)

router = APIRouter(tags=["timeline"])


def _col(s: pd.Series) -> list[float | None]:
    return [nan_to_none(float(v)) for v in s.to_numpy(dtype="float64")]


def _dates(s: pd.Series) -> list[date]:
    return [d.date() for d in pd.to_datetime(s)]


@router.get("/timeline", response_model=TimelineResponse)
def timeline(
    city: str = Query(..., description="city id, e.g. coimbra"),
    repo: Repository = Depends(get_repository),
) -> TimelineResponse:
    if city not in city_configs():
        raise HTTPException(404, f"unknown city {city!r}")
    obs = repo.ok_observations(city).sort_values(["date", "reach_id", "variable"])
    fc = repo.latest_forecasts(city, FORECAST_VARIANT)
    issued = fc["issued_date"].iloc[0] if not fc.empty else None
    fit_end = issued or (pd.to_datetime(obs["date"]).max().date() if not obs.empty else None)
    table = (
        threshold_table(repo, city, fit_end)
        if fit_end
        else pd.DataFrame(columns=["reach_id", "variable", "season", "threshold"])
    )
    scored = (
        with_exceedance(fc, table).sort_values(["reach_id", "variable", "target_date"])
        if issued
        else fc
    )
    return TimelineResponse(
        city=city,
        issued_date=issued,
        model_version=str(fc["model_version"].iloc[0]) if not fc.empty else None,
        seasons={k: [int(m) for m in v] for k, v in thresholds_config()["seasons"].items()},
        threshold_note=(
            "Seasonal thresholds fitted on observations up to the forecast issue date. The rail "
            "colours past observations against today's thresholds - a display scale, not a "
            "re-scored hindcast (the walk-forward metrics are on the validation page)."
        ),
        observations=TimelineObservations(
            reach_id=[str(r) for r in obs["reach_id"]],
            date=_dates(obs["date"]),
            variable=[str(v) for v in obs["variable"]],  # type: ignore[misc]
            value=_col(obs["value"]),
        ),
        forecast=TimelineForecast(
            reach_id=[str(r) for r in scored["reach_id"]] if issued else [],
            target_date=_dates(scored["target_date"]) if issued else [],
            variable=[str(v) for v in scored["variable"]] if issued else [],  # type: ignore[misc]
            p10=_col(scored["p10"]) if issued else [],
            p50=_col(scored["p50"]) if issued else [],
            p90=_col(scored["p90"]) if issued else [],
            threshold=_col(scored["threshold"]) if issued else [],
            exceedance_prob=_col(scored["exceedance_prob"]) if issued else [],
        ),
        thresholds=TimelineThresholds(
            reach_id=[str(r) for r in table["reach_id"]],
            variable=[str(v) for v in table["variable"]],  # type: ignore[misc]
            season=[str(s) for s in table["season"]],
            threshold=_col(table["threshold"]),
        ),
    )
