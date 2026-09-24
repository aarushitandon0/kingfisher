"""GET /api/priorities?city= - ranked next-field-visit list (MASTERSPEC 10)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_repository
from api.repository import Repository
from api.schemas import PrioritiesResponse, RankedReachOut, observability
from api.services import FORECAST_VARIANT, city_configs, threshold_table, with_exceedance
from core.config import load_config
from engine.priorities import rank_reaches

router = APIRouter(tags=["priorities"])


@router.get("/priorities", response_model=PrioritiesResponse)
def priorities(
    city: str = Query(..., description="city id"),
    repo: Repository = Depends(get_repository),
) -> PrioritiesResponse:
    if city not in city_configs():
        raise HTTPException(404, f"unknown city {city!r}")
    reaches = {r["reach_id"]: r for r in repo.reaches(city)}
    if not reaches:
        raise HTTPException(404, f"no reaches for {city}")
    fc = repo.latest_forecasts(city, FORECAST_VARIANT)
    if fc.empty:
        raise HTTPException(404, f"no production forecast on record for {city}")
    issued = fc["issued_date"].iloc[0]
    scored = with_exceedance(fc, threshold_table(repo, city, issued))
    cfg = load_config("priorities")
    result = rank_reaches(
        sorted(reaches),
        scored[["reach_id", "variable", "p10", "p90", "exceedance_prob"]],
        repo.exposure(sorted(reaches)),
        cfg["exposure_weights"],
        float(cfg["population_unit"]),
    )
    return PrioritiesResponse(
        city=city,
        issued_date=issued,
        model_version=str(fc["model_version"].iloc[0]),
        method=result.method,
        weights=result.weights,
        population_unit=result.population_unit,
        ranked=[
            RankedReachOut(
                **r.model_dump(),
                name=reaches[r.reach_id]["name"],
                observability=observability(reaches[r.reach_id]["observable"]),
            )
            for r in result.ranked
        ],
        not_ranked=list(result.not_ranked),
    )
