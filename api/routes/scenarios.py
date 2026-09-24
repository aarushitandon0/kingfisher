"""Scenario routes (MASTERSPEC 9, 12):

GET  /api/scenarios/interventions?city=   the cited lever catalogue + each lever's path
POST /api/scenarios        {name?, variables?, interventions: [{type, reach_ids, extent}]}
GET  /api/scenarios/{id}   per-reach deltas + intervals + citations + path per lever

Every result is a planning estimate from engine.scenarios: effect sizes from the cited
coefficient table only, variant B only, response-checked levers, widened intervals.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import ScenarioRunnerFactory, get_repository, get_scenario_runner
from api.repository import Repository
from api.schemas import (
    InterventionCatalogue,
    InterventionOut,
    LeverPathOut,
    NotQuantifiedOut,
    ScenarioRequest,
    ScenarioResponse,
)
from api.services import city_configs, city_of_reach
from core.config import load_intervention_coefficients
from engine.coefficients import UncitedCoefficientError
from engine.scenarios import MissingResponseCheck, ScenarioResult, StaleResponseCheck

router = APIRouter(tags=["scenarios"])

MAX_REACHES = 100


# How `extent` is read per operation (config/intervention_coefficients.yaml, OPERATIONS;
# bounds enforced by engine.scenarios.validate_extent).
EXTENT: dict[str, tuple[str | None, float | None, float | None]] = {
    "subtract_treated_share": ("percentage points of catchment area treated", 0.0, 100.0),
    "scale_down": ("fraction of the catchment the programme covers", 0.0, 1.0),
    "floor": (None, None, None),
}


@router.get("/scenarios/interventions", response_model=InterventionCatalogue)
def interventions(
    city: str = Query(..., description="city id, e.g. coimbra"),
    runner: ScenarioRunnerFactory = Depends(get_scenario_runner),
) -> InterventionCatalogue:
    """Every cited lever, the citations for each number, and what a run in this city would
    do with it per variable. Read from the validated coefficient table; nothing here is
    estimated."""
    if city not in city_configs():
        raise HTTPException(404, f"unknown city {city!r}")
    try:
        table = load_intervention_coefficients()
    except UncitedCoefficientError as exc:
        raise HTTPException(503, str(exc)) from exc
    try:
        paths = runner(city).lever_paths()
    except (FileNotFoundError, MissingResponseCheck, StaleResponseCheck) as exc:
        raise HTTPException(503, f"scenario engine unavailable for {city}: {exc}") from exc
    out = []
    for c in table.interventions:
        unit, lo, hi = EXTENT[c.effect.operation]
        out.append(
            InterventionOut(
                id=c.id,
                name=c.name,
                feature=c.effect.feature,
                operation=c.effect.operation,
                direction=c.effect.direction,
                effect_unit=c.effect.unit,
                description=c.effect.description.strip(),
                extent_unit=unit,
                extent_min=lo,
                extent_max=hi,
                magnitude=c.magnitude,
                uncertainty_range=c.uncertainty_range,
                applies_to=c.applies_to.model_dump(mode="json"),
                cost_per_unit=c.cost_per_unit.model_dump(mode="json"),
                citations=[r.model_dump(mode="json") for r in c.citation],
                direct_effect=c.direct_effect.model_dump(mode="json") if c.direct_effect else None,
                paths={
                    v: LeverPathOut(path=p, reason=why)  # type: ignore[arg-type]
                    for v, (p, why) in paths.get(c.id, {}).items()
                },
            )
        )
    return InterventionCatalogue(
        city=city,
        coefficient_table_version=table.version,
        caveat=str(table.uncertainty_inflation["caveat"]),
        interventions=out,
        not_quantified=[
            NotQuantifiedOut(
                id=n.id,
                name=n.name,
                reason=n.reason.strip(),
                citations=[r.model_dump(mode="json") for r in n.citation],
            )
            for n in table.not_quantified
        ],
    )


@router.post("/scenarios", response_model=ScenarioResponse, status_code=status.HTTP_201_CREATED)
def create_scenario(
    body: ScenarioRequest,
    repo: Repository = Depends(get_repository),
    runner: ScenarioRunnerFactory = Depends(get_scenario_runner),
) -> ScenarioResponse:
    rids = sorted({r for i in body.interventions for r in i.reach_ids})
    if len(rids) > MAX_REACHES:
        raise HTTPException(422, f"{len(rids)} reaches in one scenario (max {MAX_REACHES})")
    cities = {city_of_reach(r) for r in rids}
    if None in cities:
        bad = [r for r in rids if city_of_reach(r) is None]
        raise HTTPException(404, f"no city has reach(es) {bad}")
    if len(cities) != 1:
        raise HTTPException(422, f"a scenario covers one city; got {sorted(map(str, cities))}")
    city = str(cities.pop())
    try:
        ctx = runner(city)
    except (FileNotFoundError, MissingResponseCheck, StaleResponseCheck) as exc:
        raise HTTPException(503, f"scenario engine unavailable for {city}: {exc}") from exc
    except UncitedCoefficientError as exc:
        raise HTTPException(503, str(exc)) from exc
    unknown = sorted(set(rids) - ctx.reach_ids)
    if unknown:
        raise HTTPException(404, f"unknown reach(es) {unknown}")
    try:
        result = ctx.run(body.interventions, body.variables)
    except (MissingResponseCheck, StaleResponseCheck) as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    config = body.model_dump(mode="json")
    sid, created = repo.save_scenario(body.name, city, config, result)
    return ScenarioResponse(
        scenario_id=sid, name=body.name, city=city, created_at=created, result=result
    )


@router.get("/scenarios/{scenario_id}", response_model=ScenarioResponse)
def get_scenario(scenario_id: UUID, repo: Repository = Depends(get_repository)) -> ScenarioResponse:
    s = repo.scenario(scenario_id)
    if s is None:
        raise HTTPException(404, f"unknown scenario {scenario_id}")
    if not s.get("result"):
        raise HTTPException(500, f"scenario {scenario_id} has no stored result")
    # Re-validated on the way out: a stored result that no longer satisfies the contract
    # (citations, strictly wider intervals) is an error, not a response.
    return ScenarioResponse(
        scenario_id=s["scenario_id"],
        name=s["name"],
        city=s["city"],
        created_at=s["created_at"],
        result=ScenarioResult.model_validate(s["result"]),
    )
