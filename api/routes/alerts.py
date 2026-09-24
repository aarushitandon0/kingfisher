"""Alert routes (MASTERSPEC 12):

GET /api/alerts?city=&active=true   alerts incl. INSUFFICIENT_EVIDENCE, severity-sorted
GET /api/alerts/{id}                attribution + exposure + basis + guardrail status

`active=true` means the alerts of the latest alert run (the most recent issue date). With
no alert run on record the response says NO_ALERT_RUN - an empty list is never passed off
as "no alerts".
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_repository
from api.repository import Repository
from api.schemas import AlertDetail, AlertsResponse, AlertSummary, DriverContributionOut
from api.services import city_configs

router = APIRouter(tags=["alerts"])

SEVERITY_ORDER = {"ALERT": 0, "WATCH": 1, "INSUFFICIENT_EVIDENCE": 2}
SUMMARY_FIELDS = tuple(AlertSummary.model_fields)


def _summary(a: dict[str, Any]) -> AlertSummary:
    return AlertSummary(**{k: a.get(k) for k in SUMMARY_FIELDS})


@router.get("/alerts", response_model=AlertsResponse)
def list_alerts(
    city: str | None = Query(None, description="city id; omit for every city"),
    active: bool = Query(True, description="only the latest alert run"),
    repo: Repository = Depends(get_repository),
) -> AlertsResponse:
    if city is not None and city not in city_configs():
        raise HTTPException(404, f"unknown city {city!r}")
    run = repo.alert_run_date(city)
    rows = [] if run is None else repo.alerts(city, run if active else None)
    alerts = sorted(
        (_summary(a) for a in rows),
        key=lambda a: (
            SEVERITY_ORDER[a.severity],
            -(a.exceedance_prob if a.exceedance_prob is not None else -1.0),
            a.reach_id,
            a.variable,
        ),
    )
    counts: dict[str, int] = {s: 0 for s in SEVERITY_ORDER}
    for a in alerts:
        counts[a.severity] += 1
    return AlertsResponse(
        city=city,
        active=active,
        alert_run="OK" if run else "NO_ALERT_RUN",
        alert_run_issued_date=run,
        counts=counts,  # type: ignore[arg-type]
        suppressed=repo.alert_run_suppressed(city, run) if run else None,
        alerts=alerts,
    )


@router.get("/alerts/{alert_id}", response_model=AlertDetail)
def alert_detail(alert_id: UUID, repo: Repository = Depends(get_repository)) -> AlertDetail:
    a = repo.alert(alert_id)
    if a is None:
        raise HTTPException(404, f"unknown alert {alert_id}")
    return AlertDetail(
        **{k: a.get(k) for k in SUMMARY_FIELDS if k != "exposure"},
        threshold_derivation=a.get("threshold_derivation"),
        attribution=[DriverContributionOut(**d) for d in (a.get("attribution") or [])],
        exposure=a.get("exposure"),
        basis=a.get("basis"),
        guardrails=a.get("guardrails"),
        withheld_exceedance_prob=a.get("withheld_exceedance_prob"),
    )
