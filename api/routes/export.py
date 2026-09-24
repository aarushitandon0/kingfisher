"""GET /api/export/fhir/{alert_id} - FHIR R4 Observation + Location + Device bundle."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from api.deps import get_repository
from api.fhir import alert_to_fhir_bundle
from api.repository import Repository
from api.services import thresholds_config

router = APIRouter(tags=["export"])


class FhirBundle(BaseModel):
    """The FHIR Bundle envelope. Resource bodies pass through as FHIR JSON."""

    model_config = ConfigDict(extra="forbid")
    resourceType: Literal["Bundle"]
    id: str
    type: Literal["collection"]
    timestamp: str
    entry: list[dict[str, Any]]


@router.get("/export/fhir/{alert_id}", response_model=FhirBundle)
def export_fhir(alert_id: UUID, repo: Repository = Depends(get_repository)) -> FhirBundle:
    a = repo.alert(alert_id)
    if a is None:
        raise HTTPException(404, f"unknown alert {alert_id}")
    name = thresholds_config()["variables"].get(a["variable"], {}).get("display_name")
    return FhirBundle(**alert_to_fhir_bundle(a, display_name=name))
