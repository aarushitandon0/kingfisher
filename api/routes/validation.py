"""GET /api/validation/metrics - every MASTERSPEC 11 metric, served LIVE.

The file models/evaluate.py wrote (results/metrics.json; results/<city>/metrics.json for a
city other than the primary one) is read on every request and
returned with its sha256 and modification time - not a static image, not a cached copy -
so what the UI shows is exactly what the evaluation produced, losses included. The
scenario response check (MASTERSPEC 9.6) is served next to it when it exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_results_dir
from api.schemas import MetricsDocument, SourceFile, ValidationMetricsResponse
from api.services import city_configs, file_meta
from core.settings import PRIMARY_CITY, results_dir_for

router = APIRouter(tags=["validation"])


@router.get("/validation/metrics", response_model=ValidationMetricsResponse)
def validation_metrics(
    city: str | None = Query(None, description="city id; omit for the primary city"),
    results_dir: Path = Depends(get_results_dir),
) -> ValidationMetricsResponse:
    if city is not None and city not in city_configs():
        raise HTTPException(404, f"unknown city {city!r}")
    path = results_dir_for(city or PRIMARY_CITY, results_dir) / "metrics.json"
    if not path.exists():
        raise HTTPException(
            503,
            f"{path.relative_to(results_dir.parent).as_posix()} missing - run "
            f"`make evaluate city={city or PRIMARY_CITY}`",
        )
    try:
        metrics = MetricsDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            500, f"results/metrics.json is not a valid metrics document: {exc}"
        ) from exc
    if city is not None and metrics.city != city:
        raise HTTPException(500, f"{path.name} is for {metrics.city!r}, not {city!r}")
    check_path = results_dir / f"scenario_response_check_{metrics.city}.json"
    check_doc = check_src = None
    if check_path.exists():
        check_doc = json.loads(check_path.read_text(encoding="utf-8"))
        check_src = SourceFile(**file_meta(check_path))
    return ValidationMetricsResponse(
        served_at=datetime.now(UTC),
        source=SourceFile(**file_meta(path)),
        metrics=metrics,
        scenario_response_check=check_doc,
        scenario_response_check_source=check_src,
    )
