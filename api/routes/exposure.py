"""GET /api/exposure?city= - the exposure features the map can toggle on.

Exposure pathways and proximity only (CLAUDE.md #8): the stored nearest feature of each type
within each reach's buffer (pipeline/exposure_features.py), deduplicated.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_repository
from api.repository import Repository
from api.schemas import ExposureLayer, ExposureLayerFeature, ExposureLayerProperties, Geometry
from api.services import city_configs

router = APIRouter(tags=["exposure"])


@router.get("/exposure", response_model=ExposureLayer)
def exposure_layer(
    city: str = Query(..., description="city id, e.g. coimbra"),
    repo: Repository = Depends(get_repository),
) -> ExposureLayer:
    if city not in city_configs():
        raise HTTPException(404, f"unknown city {city!r}")
    rows = repo.exposure_layer(city)
    return ExposureLayer(
        city=city,
        features=[
            ExposureLayerFeature(
                geometry=Geometry(**r["geometry"]),
                properties=ExposureLayerProperties(
                    feature_type=r["feature_type"],
                    reaches=len(r["reach_ids"]),
                    reach_ids=",".join(r["reach_ids"]),
                ),
            )
            for r in rows
        ],
    )
