"""GET /api/cities - every configured city and how much of it is built."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_repository
from api.repository import Repository
from api.schemas import BBox, CitiesResponse, CityOut
from api.services import city_configs

router = APIRouter(tags=["cities"])


@router.get("/cities", response_model=CitiesResponse)
def list_cities(repo: Repository = Depends(get_repository)) -> CitiesResponse:
    counts = repo.reach_counts()
    out = []
    for city, cfg in city_configs().items():
        c = counts.get(city, {})
        n = int(c.get("reaches", 0))
        out.append(
            CityOut(
                city=city,
                name=str(cfg.get("name", city)),
                country=str(cfg.get("country", "")),
                reach_id_prefix=str(cfg["reach_id_prefix"]),
                bbox=BBox(**cfg["bbox"]),
                timezone=cfg.get("timezone"),
                status="READY" if n else "NOT_BUILT",
                reaches=n,
                optically_observable=int(c.get("optically_observable", 0)),
                driver_only=int(c.get("driver_only", 0)),
                not_assessed=int(c.get("not_assessed", 0)),
            )
        )
    return CitiesResponse(cities=out)
