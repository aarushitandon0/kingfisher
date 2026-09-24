"""FastAPI application entry point (MASTERSPEC section 12).

    uvicorn api.main:app --reload            (make api)

Routes live in api/routes/, one module per resource, all under /api. Schemas are in
api/schemas.py; data access is behind api.repository.Repository; every computed number
comes from engine/. CORS is open to the local Vite dev server (CORS_ORIGINS in .env).
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routes import (
    alerts,
    cities,
    export,
    exposure,
    priorities,
    reaches,
    scenarios,
    timeline,
    validation,
)
from api.services import city_configs
from core.logging import get_logger
from core.settings import REPO_ROOT, get_settings

log = get_logger(__name__)


def _warm_scenarios() -> None:
    """Build the scenario model of every city that has a response check on disk. A city
    that fails here is logged and left to fail loudly on its first POST instead."""
    from models.scenario_model import check_path, scenario_context

    for city in city_configs():
        if not check_path(city, REPO_ROOT / "results").exists():
            continue
        t0 = time.monotonic()
        try:
            scenario_context(city)
        except Exception as exc:  # noqa: BLE001 - the request path reports it properly
            log.warning("api.scenario_warmup_failed", city=city, error=str(exc)[:300])
            continue
        log.info("api.scenario_warmup", city=city, seconds=round(time.monotonic() - t0, 1))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if get_settings().scenario_warmup:
        threading.Thread(target=_warm_scenarios, name="scenario-warmup", daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Kingfisher",
    version="0.2.0",
    description=(
        "Early-warning and resilience planning for urban streams. "
        "Kingfisher maps exposure pathways; it does not predict health outcomes "
        "and does not declare water safe or unsafe. Scenario outputs are planning "
        "estimates, not causal claims."
    ),
)

app.add_middleware(GZipMiddleware, minimum_size=4096)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

for module in (
    cities,
    reaches,
    alerts,
    scenarios,
    priorities,
    validation,
    export,
    exposure,
    timeline,
):
    app.include_router(module.router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": get_settings().env, "version": app.version}
