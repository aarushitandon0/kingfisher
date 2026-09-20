"""FastAPI application entry point (MASTERSPEC section 12).

Routes are registered from `api/routes/`; none are implemented yet. `/health` is
here so `docker compose up` plus `uvicorn api.main:app` can be smoke-tested on Day 0.
"""

from __future__ import annotations

from fastapi import FastAPI

from core.logging import get_logger
from core.settings import get_settings

log = get_logger(__name__)

app = FastAPI(
    title="Kingfisher",
    version="0.1.0",
    description=(
        "Early-warning and resilience planning for urban streams. "
        "Kingfisher maps exposure pathways; it does not predict health outcomes "
        "and does not declare water safe or unsafe."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": get_settings().env, "version": app.version}
