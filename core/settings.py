"""Environment-backed settings. The single place `.env` is read."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
# The primary study city. Its evaluation documents live directly under results/ (the
# README, the tests and the EA-LSTM gate refer to them by those paths); every other city
# writes to results/<city>/ so a transfer run can never overwrite them.
PRIMARY_CITY = "coimbra"


def results_dir_for(city: str, root: Path = RESULTS_DIR) -> Path:
    return root if city == PRIMARY_CITY else root / city


class Settings(BaseSettings):
    """Runtime configuration. Missing secrets fail loudly at first use, not silently."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sentinel Hub (Statistical API)
    sh_client_id: str = Field(default="", alias="SH_CLIENT_ID")
    sh_client_secret: str = Field(default="", alias="SH_CLIENT_SECRET")

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://kingfisher:kingfisher@localhost:5432/kingfisher",
        alias="DATABASE_URL",
    )

    # Climate Data Store (optional)
    cds_api_key: str = Field(default="", alias="CDS_API_KEY")

    # Runtime
    env: str = Field(default="dev", alias="KINGFISHER_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="console", alias="LOG_FORMAT")
    cache_dir: Path = Field(default=DATA_DIR / "raw" / "_cache", alias="CACHE_DIR")

    # API: browser origins allowed by CORS (comma-separated). Default: the Vite dev server.
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173", alias="CORS_ORIGINS"
    )

    # API: build each city's scenario model in a background thread at startup, so the
    # first POST /api/scenarios does not wait ~14 s for it. Off: built on first use.
    scenario_warmup: bool = Field(default=True, alias="SCENARIO_WARMUP")
    # Built frontend (frontend/dist) served by the API in the deployed image; unset locally.
    static_dir: Path | None = Field(default=None, alias="STATIC_DIR")

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def require_sentinel_hub(self) -> tuple[str, str]:
        """Return SH credentials or raise. Never return a blank credential pair."""
        if not self.sh_client_id or not self.sh_client_secret:
            raise RuntimeError(
                "SH_CLIENT_ID / SH_CLIENT_SECRET are unset. Copy .env.example to .env "
                "and fill in the CDSE OAuth client credentials."
            )
        return self.sh_client_id, self.sh_client_secret


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
