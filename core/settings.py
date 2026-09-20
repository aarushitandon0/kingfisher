"""Environment-backed settings. The single place `.env` is read."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"


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
