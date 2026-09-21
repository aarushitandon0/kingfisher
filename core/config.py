"""YAML config loading. Cities have their own loader (pipeline.l0_network.load_city_config)
because they carry required keys; everything else in config/ comes through here."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any

import yaml

from core.settings import CONFIG_DIR


@lru_cache(maxsize=16)
def load_config(name: str) -> dict[str, Any]:
    """config/<name>.yaml as a dict. Raises if the file is missing - no silent defaults."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file {path}")
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg


def as_date(value: Any) -> date:
    """YAML parses 2023-12-31 as a date already; strings are accepted too."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
