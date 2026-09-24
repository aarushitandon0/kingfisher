"""Glue between the repository and engine/ for the routes. Every number the API serves that
is not a stored value is computed here by deterministic engine code: thresholds by
engine.thresholds, exceedance probabilities by engine.probability (the same CDF the alerts
and the reliability diagram use), rankings by engine.priorities.
"""

from __future__ import annotations

import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api.repository import Repository
from core.config import load_config
from core.settings import CONFIG_DIR
from engine.probability import exceedance_probability_multi
from engine.thresholds import derivation_text, seasonal_thresholds
from engine.thresholds import lookup as threshold_lookup

QUANTILES = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
FORECAST_VARIANT = "A"  # the forecasting model; variant B is for scenarios only (9.6)


def thresholds_config() -> dict[str, Any]:
    return load_config("thresholds")


def threshold_derivation(variable: str | None = None) -> str:
    cfg = thresholds_config()
    if variable is not None:
        return derivation_text(cfg, variable)
    return "; ".join(f"{k}: {derivation_text(cfg, k)}" for k in cfg["variables"])


@lru_cache(maxsize=1)
def city_configs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((CONFIG_DIR / "cities").glob("*.yaml")):
        cfg = load_config(f"cities/{path.stem}")
        out[str(cfg["city"])] = cfg
    return out


def city_of_reach(reach_id: str) -> str | None:
    prefix = reach_id.split("-")[0]
    for city, cfg in city_configs().items():
        if cfg.get("reach_id_prefix") == prefix:
            return city
    return None


def threshold_table(repo: Repository, city: str, fit_end: date) -> pd.DataFrame:
    """Per-reach seasonal thresholds from OK observations dated <= fit_end - the issue
    date of the forecast they judge, so a threshold never sees the period it judges."""
    return seasonal_thresholds(repo.ok_observations(city), fit_end, thresholds_config())


def with_exceedance(forecasts: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add threshold and exceedance_prob to forecast rows. NaN where the reach-season has
    no threshold or a quantile is missing - never 0."""
    if forecasts.empty:
        return forecasts.assign(
            threshold=pd.Series(dtype=float), exceedance_prob=pd.Series(dtype=float)
        )
    seasons = thresholds_config()["seasons"]
    thr = threshold_lookup(forecasts[["reach_id", "variable", "target_date"]], table, seasons)[
        "threshold"
    ].to_numpy(dtype="float64")
    q = forecasts[list(QUANTILES)].to_numpy(dtype="float64")
    p = exceedance_probability_multi(q, LEVELS, thr)
    return forecasts.assign(threshold=thr, exceedance_prob=p)


def shap_path(city: str) -> Path:
    from core.settings import DATA_DIR

    return DATA_DIR / "processed" / f"gbm_shap_{city}.parquet"


@lru_cache(maxsize=4)
def _production_shap(path: Path, mtime: float, model_version: str, quantile: str) -> pd.DataFrame:
    shap = pd.read_parquet(path)
    return shap[
        (shap["fold"] == "production")
        & (shap["model_version"] == model_version)
        & (shap["quantile"] == quantile)
    ].reset_index(drop=True)


def production_shap(city: str, model_version: str, quantile: str) -> pd.DataFrame:
    """TreeSHAP rows written by models/baseline_gbm.py for the production forecast of
    `model_version`. Stored values, not recomputed. Empty if that model has none."""
    path = shap_path(city)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing - run `make train city={city}`")
    return _production_shap(path, path.stat().st_mtime, model_version, quantile)


def nan_to_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float | np.floating) and math.isnan(float(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def file_meta(path: Path) -> dict[str, Any]:
    import hashlib
    from datetime import UTC, datetime

    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, UTC),
        "size_bytes": len(data),
    }
