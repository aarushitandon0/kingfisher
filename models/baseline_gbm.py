"""LightGBM quantile baseline (MASTERSPEC 7.1). Build this first - it is the system that
ships if the hybrid does not beat it.

Run:

    python -m models.baseline_gbm --city coimbra

MODELS
------
One LightGBM booster per (variable, horizon bucket, quantile): variables turbidity_proxy
and ndci, buckets 1-3 d / 4-7 d / 8-10 d, alpha 0.1 / 0.5 / 0.9 - 18 boosters per fit,
all in config/modelling.yaml (baseline_gbm). Each is pooled across every reach, with
reach_id as a LightGBM categorical.

Within a bucket the frame is stacked to one row per (issue date t, horizon h) that has an
OK observation at t+h. Features are what is known at t (drivers, static and reach
attributes, the labelled as-of observations and their age), the horizon itself, and the
drivers at t+h (`<col>_fut`, plus `precip_fut_cum_mm` = rain over t+1..t+h). In training
the future drivers are archive weather (perfect prognosis); live they are the Open-Meteo
forecast. Missing features stay NaN - LightGBM routes them, nothing is filled.

Quantile models are fitted independently, so their predictions can cross. They are
rearranged (sorted per row; Chernozhukov, Fernandez-Val & Galichon 2010) and the number
of crossed rows is recorded, not hidden.

FITS
----
  wf-val       train <= 2023, predicts 2024       } the walk-forward folds from
  wf-test      train <= 2024, predicts 2025-2026  } pipeline.build_dataset.walk_forward
  production   every row in the frame             - what predict() serves

Hyperparameters are fixed in config, not tuned on val or test, and there is no early
stopping (it would leak the evaluation fold into the model).

SHAP
----
LightGBM's `pred_contrib=True` is TreeSHAP (Lundberg et al. 2020) - the same exact
values as shap.TreeExplainer, computed inside LightGBM. Persisted for the P50 and P90
models on every evaluation forecast and on each reach's latest forecast, as
`shap__<feature>` next to `value__<feature>`, plus `shap_bias`. Contributions are in the
target's units and sum, with the bias, to the raw model output. `shap__reach_id` is the
reach's own offset, not a driver - the alert engine must not present it as one.

OUTPUTS
-------
  artifacts/models/<version>/            boosters (LightGBM text) + manifest.json
  artifacts/models/<city>__<fit>.json    pointer to the current version of that fit
  data/processed/gbm_eval_<city>.parquet     walk-forward forecasts with observations
  data/processed/gbm_latest_<city>.parquet   production forecast at the last issue date
  data/processed/gbm_shap_<city>.parquet     SHAP rows for both of the above
  data/processed/gbm_run_<city>.json         run manifest (versions, frame hash)

Version strings are gbm-<semver>+<city>.<fit>.<fingerprint>, where the fingerprint
hashes the config, the feature list and the training data: retraining on unchanged
inputs reproduces the same version, and any change produces a new one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.config import as_date, load_config
from core.logging import get_logger, stage
from core.settings import REPO_ROOT
from pipeline.build_dataset import (
    DRIVER_FEATURES,
    PROCESSED_DIR,
    REACH_FEATURES,
    STATIC_FEATURES,
    FrameSettings,
    embargo,
    future_column,
    target_column,
    walk_forward,
)

log = get_logger(__name__)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "models"
KEY_COLUMNS = ["reach_id", "issued_date", "horizon", "target_date"]
PRODUCTION = "production"
SERVING_FIT = f"A.{PRODUCTION}"  # predict()/explain() default; scenarios use B.production


class InsufficientTrainingData(RuntimeError):
    """A model would be fitted on too few target rows to mean anything."""


def qname(alpha: float) -> str:
    """0.1 -> 'p10'."""
    return f"p{round(alpha * 100):02d}"


@dataclass(frozen=True)
class VariantSpec:
    """A = reach_id as a feature; B = no reach identity (config baseline_gbm.variants)."""

    name: str
    reach_id: bool = True
    drop: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "reach_id": self.reach_id, "drop": list(self.drop)}


DEFAULT_VARIANT = VariantSpec("A")


@dataclass(frozen=True)
class GBMSettings:
    version: str
    quantiles: tuple[float, ...]
    buckets: dict[str, tuple[int, ...]]
    min_train_rows: int
    params: dict[str, Any]
    shap_quantiles: tuple[float, ...]
    variants: dict[str, VariantSpec] = field(default_factory=lambda: {"A": DEFAULT_VARIANT})
    drop_when_proxy: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> GBMSettings:
        cfg = cfg or load_config("modelling")
        g = cfg["baseline_gbm"]
        variants = {
            name: VariantSpec(name, bool(v["reach_id"]), tuple(v.get("drop") or ()))
            for name, v in (g.get("variants") or {"A": {"reach_id": True}}).items()
        }
        return cls(
            version=str(g["version"]),
            quantiles=tuple(float(a) for a in g["quantiles"]),
            buckets={k: tuple(int(h) for h in v) for k, v in g["horizon_buckets"].items()},
            min_train_rows=int(g["min_train_rows"]),
            params=dict(g["params"]),
            shap_quantiles=tuple(float(a) for a in g["shap_quantiles"]),
            variants=variants,
            drop_when_proxy=tuple(g.get("drop_when_proxy") or ()),
        )

    def bucket_of(self, horizon: int) -> str:
        for name, hs in self.buckets.items():
            if horizon in hs:
                return name
        raise ValueError(f"horizon {horizon} is in no bucket {self.buckets}")


def validate_settings(gs: GBMSettings, fs: FrameSettings) -> None:
    covered = sorted(h for hs in gs.buckets.values() for h in hs)
    if covered != sorted(fs.horizons):
        raise ValueError(f"horizon buckets cover {covered}, frame horizons are {fs.horizons}")
    if sorted(gs.quantiles) != list(gs.quantiles) or len(set(gs.quantiles)) != len(gs.quantiles):
        raise ValueError(f"quantiles must be strictly increasing: {gs.quantiles}")
    if not set(gs.shap_quantiles) <= set(gs.quantiles):
        raise ValueError("shap_quantiles must be a subset of quantiles")


# ---------------------------------------------------------------------------
# features (pure)
# ---------------------------------------------------------------------------
def base_features(fs: FrameSettings) -> list[str]:
    """Everything known on the issue date."""
    asof = [f"{v}_asof" for v in fs.variables] + ["obs_asof_age_days"]
    return [*DRIVER_FEATURES, *STATIC_FEATURES, *REACH_FEATURES, *asof]


def future_features(fs: FrameSettings) -> list[str]:
    out = [f"{c}_fut" for c in fs.future_driver_columns]
    if "precip_mm" in fs.future_driver_columns:
        out.append("precip_fut_cum_mm")
    return out


def all_features(fs: FrameSettings) -> list[str]:
    """Every column to_long builds, whatever the variant."""
    return ["reach_id", "horizon", *base_features(fs), *future_features(fs)]


def feature_names(
    fs: FrameSettings, variant: VariantSpec | None = None, dropped: Iterable[str] = ()
) -> list[str]:
    """The model's features: all_features minus the variant's exclusions and any
    feature dropped by policy (e.g. a static attribute that is a land-cover proxy)."""
    v = variant or DEFAULT_VARIANT
    out = set(v.drop) | set(dropped)
    if not v.reach_id:
        out.add("reach_id")
    unknown = out - set(all_features(fs))
    if unknown:
        raise ValueError(f"cannot drop unknown features {sorted(unknown)}")
    return [f for f in all_features(fs) if f not in out]


def proxy_dropped_features(
    gs: GBMSettings, static_flags: dict[str, dict[str, int]]
) -> dict[str, str]:
    """{feature: why} for every drop_when_proxy feature whose value is currently a
    land-cover proxy on any reach (a catchment_attributes flag starting PROXY_)."""
    out = {}
    for f in gs.drop_when_proxy:
        proxies = {
            flag: n for flag, n in static_flags.get(f, {}).items() if flag.startswith("PROXY_")
        }
        if proxies:
            out[f] = ", ".join(f"{flag} on {n} reaches" for flag, n in sorted(proxies.items()))
    return out


def _numeric(col: pd.Series, name: str) -> pd.Series:
    """Float column. Booleans -> 0/1 with NA kept NA. A non-null value that is not a
    number raises - coercing it to NaN would be a silent imputation."""
    if pd.api.types.is_bool_dtype(col) or str(col.dtype) == "boolean":
        return col.astype("Float64").astype("float64")
    out = pd.to_numeric(col, errors="coerce")
    bad = out.isna() & col.notna()
    if bad.any():
        raise TypeError(
            f"feature {name}: {int(bad.sum())} non-numeric values, e.g. {col[bad].iloc[0]!r}"
        )
    return out.astype("float64")


def to_long(
    frame: pd.DataFrame,
    fs: FrameSettings,
    variable: str,
    horizons: Iterable[int],
    categories: list[str],
    *,
    require_target: bool = True,
) -> pd.DataFrame:
    """Stack issue-date rows into one row per (issue date, horizon).

    Columns: KEY_COLUMNS, `target` (NaN when unobserved), `reach_observable`, then
    feature_names(fs). With require_target, rows whose target is NaN are dropped: an
    unobserved reach-date is not a training example and is never a zero.
    """
    if variable not in fs.variables:
        raise ValueError(f"unknown variable {variable!r}; frame has {fs.variables}")
    base = base_features(fs)
    parts = []
    for h in horizons:
        tcol = target_column(variable, h)
        rows = frame[frame[tcol].notna()] if require_target else frame
        if rows.empty:
            continue
        issued = pd.to_datetime(rows["date"])
        part = pd.DataFrame(
            {
                "reach_id": rows["reach_id"].astype(str).to_numpy(),
                "issued_date": issued.dt.date.to_numpy(),
                "horizon": h,
                "target_date": (issued + pd.Timedelta(days=h)).dt.date.to_numpy(),
                "target": rows[tcol].to_numpy(dtype="float64"),
                "reach_observable": (
                    rows["observable"].to_numpy() if "observable" in rows else pd.NA
                ),
            }
        )
        for col in base:
            src = rows[col] if col in rows else pd.Series(np.nan, index=rows.index)
            part[col] = _numeric(src, col).to_numpy()
        for col in fs.future_driver_columns:
            fcol = future_column(col, h)
            part[f"{col}_fut"] = _numeric(rows[fcol], fcol).to_numpy() if fcol in rows else np.nan
        if "precip_mm" in fs.future_driver_columns:
            cum = [future_column("precip_mm", k) for k in range(1, h + 1)]
            # Any missing day in t+1..t+h -> NaN, not a partial sum.
            part["precip_fut_cum_mm"] = (
                rows[cum].astype("float64").sum(axis=1, min_count=h).to_numpy()
            )
        parts.append(part)
    # reach_id and horizon are both keys and features - one column each.
    cols = list(dict.fromkeys([*KEY_COLUMNS, "target", "reach_observable", *all_features(fs)]))
    if not parts:
        return pd.DataFrame(columns=cols)
    long = pd.concat(parts, ignore_index=True)
    long["reach_id"] = pd.Categorical(long["reach_id"], categories=categories)
    long["horizon"] = long["horizon"].astype("float64")
    return long[cols]


def rearrange(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort quantile predictions per row. Returns (sorted, crossed) where crossed marks
    rows that were not already monotone."""
    crossed = np.any(np.diff(q, axis=1) < 0, axis=1)
    return np.sort(q, axis=1), crossed


# ---------------------------------------------------------------------------
# fit / predict (pure: DataFrames in, objects out)
# ---------------------------------------------------------------------------
@dataclass
class FittedGBM:
    version: str
    city: str
    fit_name: str
    train_end: date
    features: list[str]
    categories: list[str]
    settings: GBMSettings
    frame_settings: FrameSettings
    boosters: dict[tuple[str, str, float], lgb.Booster]
    manifest: dict[str, Any] = field(default_factory=dict)

    def booster(self, variable: str, horizon: int, alpha: float) -> lgb.Booster:
        return self.boosters[(variable, self.settings.bucket_of(horizon), alpha)]


def _lgb_params(gs: GBMSettings, alpha: float) -> dict[str, Any]:
    p = {k: v for k, v in gs.params.items() if k != "n_estimators"}
    p.update(
        objective="quantile",
        alpha=alpha,
        metric="quantile",
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    return p


def _data_fingerprint(long: pd.DataFrame, features: list[str]) -> str:
    h = pd.util.hash_pandas_object(long[[*KEY_COLUMNS, "target", *features]], index=False)
    return hashlib.sha256(h.to_numpy().tobytes()).hexdigest()


def fit(
    train: pd.DataFrame,
    fs: FrameSettings,
    gs: GBMSettings,
    *,
    city: str,
    fit_name: str,
    train_end: date,
    categories: list[str],
    variant: VariantSpec | None = None,
    dropped: dict[str, str] | None = None,
) -> FittedGBM:
    """Fit every (variable, bucket, quantile) booster on `train`, which must already be
    embargoed at `train_end` (pipeline.build_dataset.walk_forward / embargo).

    `variant` picks the feature set (A: with reach_id; B: without reach identity);
    `dropped` is {feature: reason} for features removed by policy, recorded in the
    manifest."""
    validate_settings(gs, fs)
    variant = variant or DEFAULT_VARIANT
    dropped = dict(dropped or {})
    features = feature_names(fs, variant, dropped)
    categorical = ["reach_id"] if "reach_id" in features else []
    boosters: dict[tuple[str, str, float], lgb.Booster] = {}
    rows: dict[str, int] = {}
    unused: dict[str, list[str]] = {}
    digest = hashlib.sha256()
    for variable in fs.variables:
        for bucket, horizons in gs.buckets.items():
            long = to_long(train, fs, variable, horizons, categories)
            key = f"{variable}/{bucket}"
            if len(long) < gs.min_train_rows:
                raise InsufficientTrainingData(
                    f"{city} {fit_name} {key}: {len(long)} target rows (need "
                    f">= {gs.min_train_rows}). The frame has too few OK Sentinel-2 "
                    "observations before the training cut-off - run pipeline.l1_satellite "
                    "over the full period and rebuild the frame."
                )
            latest = max(long["target_date"])
            if latest > train_end:  # the embargo is the leakage boundary; check it held
                raise AssertionError(f"{key}: target dated {latest} > train_end {train_end}")
            rows[key] = len(long)
            unused[key] = [f for f in features if long[f].isna().all()]
            digest.update(_data_fingerprint(long, features).encode())
            X, y = long[features], long["target"].to_numpy()
            for alpha in gs.quantiles:
                ds = lgb.Dataset(X, y, categorical_feature=categorical, free_raw_data=True)
                boosters[(variable, bucket, alpha)] = lgb.train(
                    _lgb_params(gs, alpha), ds, num_boost_round=int(gs.params["n_estimators"])
                )
            log.info("gbm.fit", fit=fit_name, model=key, rows=len(long), unused=unused[key])

    spec = {
        "settings": {
            "quantiles": gs.quantiles,
            "buckets": gs.buckets,
            "params": gs.params,
            "min_train_rows": gs.min_train_rows,
        },
        "features": features,
        "variant": variant.as_dict(),
        "dropped_features": dropped,
        "future_driver_columns": fs.future_driver_columns,
        "train_end": str(train_end),
        "lightgbm": lgb.__version__,
        "data": digest.hexdigest(),
    }
    fp = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()
    version = f"gbm-{gs.version}+{city}.{fit_name}.{fp[:10]}"
    manifest = {
        "version": version,
        "city": city,
        "fit": fit_name,
        "train_end": str(train_end),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_rows": rows,
        "all_null_features": unused,
        "categories": categories,
        **spec,
    }
    return FittedGBM(
        version, city, fit_name, train_end, features, categories, gs, fs, boosters, manifest
    )


def predict_long(model: FittedGBM, long: pd.DataFrame, variable: str) -> pd.DataFrame:
    """P10/P50/P90 (rearranged) for stacked rows of one variable."""
    out = long[[*KEY_COLUMNS, "target", "reach_observable"]].copy()
    names = [qname(a) for a in model.settings.quantiles]
    q = np.full((len(long), len(names)), np.nan)
    horizons = long["horizon"].to_numpy()
    for bucket, hs in model.settings.buckets.items():
        m = np.isin(horizons, hs)
        if not m.any():
            continue
        X = long.loc[m, model.features]
        for j, alpha in enumerate(model.settings.quantiles):
            q[m, j] = model.boosters[(variable, bucket, alpha)].predict(X)
    q, crossed = rearrange(q)
    for j, n in enumerate(names):
        out[n] = q[:, j]
    out["quantiles_crossed"] = crossed
    out["variable"] = variable
    out["model_version"] = model.version
    out["horizon"] = out["horizon"].astype(int)
    return out


def shap_long(model: FittedGBM, long: pd.DataFrame, variable: str) -> pd.DataFrame:
    """TreeSHAP for the shap_quantiles models, one row per (forecast, quantile)."""
    frames = []
    horizons = long["horizon"].to_numpy()
    numeric = [f for f in model.features if f != "reach_id"]
    for bucket, hs in model.settings.buckets.items():
        m = np.isin(horizons, hs)
        if not m.any():
            continue
        X = long.loc[m, model.features]
        for alpha in model.settings.shap_quantiles:
            contrib = np.asarray(
                model.boosters[(variable, bucket, alpha)].predict(X, pred_contrib=True)
            )
            part = long.loc[m, KEY_COLUMNS].copy()
            part["variable"] = variable
            part["quantile"] = qname(alpha)
            part["model_version"] = model.version
            for j, f in enumerate(model.features):
                part[f"shap__{f}"] = contrib[:, j]
            part["shap_bias"] = contrib[:, -1]
            for f in numeric:
                part[f"value__{f}"] = X[f].to_numpy()
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["horizon"] = out["horizon"].astype(int)
    out["reach_id"] = out["reach_id"].astype(str)
    return out


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def _booster_file(variable: str, bucket: str, alpha: float) -> str:
    return f"{variable}__{bucket}__{qname(alpha)}.txt"


def save(model: FittedGBM, root: Path = ARTIFACT_DIR) -> Path:
    path = root / model.version
    path.mkdir(parents=True, exist_ok=True)
    for (variable, bucket, alpha), booster in model.boosters.items():
        booster.save_model(str(path / _booster_file(variable, bucket, alpha)))
    (path / "manifest.json").write_text(
        json.dumps(model.manifest, indent=2, default=str), encoding="utf-8"
    )
    pointer = {"version": model.version, "trained_at": model.manifest["trained_at"]}
    (root / f"{model.city}__{model.fit_name}.json").write_text(json.dumps(pointer, indent=2))
    log.info("gbm.saved", version=model.version, path=str(path))
    return path


def load(
    city: str, fit_name: str = PRODUCTION, *, version: str | None = None, root: Path = ARTIFACT_DIR
) -> FittedGBM:
    if version is None:
        pointer = root / f"{city}__{fit_name}.json"
        if not pointer.exists():
            raise FileNotFoundError(
                f"no trained {fit_name} model for {city} ({pointer}) - run "
                f"`python -m models.baseline_gbm --city {city}`"
            )
        version = json.loads(pointer.read_text())["version"]
    path = root / version
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    fs = FrameSettings.from_config()
    s = manifest["settings"]
    gs = GBMSettings(
        version=version.split("+")[0].removeprefix("gbm-"),
        quantiles=tuple(s["quantiles"]),
        buckets={k: tuple(v) for k, v in s["buckets"].items()},
        min_train_rows=int(s["min_train_rows"]),
        params=dict(s["params"]),
        shap_quantiles=GBMSettings.from_config().shap_quantiles,
    )
    v = manifest.get("variant") or DEFAULT_VARIANT.as_dict()
    variant = VariantSpec(v["name"], bool(v["reach_id"]), tuple(v["drop"]))
    expected = feature_names(fs, variant, manifest.get("dropped_features") or {})
    if manifest["features"] != expected:
        raise RuntimeError(
            f"{version} was trained on a different feature list than the current config - retrain"
        )
    boosters = {
        (variable, bucket, float(alpha)): lgb.Booster(
            model_file=str(path / _booster_file(variable, bucket, float(alpha)))
        )
        for variable in fs.variables
        for bucket in gs.buckets
        for alpha in gs.quantiles
    }
    return FittedGBM(
        version,
        manifest["city"],
        manifest["fit"],
        as_date(manifest["train_end"]),
        manifest["features"],
        manifest["categories"],
        gs,
        fs,
        boosters,
        manifest,
    )


def frame_path(city: str) -> Path:
    return PROCESSED_DIR / f"frame_{city}.parquet"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frame(city: str, *, reach_id: str | None = None) -> pd.DataFrame:
    path = frame_path(city)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m pipeline.build_dataset`")
    filters = [("reach_id", "==", reach_id)] if reach_id else None
    frame = pd.read_parquet(path, filters=filters)
    if frame.empty:
        raise LookupError(f"no frame rows for {reach_id or city}")
    return frame


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------
def city_for_reach(reach_id: str) -> str:
    prefix = reach_id.split("-")[0]
    from core.settings import CONFIG_DIR

    for path in sorted((CONFIG_DIR / "cities").glob("*.yaml")):
        cfg = load_config(f"cities/{path.stem}")
        if cfg.get("reach_id_prefix") == prefix:
            return str(cfg["city"])
    raise LookupError(f"no city config has reach_id_prefix {prefix!r}")


@lru_cache(maxsize=8)
def _cached_model(city: str, fit_name: str) -> FittedGBM:
    return load(city, fit_name)


@lru_cache(maxsize=64)
def _cached_reach_frame(city: str, reach_id: str) -> pd.DataFrame:
    return load_frame(city, reach_id=reach_id)


def _issue_row(
    reach_id: str, issued_date: date, horizon: int, variable: str, fit_name: str
) -> tuple[FittedGBM, pd.DataFrame, pd.Series]:
    city = city_for_reach(reach_id)
    model = _cached_model(city, fit_name)
    fs = model.frame_settings
    if horizon not in fs.horizons:
        raise ValueError(f"horizon {horizon} not in {fs.horizons}")
    frame = _cached_reach_frame(city, reach_id)
    row = frame[pd.to_datetime(frame["date"]).dt.date == issued_date]
    if row.empty:
        raise LookupError(f"{reach_id} has no frame row for issue date {issued_date}")
    long = to_long(row, fs, variable, [horizon], model.categories, require_target=False)
    fut = [f for f in future_features(fs)]
    return model, long, long[fut].isna().sum(axis=1)


def predict(
    reach_id: str,
    issued_date: date,
    horizon: int,
    variable: str = "turbidity_proxy",
    *,
    fit_name: str = SERVING_FIT,
) -> dict[str, Any]:
    """{p10, p50, p90} for one reach, issue date and horizon (days), plus provenance.

    `future_drivers_missing` counts the t+h driver features that were NULL (no weather
    for the target date yet) - the forecast is still produced, but it is weaker, and the
    caller can see that. `in_sample` is true when the target date is inside the model's
    training period, i.e. the number is a fit, not a forecast.
    """
    model, long, missing = _issue_row(reach_id, issued_date, horizon, variable, fit_name)
    pred = predict_long(model, long, variable).iloc[0]
    target_date = pred["target_date"]
    return {
        **{qname(a): float(pred[qname(a)]) for a in model.settings.quantiles},
        "reach_id": reach_id,
        "variable": variable,
        "issued_date": issued_date,
        "horizon": horizon,
        "target_date": target_date,
        "model_version": model.version,
        "quantiles_crossed": bool(pred["quantiles_crossed"]),
        "future_drivers_missing": int(missing.iloc[0]),
        "in_sample": bool(target_date <= model.train_end),
    }


def explain(
    reach_id: str,
    issued_date: date,
    horizon: int,
    variable: str = "turbidity_proxy",
    *,
    alpha: float = 0.9,
    fit_name: str = SERVING_FIT,
) -> dict[str, Any]:
    """TreeSHAP contributions for one forecast quantile, largest magnitude first."""
    model, long, _ = _issue_row(reach_id, issued_date, horizon, variable, fit_name)
    booster = model.booster(variable, horizon, alpha)
    contrib = np.asarray(booster.predict(long[model.features], pred_contrib=True))[0]
    drivers: list[dict[str, Any]] = sorted(
        (
            {
                "feature": f,
                "value": None if f == "reach_id" else float(long[f].iloc[0]),
                "contribution": float(c),
            }
            for f, c in zip(model.features, contrib[:-1], strict=True)
        ),
        key=lambda d: -abs(float(d["contribution"])),
    )
    return {
        "reach_id": reach_id,
        "variable": variable,
        "horizon": horizon,
        "quantile": qname(alpha),
        "bias": float(contrib[-1]),
        "raw_prediction": float(contrib.sum()),
        "drivers": drivers,
        "model_version": model.version,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
ORACLE, ASISSUED = "ORACLE", "ASISSUED"


def substitute_future_drivers(
    long: pd.DataFrame, asissued: pd.DataFrame, fs: FrameSettings
) -> pd.DataFrame:
    """Replace the archive (oracle) t+h driver features with as-issued ones.

    asissued: reach_id, issued_date, horizon, and the future-driver columns named as in
    pipeline.asissued_weather (precip_mm, ..., precip_fut_cum_mm). Rows with no as-issued
    run are DROPPED - an as-issued score exists only where a forecast existed, and the
    caller compares oracle and as-issued on the rows both have.
    """
    cols = list(fs.future_driver_columns)
    src = asissued[["reach_id", "issued_date", "horizon", *cols, "precip_fut_cum_mm"]]
    src = src.rename(columns={c: f"{c}_fut" for c in cols}).assign(
        _rid=src["reach_id"].astype(str),
        horizon=src["horizon"].astype("float64"),
        issued_date=pd.to_datetime(src["issued_date"]).dt.date,
    )
    src = src.drop(columns="reach_id")
    base = long.drop(columns=future_features(fs)).assign(_rid=long["reach_id"].astype(str))
    merged = base.merge(
        src, on=["_rid", "issued_date", "horizon"], how="inner", validate="many_to_one"
    ).drop(columns="_rid")
    merged["reach_id"] = pd.Categorical(
        merged["reach_id"].astype(str), categories=long["reach_id"].cat.categories
    )
    return merged[list(long.columns)]


def forecast_and_explain(
    model: FittedGBM,
    rows: pd.DataFrame,
    *,
    require_target: bool,
    asissued: pd.DataFrame | None = None,
    explain: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict (and TreeSHAP-explain) every variable x horizon of `rows`. With
    `asissued`, the t+h drivers are replaced by as-issued forecasts first (rows without
    one are dropped) and the output is labelled weather=ASISSUED."""
    preds, shaps = [], []
    fs = model.frame_settings
    vname = model.manifest.get("variant", DEFAULT_VARIANT.as_dict())["name"]
    for variable in fs.variables:
        long = to_long(
            rows, fs, variable, fs.horizons, model.categories, require_target=require_target
        )
        if asissued is not None and not long.empty:
            long = substitute_future_drivers(long, asissued, fs)
        if long.empty:
            continue
        p = predict_long(model, long, variable)
        p["future_drivers_missing"] = long[future_features(fs)].isna().sum(axis=1).to_numpy()
        preds.append(p)
        if explain:
            shaps.append(shap_long(model, long, variable))
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    shap = pd.concat(shaps, ignore_index=True) if shaps else pd.DataFrame()
    if not pred.empty:
        pred["reach_id"] = pred["reach_id"].astype(str)
        pred["weather"] = ASISSUED if asissued is not None else ORACLE
        pred["variant"] = vname
    if not shap.empty:
        shap["variant"] = vname
    return pred, shap


def load_asissued(city: str) -> pd.DataFrame | None:
    """As-issued future drivers per reach (pipeline.asissued_weather output joined to
    each reach's weather cell). None if that pipeline has not been built."""
    path = PROCESSED_DIR / f"asissued_{city}.parquet"
    if not path.exists():
        return None
    from sqlalchemy import text

    from core.db import session_scope

    cells = pd.read_parquet(path)
    with session_scope() as session:
        points = pd.read_sql(
            text(
                "SELECT q.reach_id, q.query_lat, q.query_lon FROM driver_query_points q "
                "JOIN reaches r USING (reach_id) WHERE r.city = :c"
            ),
            session.connection(),
            params={"c": city},
        )
    out = points.merge(cells, on=["query_lat", "query_lon"], how="inner")
    return out.drop(columns=["query_lat", "query_lon"])


def train_all(city: str) -> dict[str, Any]:
    from pipeline.build_dataset import frame_meta_path

    fs, gs = FrameSettings.from_config(), GBMSettings.from_config()
    validate_settings(gs, fs)
    path = frame_path(city)
    frame = load_frame(city)
    categories = sorted(frame["reach_id"].astype(str).unique())
    meta_path = frame_meta_path(city)
    if gs.drop_when_proxy and not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing: cannot tell whether {list(gs.drop_when_proxy)} are "
            "land-cover proxies - rebuild the frame with pipeline.build_dataset"
        )
    static_flags = (
        json.loads(meta_path.read_text(encoding="utf-8"))["static_flags"]
        if meta_path.exists()
        else {}
    )
    dropped = proxy_dropped_features(gs, static_flags)
    asissued = load_asissued(city)
    run: dict[str, Any] = {
        "city": city,
        "frame": str(path.relative_to(REPO_ROOT)),
        "frame_sha256": file_sha256(path),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dropped_features": dropped,
        "variants": {k: v.as_dict() for k, v in gs.variants.items()},
        "asissued_rows": 0 if asissued is None else len(asissued),
        "fits": {},
    }
    if asissued is None:
        log.warning("gbm.no_asissued", note="as-issued weather not built - ORACLE only")
    evals, shaps, latests = [], [], []
    with stage(log, "baseline_gbm", city=city) as counters:
        for vname, variant in gs.variants.items():
            for fold in walk_forward(frame, fs):
                name = f"{vname}.wf-{fold['name']}"
                model = fit(
                    fold["train"],
                    fs,
                    gs,
                    city=city,
                    fit_name=name,
                    train_end=fold["train_end"],
                    categories=categories,
                    variant=variant,
                    dropped=dropped,
                )
                save(model)
                pred, shap = forecast_and_explain(model, fold["eval"], require_target=True)
                if pred.empty:
                    log.error("gbm.no_eval_targets", fold=name, note="evaluate will refuse")
                else:
                    pred["fold"] = fold["name"]
                    shap["fold"] = fold["name"]
                    evals.append(pred)
                    shaps.append(shap)
                n_asissued = 0
                if asissued is not None:
                    pa, _ = forecast_and_explain(
                        model, fold["eval"], require_target=True, asissued=asissued, explain=False
                    )
                    if not pa.empty:
                        pa["fold"] = fold["name"]
                        evals.append(pa)
                        n_asissued = len(pa)
                run["fits"][name] = {
                    "version": model.version,
                    "variant": vname,
                    "train_end": str(fold["train_end"]),
                    "eval_forecasts": len(pred),
                    "eval_forecasts_asissued": n_asissued,
                    "quantiles_crossed": int(pred["quantiles_crossed"].sum()) if len(pred) else 0,
                }

            last = pd.to_datetime(frame["date"]).max().date()
            pname = f"{vname}.{PRODUCTION}"
            prod = fit(
                embargo(frame, last, fs),
                fs,
                gs,
                city=city,
                fit_name=pname,
                train_end=last,
                categories=categories,
                variant=variant,
                dropped=dropped,
            )
            save(prod)
            latest_rows = frame[pd.to_datetime(frame["date"]).dt.date == last]
            latest, latest_shap = forecast_and_explain(prod, latest_rows, require_target=False)
            latest_shap["fold"] = PRODUCTION
            shaps.append(latest_shap)
            latests.append(latest)
            run["fits"][pname] = {
                "version": prod.version,
                "variant": vname,
                "train_end": str(last),
                "latest_issue_date": str(last),
                "latest_forecasts": len(latest),
                "latest_with_missing_future_drivers": int(
                    (latest["future_drivers_missing"] > 0).sum()
                ),
            }

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ev = pd.concat(evals, ignore_index=True) if evals else pd.DataFrame()
        latest_all = pd.concat(latests, ignore_index=True)
        ev.to_parquet(PROCESSED_DIR / f"gbm_eval_{city}.parquet", index=False)
        latest_all.to_parquet(PROCESSED_DIR / f"gbm_latest_{city}.parquet", index=False)
        pd.concat(shaps, ignore_index=True).to_parquet(
            PROCESSED_DIR / f"gbm_shap_{city}.parquet", index=False
        )
        (PROCESSED_DIR / f"gbm_run_{city}.json").write_text(json.dumps(run, indent=2))
        counters.record(
            rows_in=len(frame),
            rows_out=len(ev) + len(latest_all),
            eval_forecasts=len(ev),
            latest_forecasts=len(latest_all),
            dropped_features=dropped,
        )
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the LightGBM quantile baseline")
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    run = train_all(args.city)
    print(json.dumps(run, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
