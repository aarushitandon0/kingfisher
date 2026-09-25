"""Production calibration - the P5.2 split-conformal CQR, fitted on the 2024 validation
fold, applied to every variant-A forecast that reaches the `forecasts` table (and so to
the alert engine, the drift band and the API).

    python -m models.production_calibration --city coimbra      (run by `make evaluate`)

WHY THIS EXISTS
---------------
models/head_to_head.py fitted and scored the calibration (results/head_to_head.json), but
the production path wrote the RAW LightGBM quantiles: their 80% interval covered only
~55-61% of test observations, so exceedance probabilities were over-confident and the
drift guardrail's P10-P90 band was too narrow. The alert engine promises "a calibrated
probability of threshold exceedance" (engine/alerts.py); this module keeps that promise.

WHAT IS FITTED, ON WHAT
-----------------------
One CQR fit (models.calibration.fit_conformal - the same code the head-to-head uses) per
weather source, on variant A walk-forward VALIDATION rows only (fit_conformal refuses
anything else):

  ORACLE    applied to ORACLE rows
  ASISSUED  applied to ASISSUED and LIVE rows - a live forecast runs on forecast weather,
            so its errors look like the as-issued hindcast, not the observed-weather one.
            If a city has no as-issued validation rows (Pune: backfill not run), LIVE
            rows use the ORACLE fit and the record says so. That interval is then a
            LOWER bound on the width live weather needs.

Variant B is the scenario model; its intervals are widened by the scenario engine's own
rule and are not touched here. Isotonic recalibration of the exceedance probability is
scored in the head-to-head but NOT applied live - the probability comes from the
conformal-calibrated quantiles, and the record says that too.

The fit and its test-fold check (coverage before -> after, on rows the fit never saw)
are written to results/[<city>/]calibration_production.json.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_config
from core.logging import get_logger, stage
from core.settings import RESULTS_DIR, results_dir_for
from models import calibration as cal
from models.assimilation import _fwd, qcols, transforms_from_config

log = get_logger(__name__)

VARIANT = "A"
FIT_FOR_WEATHER = {"ORACLE": "ORACLE", "ASISSUED": "ASISSUED", "LIVE": "ASISSUED"}
FALLBACK_FIT = "ORACLE"


def _settings() -> tuple[dict[str, str], dict[str, tuple[int, ...]], dict[str, Any], Any]:
    from pipeline.build_dataset import FrameSettings

    cfg = load_config("modelling")
    buckets = {k: tuple(v) for k, v in cfg["baseline_gbm"]["horizon_buckets"].items()}
    return transforms_from_config(cfg), buckets, cfg["calibration"], FrameSettings.from_config()


def coverage_80(rows: pd.DataFrame, transforms: Mapping[str, str]) -> float | None:
    r = rows[rows["target"].notna()]
    if r.empty:
        return None
    inside = np.zeros(len(r), dtype=bool)
    for var, idx in r.groupby("variable").groups.items():
        g = r.loc[idx]
        tr = transforms[str(var)]
        y = _fwd(g["target"].to_numpy(dtype="float64"), tr)
        lo = _fwd(g["p10"].to_numpy(dtype="float64"), tr)
        hi = _fwd(g["p90"].to_numpy(dtype="float64"), tr)
        inside[r.index.get_indexer(idx)] = (y >= lo) & (y <= hi)
    return round(float(inside.mean()), 4)


def fit_all(
    ev: pd.DataFrame,
    *,
    transforms: Mapping[str, str],
    buckets: Mapping[str, Sequence[int]],
    ccfg: Mapping[str, Any],
    val_end: Any,
) -> dict[str, cal.ConformalFit]:
    """{weather: fit} from variant-A validation rows. A weather source with no validation
    rows gets no fit (never borrowed silently - apply() records the fallback)."""
    fits: dict[str, cal.ConformalFit] = {}
    a = ev[ev["variant"] == VARIANT]
    for wx in sorted(a["weather"].unique()):
        val = a[(a["weather"] == wx) & (a["fold"] == "val")]
        if val["target"].notna().sum() == 0:
            continue
        fits[str(wx)] = cal.fit_conformal(
            val,
            transforms=transforms,
            buckets=buckets,
            val_end=val_end,
            target_coverage=float(ccfg["target_coverage"]),
            min_rows=int(ccfg["min_conformal_rows"]),
        )
    if FALLBACK_FIT not in fits:
        raise RuntimeError(
            "no ORACLE variant-A validation rows to calibrate on - run `make train evaluate`"
        )
    return fits


def fit_used_for(weather: str, fits: Mapping[str, Any]) -> str:
    want = FIT_FOR_WEATHER.get(weather, FALLBACK_FIT)
    return want if want in fits else FALLBACK_FIT


def apply(
    rows: pd.DataFrame,
    fits: Mapping[str, cal.ConformalFit],
    *,
    transforms: Mapping[str, str],
    buckets: Mapping[str, Sequence[int]],
) -> pd.DataFrame:
    """Calibrated copy of the variant-A rows (variant B untouched). Needs reach_observable
    (NaN -> 'unassessed', which is never adjusted), horizon, variable, weather."""
    out = rows.copy()
    mask_a = (
        (out["variant"] == VARIANT).to_numpy()
        if "variant" in out
        else np.ones(len(out), dtype=bool)
    )
    parts = []
    for wx in out.loc[mask_a, "weather"].unique():
        m = mask_a & (out["weather"] == wx).to_numpy()
        fit = fits[fit_used_for(str(wx), fits)]
        adj = cal.apply_conformal(
            out[m].reset_index(drop=False), fit, transforms=transforms, buckets=buckets
        )
        adj = adj.set_index("index")
        parts.append(adj)
    if not parts:
        return out
    adj_all = pd.concat(parts)
    for c in qcols(out):
        out.loc[adj_all.index, c] = adj_all[c].to_numpy()
    return out


def load_fits(city: str) -> tuple[dict[str, cal.ConformalFit], dict[str, Any]]:
    """Refit from the walk-forward eval parquet (cheap: a sort per cell). Deterministic
    - the same parquet gives the same fit."""
    from pipeline.build_dataset import PROCESSED_DIR

    path = PROCESSED_DIR / f"gbm_eval_{city}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `make train` for {city}")
    ev = pd.read_parquet(path)
    ev["reach_id"] = ev["reach_id"].astype(str)
    transforms, buckets, ccfg, fs = _settings()
    fits = fit_all(ev, transforms=transforms, buckets=buckets, ccfg=ccfg, val_end=fs.val_end)
    return fits, {"eval": ev, "transforms": transforms, "buckets": buckets}


def calibrate_for_city(rows: pd.DataFrame, city: str) -> pd.DataFrame:
    fits, ctx = load_fits(city)
    return apply(rows, fits, transforms=ctx["transforms"], buckets=ctx["buckets"])


def report(city: str) -> dict[str, Any]:
    fits, ctx = load_fits(city)
    ev, transforms, buckets = ctx["eval"], ctx["transforms"], ctx["buckets"]
    a = ev[ev["variant"] == VARIANT]
    checks: dict[str, Any] = {}
    for wx in sorted(a["weather"].unique()):
        fit_name = fit_used_for(str(wx), fits)
        for fold in ("val", "test"):
            rows = a[(a["weather"] == wx) & (a["fold"] == fold)]
            if rows.empty:
                continue
            after = apply(rows, fits, transforms=transforms, buckets=buckets)
            per_var = {}
            for var in sorted(rows["variable"].unique()):
                per_var[str(var)] = {
                    "coverage_80_raw": coverage_80(rows[rows["variable"] == var], transforms),
                    "coverage_80_calibrated": coverage_80(
                        after[after["variable"] == var], transforms
                    ),
                }
            checks.setdefault(str(wx), {})[fold] = {
                "fit": fit_name,
                "n": int(rows["target"].notna().sum()),
                "by_variable": per_var,
                "note": "in-sample (the fit's own rows)" if fold == "val" else "held out",
            }
    live_fit = fit_used_for("LIVE", fits)
    return {
        "city": city,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "split-conformal CQR (Romano, Patterson & Candes 2019), models/calibration.py",
        "variant": VARIANT,
        "fitted_on": "variant A walk-forward validation fold only",
        "fits": {k: v.as_dict() for k, v in fits.items()},
        "live_rows_use_fit": live_fit,
        "live_note": None
        if live_fit == "ASISSUED"
        else "no as-issued validation rows for this city: LIVE intervals use the ORACLE "
        "fit, a lower bound on the width forecast weather needs",
        "probability": "exceedance probability from the calibrated quantiles; isotonic "
        "recalibration is scored in head_to_head.json but not applied live",
        "coverage_check": checks,
    }


def write_report(city: str, root: Path = RESULTS_DIR) -> Path:
    out_dir = results_dir_for(city, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration_production.json"
    with stage(log, "production_calibration", city=city) as s:
        rec = report(city)
        path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        s.record(rows_in=sum(c["n"] for w in rec["coverage_check"].values() for c in w.values()))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Production CQR calibration report")
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    path = write_report(args.city)
    rec = json.loads(path.read_text(encoding="utf-8"))
    for wx, folds in rec["coverage_check"].items():
        for fold, c in folds.items():
            for var, v in c["by_variable"].items():
                print(
                    f"{wx:9s} {fold:5s} {var:16s} cov80 raw {v['coverage_80_raw']} -> "
                    f"calibrated {v['coverage_80_calibrated']}  (fit {c['fit']})"
                )
    print(f"LIVE rows use the {rec['live_rows_use_fit']} fit. -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
