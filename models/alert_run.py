"""The live alert run: the latest production forecast -> guardrailed alerts in PostGIS.

    python -m models.alert_run --city coimbra          (make alerts)

Run after `make forecasts-latest`. The deciding logic is engine/alert_run.py (pure); this
module loads its inputs and writes:

  alerts       one row per ALERT / WATCH / INSUFFICIENT_EVIDENCE outcome (the refusals
               are rows too - CLAUDE.md #5). Re-running an issue date replaces its rows.
  alert_runs   one row per (city, issue date): severity counts, candidates suppressed
               per guardrail, forecast provenance. A run that alerts on nothing is still
               recorded, so /api/alerts can say "OK, nothing" instead of "no run".

Inputs: variant A production forecasts (the forecasting model), seasonal thresholds
fitted on OK observations up to the issue date, reach topology, earlier forecasts for
the drift residuals, earlier ALERT/WATCH rows for the cooldown, TreeSHAP rows written by
models.baseline_gbm for the same model version, and exposure_features.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from api.repository import PostgresRepository
from core.config import load_config
from core.db import session_scope
from core.logging import get_logger, stage
from engine.alert_run import (
    ATTRIBUTION_QUANTILE,
    RunInputs,
    RunResult,
    issued_at,
    run_alerts,
)
from engine.alerts import Alert, GuardrailConfig, HistoryEntry, Severity, to_record
from engine.thresholds import derivation_text, seasonal_thresholds
from pipeline.build_dataset import PROCESSED_DIR

log = get_logger(__name__)

FORECAST_VARIANT = "A"


def _reaches(city: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    with session_scope() as s:
        reaches = pd.DataFrame(
            s.execute(
                text("SELECT reach_id, observable FROM reaches WHERE city = :c"), {"c": city}
            ).fetchall(),
            columns=["reach_id", "observable"],
        )
        edges = s.execute(
            text(
                "SELECT t.downstream_id, t.upstream_id FROM reach_topology t "
                "JOIN reaches r ON r.reach_id = t.downstream_id WHERE r.city = :c"
            ),
            {"c": city},
        ).fetchall()
    if reaches.empty:
        raise RuntimeError(f"no reaches for {city} - run `make l0 city={city}`")
    upstream: dict[str, list[str]] = {}
    for down, up in edges:
        upstream.setdefault(str(down), []).append(str(up))
    return reaches, upstream


def _past_forecasts(city: str, issued: Any, window_days: int) -> pd.DataFrame:
    cols = [
        "reach_id",
        "variable",
        "issued_date",
        "target_date",
        "horizon",
        "fit",
        "weather",
        "p10",
        "p50",
        "p90",
        "model_version",
    ]
    sql = text(
        f"SELECT {', '.join('f.' + c for c in cols)} FROM forecasts f "
        "JOIN reaches r USING (reach_id) WHERE r.city = :c AND f.variant = :v "
        "AND f.target_date < :d AND f.target_date > :start"
    )
    with session_scope() as s:
        rows = s.execute(
            sql,
            {
                "c": city,
                "v": FORECAST_VARIANT,
                "d": issued,
                "start": issued - timedelta(days=window_days),
            },
        ).fetchall()
    return pd.DataFrame(rows, columns=cols)


def _history(city: str, issued: Any, cooldown_hours: float) -> list[HistoryEntry]:
    days = int(cooldown_hours // 24) + 1
    sql = text(
        "SELECT a.reach_id, a.issued_date, a.severity, a.variable FROM alerts a "
        "JOIN reaches r USING (reach_id) WHERE r.city = :c AND a.issued_date < :d "
        "AND a.issued_date >= :start AND a.severity IN ('ALERT', 'WATCH')"
    )
    with session_scope() as s:
        rows = s.execute(
            sql, {"c": city, "d": issued, "start": issued - timedelta(days=days)}
        ).fetchall()
    return [
        HistoryEntry(
            reach_id=r.reach_id,
            issued_at=issued_at(r.issued_date),
            severity=Severity(r.severity),
            variable=r.variable,
        )
        for r in rows
    ]


def _shap(city: str, model_version: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"gbm_shap_{city}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `make train`")
    shap = pd.read_parquet(path)
    shap = shap[
        (shap["fold"] == "production")
        & (shap["model_version"] == model_version)
        & (shap["quantile"] == ATTRIBUTION_QUANTILE)
    ]
    if shap.empty:
        raise RuntimeError(
            f"no {ATTRIBUTION_QUANTILE} SHAP rows for {model_version} in {path.name} - the "
            "forecast in the DB and the SHAP on disk are from different runs; re-run "
            "`make forecasts-latest`"
        )
    return shap


def load_inputs(city: str) -> RunInputs:
    repo = PostgresRepository()
    fc = repo.latest_forecasts(city, FORECAST_VARIANT)
    if fc.empty:
        raise RuntimeError(f"no production forecast for {city} - run `make forecasts-latest`")
    issued = pd.Timestamp(fc["issued_date"].iloc[0]).date()
    tcfg = load_config("thresholds")
    gcfg = GuardrailConfig.from_config(tcfg)
    obs = repo.ok_observations(city)
    if obs.empty:
        raise RuntimeError(f"no OK observations for {city} - nothing can be judged")
    reaches, upstream = _reaches(city)
    return RunInputs(
        issued=issued,
        forecasts=fc,
        thresholds=seasonal_thresholds(obs, issued, tcfg),
        seasons=tcfg["seasons"],
        reaches=reaches,
        upstream=upstream,
        observations=obs,
        past_forecasts=_past_forecasts(city, issued, gcfg.drift_window_days),
        history=_history(city, issued, gcfg.cooldown_hours),
        shap=_shap(city, str(fc["model_version"].iloc[0])),
        exposure=repo.exposure(reaches["reach_id"].tolist()),
        threshold_derivations={v: derivation_text(tcfg, v) for v in tcfg["variables"]},
    )


def _default(o: Any) -> Any:
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def persist(city: str, result: RunResult, derivations: dict[str, str]) -> int:
    """Replace the city's alerts for this issue date and record the run, atomically."""
    rows = [(o, to_record(o)) for o in result.alerts]
    with session_scope() as s:
        s.execute(
            text(
                "DELETE FROM alerts a USING reaches r WHERE a.reach_id = r.reach_id "
                "AND r.city = :c AND a.issued_date = :d"
            ),
            {"c": city, "d": result.issued},
        )
        for alert, rec in rows:
            assert isinstance(alert, Alert)
            lo, hi = rec["target_window"]
            s.execute(
                text(
                    "INSERT INTO alerts (reach_id, issued_date, target_window, variable, "
                    "exceedance_prob, threshold_value, severity, attribution, exposure, "
                    "suppressed_reason, model_version, basis, guardrails, "
                    "withheld_exceedance_prob, threshold_derivation) VALUES "
                    "(:r, :d, daterange(:lo, :hi, '[]'), :var, :p, :thr, :sev, "
                    "CAST(:attr AS jsonb), CAST(:exp AS jsonb), :why, :mv, "
                    "CAST(:basis AS jsonb), CAST(:g AS jsonb), :wh, :deriv)"
                ),
                {
                    "r": rec["reach_id"],
                    "d": rec["issued_date"],
                    "lo": lo,
                    "hi": hi,
                    "var": rec["variable"],
                    "p": rec["exceedance_prob"],
                    "thr": rec["threshold_value"],
                    "sev": rec["severity"],
                    "attr": json.dumps(rec["attribution"]),
                    "exp": json.dumps(rec["exposure"], default=_default),
                    "why": rec["suppressed_reason"],
                    "mv": rec["model_version"],
                    "basis": json.dumps(dict(alert.basis), default=_default),
                    "g": json.dumps(rec["guardrails"]),
                    "wh": rec["withheld_exceedance_prob"],
                    "deriv": derivations[rec["variable"]],
                },
            )
        s.execute(
            text(
                "INSERT INTO alert_runs (city, issued_date, model_version, weather, counts, "
                "suppressed, basis) VALUES (:c, :d, :mv, :w, CAST(:n AS jsonb), "
                "CAST(:sup AS jsonb), CAST(:b AS jsonb)) "
                "ON CONFLICT (city, issued_date) DO UPDATE SET model_version = EXCLUDED."
                "model_version, weather = EXCLUDED.weather, counts = EXCLUDED.counts, "
                "suppressed = EXCLUDED.suppressed, basis = EXCLUDED.basis, created_at = now()"
            ),
            {
                "c": city,
                "d": result.issued,
                "mv": result.model_version,
                "w": result.weather,
                "n": json.dumps(result.counts),
                "sup": json.dumps(result.suppressed),
                "b": json.dumps({"candidates": len(result.outcomes)}),
            },
        )
    return len(rows)


def run(city: str) -> dict[str, Any]:
    with stage(log, "alert_run", city=city) as counters:
        inp = load_inputs(city)
        cfg = GuardrailConfig.from_config(load_config("thresholds"))
        result = run_alerts(inp, cfg)
        written = persist(city, result, dict(inp.threshold_derivations))
        counters.record(
            rows_in=len(result.outcomes),
            rows_out=written,
            counts=result.counts,
            suppressed=result.suppressed,
        )
    return {
        "city": city,
        "issued_date": str(result.issued),
        "model_version": result.model_version,
        "weather": result.weather,
        "candidates": len(result.outcomes),
        "alerts_written": written,
        "counts": result.counts,
        "suppressed_by_guardrail": result.suppressed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live alert run -> alerts table")
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.city), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
