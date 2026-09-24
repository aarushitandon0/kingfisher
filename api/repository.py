"""Data access for the API. PostGIS is the system of record (MASTERSPEC 5).

Routes depend on the `Repository` protocol, never on SQL: `PostgresRepository` is what the
server uses, and the contract tests substitute an in-memory one through FastAPI's
dependency overrides. Nothing here computes a number the UI shows - probabilities,
thresholds and rankings are computed by engine/ from what this returns.

Missing data is returned as NULL/None, never 0 (CLAUDE.md #2).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Protocol
from uuid import UUID

import pandas as pd
from sqlalchemy import text

from core.db import session_scope
from engine.scenarios import ScenarioResult, Status

FORECAST_COLUMNS = (
    "reach_id",
    "issued_date",
    "target_date",
    "horizon",
    "variable",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "model_version",
    "variant",
    "weather",
    "future_drivers_missing",
)
EXPOSURE_COLUMNS = ("reach_id", "feature_type", "count", "nearest_distance_m", "buffer_m", "source")
OBS_COLUMNS = ("reach_id", "date", "variable", "value")


class Repository(Protocol):
    def reach_counts(self) -> dict[str, dict[str, int]]:
        """city -> {reaches, optically_observable, driver_only, not_assessed}."""
        ...

    def reaches(self, city: str) -> list[dict[str, Any]]:
        """Every reach of a city: reaches columns + geometry (GeoJSON dict)."""
        ...

    def reach(self, reach_id: str) -> dict[str, Any] | None:
        """One reach + geometry, catchment_geometry, upstream_ids, downstream_ids."""
        ...

    def catchment_attributes(self, reach_id: str) -> dict[str, Any] | None: ...

    def observations(
        self, reach_id: str, start: date | None, end: date | None
    ) -> list[dict[str, Any]]:
        """Every observation row, flagged ones included (a cloud day is shown as one)."""
        ...

    def last_observation_date(self, reach_id: str) -> date | None: ...

    def ok_observations(self, city: str) -> pd.DataFrame:
        """OK S2 observations, long: reach_id, date, variable, value."""
        ...

    def exposure(self, reach_ids: Sequence[str]) -> pd.DataFrame: ...

    def catchment(self, reach_id: str, simplify_deg: float) -> dict[str, Any] | None:
        """Upstream catchment polygon (GeoJSON, simplified for display) + area; None for an
        unknown reach. The geometry is None where L0 could not delineate one."""
        ...

    def exposure_layer(self, city: str) -> list[dict[str, Any]]:
        """Distinct exposure features stored for the city's reaches (the nearest feature of
        each type per reach): feature_type, geometry (GeoJSON), reach_ids."""
        ...

    def latest_forecasts(
        self, city: str, variant: str, reach_id: str | None = None
    ) -> pd.DataFrame:
        """Production forecasts of the most recent issue date for the city (FORECAST_COLUMNS)."""
        ...

    def alert_run_date(self, city: str | None) -> date | None: ...

    def alert_run_suppressed(self, city: str | None, issued_date: date) -> dict[str, int] | None:
        """guardrail -> candidates it suppressed in that run (summed over cities when city
        is None); None if no alert_runs row records the run."""
        ...

    def alerts(self, city: str | None, issued_date: date | None) -> list[dict[str, Any]]: ...

    def alert(self, alert_id: UUID) -> dict[str, Any] | None: ...

    def save_scenario(
        self, name: str | None, city: str, config: dict[str, Any], result: ScenarioResult
    ) -> tuple[UUID, datetime]: ...

    def scenario(self, scenario_id: UUID) -> dict[str, Any] | None: ...


def _json(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) else v


class PostgresRepository:
    """SQL against the schema owned by migrations/versions/."""

    def reach_counts(self) -> dict[str, dict[str, int]]:
        sql = text(
            "SELECT city, count(*) AS reaches, "
            "count(*) FILTER (WHERE observable) AS optically_observable, "
            "count(*) FILTER (WHERE observable = false) AS driver_only, "
            "count(*) FILTER (WHERE observable IS NULL) AS not_assessed "
            "FROM reaches GROUP BY city"
        )
        with session_scope() as s:
            return {
                r.city: {
                    "reaches": r.reaches,
                    "optically_observable": r.optically_observable,
                    "driver_only": r.driver_only,
                    "not_assessed": r.not_assessed,
                }
                for r in s.execute(sql)
            }

    _REACH_COLS = (
        "r.reach_id, r.city, r.name, r.length_m, r.strahler_order, r.catchment_area_km2, "
        "r.observable, r.median_water_pixels, ST_AsGeoJSON(r.geom)::json AS geometry"
    )

    def reaches(self, city: str) -> list[dict[str, Any]]:
        sql = text(
            f"SELECT {self._REACH_COLS} FROM reaches r WHERE r.city = :c ORDER BY r.reach_id"
        )
        with session_scope() as s:
            return [
                {**dict(r._mapping), "geometry": _json(r.geometry)}
                for r in s.execute(sql, {"c": city})
            ]

    def reach(self, reach_id: str) -> dict[str, Any] | None:
        sql = text(
            f"SELECT {self._REACH_COLS}, "
            "ST_AsGeoJSON(r.catchment_geom)::json AS catchment_geometry, "
            "ARRAY(SELECT upstream_id FROM reach_topology WHERE downstream_id = r.reach_id "
            "      ORDER BY upstream_id) AS upstream_ids, "
            "ARRAY(SELECT downstream_id FROM reach_topology WHERE upstream_id = r.reach_id "
            "      ORDER BY downstream_id) AS downstream_ids "
            "FROM reaches r WHERE r.reach_id = :r"
        )
        with session_scope() as s:
            row = s.execute(sql, {"r": reach_id}).first()
        if row is None:
            return None
        out = dict(row._mapping)
        out["geometry"] = _json(out["geometry"])
        out["catchment_geometry"] = _json(out["catchment_geometry"])
        out["upstream_ids"] = list(out["upstream_ids"] or [])
        out["downstream_ids"] = list(out["downstream_ids"] or [])
        return out

    def catchment_attributes(self, reach_id: str) -> dict[str, Any] | None:
        sql = text(
            "SELECT year, imperviousness_pct, riparian_ndvi_mean, riparian_width_m, "
            "road_density_km_km2, alan_radiance, population, urban_fraction, adapter, "
            "sources, flags FROM catchment_attributes WHERE reach_id = :r"
        )
        with session_scope() as s:
            row = s.execute(sql, {"r": reach_id}).first()
        if row is None:
            return None
        out = dict(row._mapping)
        out["sources"] = _json(out["sources"])
        out["flags"] = _json(out["flags"])
        return out

    def observations(
        self, reach_id: str, start: date | None, end: date | None
    ) -> list[dict[str, Any]]:
        sql = text(
            "SELECT obs_date, source, quality_flag, turbidity_proxy, ndci, mndwi, "
            "water_pixel_count, cloud_fraction FROM observations WHERE reach_id = :r "
            "AND (CAST(:s AS date) IS NULL OR obs_date >= :s) "
            "AND (CAST(:e AS date) IS NULL OR obs_date <= :e) ORDER BY obs_date, source"
        )
        with session_scope() as s:
            return [dict(r._mapping) for r in s.execute(sql, {"r": reach_id, "s": start, "e": end})]

    def last_observation_date(self, reach_id: str) -> date | None:
        with session_scope() as s:
            v = s.execute(
                text("SELECT max(obs_date) FROM observations WHERE reach_id = :r"),
                {"r": reach_id},
            ).scalar()
        return v

    def ok_observations(self, city: str) -> pd.DataFrame:
        sql = text(
            "SELECT o.reach_id, o.obs_date AS date, o.turbidity_proxy, o.ndci "
            "FROM observations o JOIN reaches r USING (reach_id) "
            "WHERE r.city = :c AND o.quality_flag = 'OK' AND o.source = 'S2'"
        )
        with session_scope() as s:
            wide = pd.DataFrame(s.execute(sql, {"c": city}).fetchall())
        if wide.empty:
            return pd.DataFrame(columns=list(OBS_COLUMNS))
        long = wide.melt(
            id_vars=["reach_id", "date"],
            value_vars=["turbidity_proxy", "ndci"],
            var_name="variable",
            value_name="value",
        )
        long["value"] = pd.to_numeric(long["value"], errors="raise")
        return long[long["value"].notna()].reset_index(drop=True)[list(OBS_COLUMNS)]

    def exposure(self, reach_ids: Sequence[str]) -> pd.DataFrame:
        sql = text(
            f"SELECT {', '.join(EXPOSURE_COLUMNS)} FROM exposure_features "
            "WHERE reach_id = ANY(:ids) ORDER BY reach_id, feature_type"
        )
        with session_scope() as s:
            rows = s.execute(sql, {"ids": list(reach_ids)}).fetchall()
        return pd.DataFrame(rows, columns=list(EXPOSURE_COLUMNS))

    def catchment(self, reach_id: str, simplify_deg: float) -> dict[str, Any] | None:
        sql = text(
            "SELECT r.catchment_area_km2, "
            "ST_AsGeoJSON(ST_SimplifyPreserveTopology(r.catchment_geom, :tol), 6)::json AS g "
            "FROM reaches r WHERE r.reach_id = :r"
        )
        with session_scope() as s:
            row = s.execute(sql, {"r": reach_id, "tol": simplify_deg}).first()
        if row is None:
            return None
        return {"catchment_area_km2": row.catchment_area_km2, "geometry": _json(row.g)}

    def exposure_layer(self, city: str) -> list[dict[str, Any]]:
        sql = text(
            "SELECT e.feature_type, ST_AsGeoJSON(e.geom, 6)::json AS g, "
            "array_agg(DISTINCT e.reach_id ORDER BY e.reach_id) AS reach_ids "
            "FROM exposure_features e JOIN reaches r ON r.reach_id = e.reach_id "
            "WHERE r.city = :c AND e.geom IS NOT NULL "
            "GROUP BY e.feature_type, e.geom ORDER BY e.feature_type"
        )
        with session_scope() as s:
            return [
                {
                    "feature_type": r.feature_type,
                    "geometry": _json(r.g),
                    "reach_ids": list(r.reach_ids),
                }
                for r in s.execute(sql, {"c": city})
            ]

    def latest_forecasts(
        self, city: str, variant: str, reach_id: str | None = None
    ) -> pd.DataFrame:
        sql = text(
            "WITH latest AS ("
            "  SELECT f.issued_date, f.model_version FROM forecasts f "
            "  JOIN reaches r USING (reach_id) "
            "  WHERE r.city = :c AND f.fit = 'production' AND f.variant = :v "
            "  ORDER BY f.issued_date DESC, f.created_at DESC LIMIT 1) "
            f"SELECT {', '.join('f.' + c for c in FORECAST_COLUMNS)} "
            "FROM forecasts f JOIN reaches r USING (reach_id) JOIN latest l "
            "  ON f.issued_date = l.issued_date AND f.model_version = l.model_version "
            "WHERE r.city = :c AND f.fit = 'production' AND f.variant = :v "
            "AND (CAST(:rid AS text) IS NULL OR f.reach_id = :rid) "
            "ORDER BY f.reach_id, f.variable, f.target_date"
        )
        with session_scope() as s:
            rows = s.execute(sql, {"c": city, "v": variant, "rid": reach_id}).fetchall()
        return pd.DataFrame(rows, columns=list(FORECAST_COLUMNS))

    _ALERT_COLS = (
        "a.alert_id, a.reach_id, r.name AS reach_name, r.city, a.issued_date, "
        "lower(a.target_window) AS window_start, "
        "CASE WHEN upper_inc(a.target_window) THEN upper(a.target_window) "
        "     ELSE upper(a.target_window) - 1 END AS window_end, "
        "a.variable, a.severity, a.exceedance_prob, a.threshold_value, a.suppressed_reason, "
        "a.model_version"
    )

    def alert_run_date(self, city: str | None) -> date | None:
        """The latest run: an alert_runs row (a run that wrote no alerts is still a run) or
        alert rows written without one."""
        sql = text(
            "SELECT max(d) FROM ("
            "  SELECT issued_date AS d FROM alert_runs WHERE CAST(:c AS text) IS NULL OR city = :c"
            "  UNION ALL"
            "  SELECT a.issued_date FROM alerts a JOIN reaches r USING (reach_id) "
            "  WHERE CAST(:c AS text) IS NULL OR r.city = :c) runs"
        )
        with session_scope() as s:
            return s.execute(sql, {"c": city}).scalar()

    def alert_run_suppressed(self, city: str | None, issued_date: date) -> dict[str, int] | None:
        sql = text(
            "SELECT suppressed FROM alert_runs WHERE issued_date = :d "
            "AND (CAST(:c AS text) IS NULL OR city = :c)"
        )
        with session_scope() as s:
            rows = [_json(r.suppressed) for r in s.execute(sql, {"c": city, "d": issued_date})]
        if not rows:
            return None
        out: dict[str, int] = {}
        for r in rows:
            for k, v in r.items():
                out[k] = out.get(k, 0) + int(v)
        return out

    def alerts(self, city: str | None, issued_date: date | None) -> list[dict[str, Any]]:
        sql = text(
            f"SELECT {self._ALERT_COLS}, a.exposure FROM alerts a JOIN reaches r USING (reach_id) "
            "WHERE (CAST(:c AS text) IS NULL OR r.city = :c) "
            "AND (CAST(:d AS date) IS NULL OR a.issued_date = :d) "
            "ORDER BY a.issued_date DESC, a.reach_id, a.variable"
        )
        with session_scope() as s:
            return [dict(r._mapping) for r in s.execute(sql, {"c": city, "d": issued_date})]

    def alert(self, alert_id: UUID) -> dict[str, Any] | None:
        sql = text(
            f"SELECT {self._ALERT_COLS}, a.attribution, a.exposure, a.basis, a.guardrails, "
            "a.withheld_exceedance_prob, a.threshold_derivation, "
            "ST_AsGeoJSON(ST_LineInterpolatePoint(r.geom, 0.5))::json AS midpoint "
            "FROM alerts a JOIN reaches r USING (reach_id) WHERE a.alert_id = :id"
        )
        with session_scope() as s:
            row = s.execute(sql, {"id": alert_id}).first()
        if row is None:
            return None
        out = dict(row._mapping)
        for k in ("attribution", "exposure", "basis", "guardrails", "midpoint"):
            out[k] = _json(out[k])
        return out

    def save_scenario(
        self, name: str | None, city: str, config: dict[str, Any], result: ScenarioResult
    ) -> tuple[UUID, datetime]:
        """The scenario, its full result, and one scenario_results row per reach x variable,
        in one transaction. The table's checks refuse an OK row without citations or with
        an interval no wider than its baseline."""
        doc = result.model_dump(mode="json")
        with session_scope() as s:
            row = s.execute(
                text(
                    "INSERT INTO scenarios (name, city, config, result, model_version) "
                    "VALUES (:n, :c, CAST(:cfg AS jsonb), CAST(:res AS jsonb), :mv) "
                    "RETURNING scenario_id, created_at"
                ),
                {
                    "n": name or "unnamed scenario",
                    "c": city,
                    "cfg": json.dumps(config),
                    "res": json.dumps(doc),
                    "mv": result.model_version,
                },
            ).one()
            sid, created = row.scenario_id, row.created_at
            for r in result.reaches:
                ok = r.status is Status.OK
                used = {o.intervention_id for o in r.levers}
                cites = [
                    c.model_dump(mode="json") for c in result.citations if c.intervention_id in used
                ]
                s.execute(
                    text(
                        "INSERT INTO scenario_results (scenario_id, reach_id, variable, status, "
                        "reason, baseline_exceedance_days, baseline_ci_low, baseline_ci_high, "
                        "scenario_exceedance_days, ci_low, ci_high, delta, citations) VALUES "
                        "(:sid, :rid, :var, :st, :why, :b, :bl, :bh, :sc, :sl, :sh, :d, "
                        "CAST(:cit AS jsonb))"
                    ),
                    {
                        "sid": sid,
                        "rid": r.reach_id,
                        "var": r.variable,
                        "st": str(r.status),
                        "why": r.reason,
                        "b": r.baseline.exceedance_days if r.baseline else None,
                        "bl": r.baseline.interval.low if r.baseline else None,
                        "bh": r.baseline.interval.high if r.baseline else None,
                        "sc": r.scenario.exceedance_days if ok and r.scenario else None,
                        "sl": r.scenario.interval.low if ok and r.scenario else None,
                        "sh": r.scenario.interval.high if ok and r.scenario else None,
                        "d": r.delta_days if ok else None,
                        "cit": json.dumps(cites) if cites else None,
                    },
                )
        return sid, created

    def scenario(self, scenario_id: UUID) -> dict[str, Any] | None:
        sql = text(
            "SELECT scenario_id, name, city, created_at, result FROM scenarios "
            "WHERE scenario_id = :id"
        )
        with session_scope() as s:
            row = s.execute(sql, {"id": scenario_id}).first()
        if row is None:
            return None
        out = dict(row._mapping)
        out["result"] = _json(out["result"])
        return out
