"""API contract tests - every MASTERSPEC 12 endpoint against its Pydantic schema.

No database, no trained model: the repository, the scenario runner and the results
directory are replaced through FastAPI dependency overrides. Each test parses the response
with the schema the route declares and then checks the behaviour the contract promises:
NULL (not 0) where evidence is missing, INSUFFICIENT_EVIDENCE shown and never given a
probability, planning-estimate scenarios with citations and strictly wider intervals,
metrics served live from the file. tests/test_api_db.py runs the real SQL.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.deps import get_repository, get_results_dir, get_scenario_runner, get_shap_source
from api.main import app
from api.repository import EXPOSURE_COLUMNS, FORECAST_COLUMNS, OBS_COLUMNS
from api.routes.export import FhirBundle
from api.schemas import (
    AlertDetail,
    AlertsResponse,
    AttributionResponse,
    CatchmentFeature,
    CitiesResponse,
    ExposureLayer,
    ForecastResponse,
    InterventionCatalogue,
    PrioritiesResponse,
    ReachCollection,
    ReachDetail,
    ScenarioResponse,
    TimelineResponse,
    ValidationMetricsResponse,
)
from core.config import load_intervention_coefficients
from engine.scenarios import (
    InterventionRequest,
    MissingResponseCheck,
    ScenarioResult,
    ScenarioSettings,
    Status,
    decide_path,
    run_scenario,
)
from tests.test_scenarios import SEASONS, LinearModel, checks, make_frame, make_thresholds

ISSUED = date(2026, 9, 20)
OBS_REACH, DRIVER_REACH = "CMB-0001", "CMB-0002"
ALERT_ID = UUID("11111111-1111-1111-1111-111111111111")
IE_ID = UUID("22222222-2222-2222-2222-222222222222")
LINE = {"type": "LineString", "coordinates": [[-8.44, 40.21], [-8.43, 40.22]]}
POLY = {
    "type": "Polygon",
    "coordinates": [[[-8.5, 40.2], [-8.4, 40.2], [-8.4, 40.3], [-8.5, 40.2]]],
}


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
def _reach(rid: str, observable: bool | None) -> dict[str, Any]:
    return {
        "reach_id": rid,
        "city": "coimbra",
        "name": "Ribeira de Teste" if observable else None,
        "length_m": 300.0,
        "strahler_order": 3,
        "catchment_area_km2": 12.5,
        "observable": observable,
        "median_water_pixels": 40.0 if observable else 0.0,
        "geometry": LINE,
    }


def _alert(aid: UUID, severity: str, prob: float | None, reason: str | None) -> dict[str, Any]:
    return {
        "alert_id": aid,
        "reach_id": OBS_REACH,
        "reach_name": "Ribeira de Teste",
        "city": "coimbra",
        "issued_date": ISSUED,
        "window_start": ISSUED + timedelta(days=1),
        "window_end": ISSUED + timedelta(days=3),
        "variable": "turbidity_proxy",
        "severity": severity,
        "exceedance_prob": prob,
        "threshold_value": 12.3,
        "suppressed_reason": reason,
        "model_version": "gbm-test+coimbra.A.production.0",
        "attribution": [
            {
                "feature": "antecedent_dry_days",
                "contribution": 0.31,
                "value": 18.0,
                "kind": "weather_driver",
            }
        ],
        "exposure": {"buffer_m": 250.0, "population": 120, "features": {"school": {"count": 1}}},
        "basis": {"usable_observations_30d": 6, "last_usable_observation": "2026-09-15"},
        "guardrails": {"staleness": "pass", "drift": "pass", "confidence": "pass"},
        "withheld_exceedance_prob": 0.7 if severity == "INSUFFICIENT_EVIDENCE" else None,
        "threshold_derivation": "per_reach_seasonal_percentile",
        "midpoint": {"type": "Point", "coordinates": [-8.435, 40.215]},
    }


class FakeRepository:
    def __init__(self, *, with_alerts: bool = True) -> None:
        self._reaches = {
            OBS_REACH: _reach(OBS_REACH, True),
            DRIVER_REACH: _reach(DRIVER_REACH, False),
        }
        self._alerts = (
            [
                _alert(ALERT_ID, "ALERT", 0.78, None),
                _alert(IE_ID, "INSUFFICIENT_EVIDENCE", None, "STALE_EVIDENCE: 30 days"),
            ]
            if with_alerts
            else []
        )
        self.scenarios: dict[UUID, dict[str, Any]] = {}

    def reach_counts(self) -> dict[str, dict[str, int]]:
        return {
            "coimbra": {
                "reaches": 2,
                "optically_observable": 1,
                "driver_only": 1,
                "not_assessed": 0,
            }
        }

    def reaches(self, city: str) -> list[dict[str, Any]]:
        return list(self._reaches.values()) if city == "coimbra" else []

    def reach(self, reach_id: str) -> dict[str, Any] | None:
        r = self._reaches.get(reach_id)
        if r is None:
            return None
        return {
            **r,
            "catchment_geometry": POLY,
            "upstream_ids": [],
            "downstream_ids": [DRIVER_REACH],
        }

    def catchment_attributes(self, reach_id: str) -> dict[str, Any] | None:
        return {
            "year": 2021,
            "imperviousness_pct": 12.0,
            "alan_radiance": None,
            "flags": {"alan_radiance": "SOURCE_UNAVAILABLE"},
        }

    def observations(
        self, reach_id: str, start: date | None, end: date | None
    ) -> list[dict[str, Any]]:
        return [
            {
                "obs_date": ISSUED - timedelta(days=10),
                "source": "S2",
                "quality_flag": "OK",
                "turbidity_proxy": 8.1,
                "ndci": 0.05,
                "mndwi": 0.2,
                "water_pixel_count": 40,
                "cloud_fraction": 0.0,
            },
            {
                "obs_date": ISSUED - timedelta(days=5),
                "source": "S2",
                "quality_flag": "CLOUD",
                "turbidity_proxy": None,
                "ndci": None,
                "mndwi": None,
                "water_pixel_count": None,
                "cloud_fraction": 0.9,
            },
        ]

    def last_observation_date(self, reach_id: str) -> date | None:
        return ISSUED - timedelta(days=5)

    def ok_observations(self, city: str) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        days = pd.date_range("2022-01-01", "2026-09-15", freq="5D")
        rows = []
        for d in days:
            rows.append((OBS_REACH, d, "turbidity_proxy", float(rng.gamma(4, 2))))
            rows.append((OBS_REACH, d, "ndci", float(rng.normal(0, 0.1))))
        return pd.DataFrame(rows, columns=list(OBS_COLUMNS))

    def exposure(self, reach_ids: Sequence[str]) -> pd.DataFrame:
        rows = []
        for rid in reach_ids:
            for ft, n in (
                ("school", 1),
                ("footway", 4),
                ("population", None if rid == DRIVER_REACH else 900),
            ):
                rows.append((rid, ft, n, 50.0 if n else None, 250.0, "OSM"))
        return pd.DataFrame(rows, columns=list(EXPOSURE_COLUMNS))

    def catchment(self, reach_id: str, simplify_deg: float) -> dict[str, Any] | None:
        if reach_id not in self._reaches:
            return None
        return {"catchment_area_km2": 12.5, "geometry": POLY}

    def exposure_layer(self, city: str) -> list[dict[str, Any]]:
        if city != "coimbra":
            return []
        return [
            {
                "feature_type": "school",
                "geometry": {"type": "Point", "coordinates": [-8.43, 40.21]},
                "reach_ids": [OBS_REACH, DRIVER_REACH],
            }
        ]

    def latest_forecasts(
        self, city: str, variant: str, reach_id: str | None = None
    ) -> pd.DataFrame:
        rows = []
        for rid in (OBS_REACH, DRIVER_REACH):
            if reach_id and rid != reach_id:
                continue
            for var, base, spread in (("turbidity_proxy", 10.0, 3.0), ("ndci", 0.0, 0.1)):
                for h in range(1, 11):
                    q = base + spread * np.array([-1.6, -1.3, -0.7, 0, 0.7, 1.3, 1.6]) + 0.1 * h
                    rows.append(
                        (
                            rid,
                            ISSUED,
                            ISSUED + timedelta(days=h),
                            h,
                            var,
                            *q,
                            "gbm-test+coimbra.A.production.0",
                            "A",
                            "ORACLE",
                            6,
                        )
                    )
        return pd.DataFrame(rows, columns=list(FORECAST_COLUMNS))

    def alert_run_date(self, city: str | None) -> date | None:
        return ISSUED if self._alerts else None

    def alert_run_suppressed(self, city: str | None, issued_date: date) -> dict[str, int] | None:
        return {"confidence": 2} if self._alerts else None

    def alerts(self, city: str | None, issued_date: date | None) -> list[dict[str, Any]]:
        return [a for a in self._alerts if issued_date is None or a["issued_date"] == issued_date]

    def alert(self, alert_id: UUID) -> dict[str, Any] | None:
        return next((a for a in self._alerts if a["alert_id"] == alert_id), None)

    def save_scenario(
        self, name: str | None, city: str, config: dict[str, Any], result: ScenarioResult
    ) -> tuple[UUID, datetime]:
        sid, created = uuid.uuid4(), datetime.now(UTC)
        self.scenarios[sid] = {
            "scenario_id": sid,
            "name": name,
            "city": city,
            "created_at": created,
            "result": result.model_dump(mode="json"),
        }
        return sid, created

    def scenario(self, scenario_id: UUID) -> dict[str, Any] | None:
        return self.scenarios.get(scenario_id)


def fake_shap(city: str, model_version: str, quantile: str) -> pd.DataFrame:
    """Stored-SHAP rows shaped like data/processed/gbm_shap_<city>.parquet."""
    rows = []
    for rid in (OBS_REACH, DRIVER_REACH):
        for var in ("turbidity_proxy", "ndci"):
            for h in range(1, 11):
                rows.append(
                    {
                        "reach_id": rid,
                        "variable": var,
                        "target_date": ISSUED + timedelta(days=h),
                        "shap__reach_id": 5.0,  # identity: never shown as a driver
                        "shap__horizon": 1.0,
                        "shap__precip_mm_fut": 0.1 * h,
                        "shap__antecedent_dry_days": -0.2,
                        "shap__imperviousness_pct": 0.0,  # did not split: dropped
                        "value__precip_mm_fut": 2.0 * h,
                        "value__antecedent_dry_days": 18.0,
                        "value__imperviousness_pct": 12.0,
                    }
                )
    return pd.DataFrame(rows)


RENAME = {"R1": OBS_REACH, "R2": DRIVER_REACH, "R3": "CMB-0003"}


class FakeRunner:
    """The real engine on the linear fake model, with CMB- reach ids."""

    def __init__(self) -> None:
        table = load_intervention_coefficients()
        self.coefficients = table
        self.settings = ScenarioSettings(
            reference_start=date(2025, 1, 1),
            reference_end=date(2025, 12, 31),
            horizon=1,
            ci_level=0.8,
            coefficient_samples=5,
            widening_factor=1.5,
            caveat=str(table.uncertainty_inflation["caveat"]),
            seasons=SEASONS,
        )
        f = make_frame()
        f["reach_id"] = f["reach_id"].map(RENAME)
        self.features = f
        t = make_thresholds(("R1",))
        t["reach_id"] = t["reach_id"].map(RENAME)
        self.thresholds = t

    @property
    def reach_ids(self) -> frozenset[str]:
        return frozenset(self.features["reach_id"])

    @property
    def variables(self) -> tuple[str, ...]:
        return ("turbidity_proxy", "ndci")

    def lever_paths(self) -> dict[str, dict[str, tuple[str, str]]]:
        model = LinearModel()
        return {
            c.id: {
                v: (str(p), why)
                for v in self.variables
                for p, _, why in [decide_path(c, v, model, checks(), self.coefficients)]
            }
            for c in self.coefficients.interventions
        }

    def run(
        self, interventions: Sequence[InterventionRequest], variables: Sequence[str] | None = None
    ) -> ScenarioResult:
        rids = tuple(dict.fromkeys(r for i in interventions for r in i.reach_ids))
        return run_scenario(
            rids,
            interventions,
            LinearModel(),
            self.features,
            coefficients=self.coefficients,
            thresholds=self.thresholds,
            response_checks=checks(),
            settings=self.settings,
            variables=tuple(variables or ("turbidity_proxy", "ndci")),
        )


METRICS = {
    "city": "coimbra",
    "generated_at": "2026-09-21T13:30:15+00:00",
    "model_versions": {"A.production": "gbm-test"},
    "headline_run": "A/ORACLE",
    "folds": {"test": {"turbidity_proxy": {"by_horizon": {}}}},
    "runs": {},
    "losses": [{"reach_id": OBS_REACH, "loses_to": "seasonal_naive"}],
    "not_computed": {"transfer": "Pune not built"},
    "anomaly": {"scored_against": "PROXY"},
}


@pytest.fixture
def repo() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    (tmp_path / "metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(repo: FakeRepository, results_dir: Path) -> Iterator[TestClient]:
    runner = FakeRunner()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_results_dir] = lambda: results_dir
    app.dependency_overrides[get_scenario_runner] = lambda: lambda city: runner
    app.dependency_overrides[get_shap_source] = lambda: fake_shap
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# the surface
# ---------------------------------------------------------------------------
SPEC_ROUTES = {
    ("GET", "/api/cities"),
    ("GET", "/api/reaches"),
    ("GET", "/api/reaches/{reach_id}"),
    ("GET", "/api/reaches/{reach_id}/forecast"),
    ("GET", "/api/alerts"),
    ("GET", "/api/alerts/{alert_id}"),
    ("POST", "/api/scenarios"),
    ("GET", "/api/scenarios/{scenario_id}"),
    ("GET", "/api/priorities"),
    ("GET", "/api/validation/metrics"),
    ("GET", "/api/export/fhir/{alert_id}"),
}


def test_every_masterspec_route_exists_with_a_response_schema() -> None:
    spec = app.openapi()
    found = {(m.upper(), p) for p, ops in spec["paths"].items() for m in ops}
    assert found >= SPEC_ROUTES
    for method, path in SPEC_ROUTES:
        op = spec["paths"][path][method.lower()]
        ok = op["responses"]["201" if method == "POST" else "200"]
        assert "$ref" in json.dumps(ok["content"]["application/json"]["schema"]), path


def test_catchment_hover_route(client: TestClient) -> None:
    body = CatchmentFeature.model_validate(client.get(f"/api/reaches/{OBS_REACH}/catchment").json())
    assert body.geometry is not None and body.geometry.type == "Polygon"
    assert body.properties["unit"] == "upstream contributing catchment"
    assert client.get("/api/reaches/CMB-9999/catchment").status_code == 404


def test_attribution_from_stored_shap(client: TestClient) -> None:
    body = AttributionResponse.model_validate(
        client.get(f"/api/reaches/{OBS_REACH}/attribution").json()
    )
    by = {s.variable: s for s in body.series}
    assert set(by) == {"turbidity_proxy", "ndci"}
    for s in body.series:
        feats = [c.feature for c in s.contributions]
        assert "reach_id" not in feats and "horizon" not in feats  # identity is not a driver
        assert "imperviousness_pct" not in feats  # zero contribution is dropped, not shown
        mags = [abs(c.contribution) for c in s.contributions]
        assert mags == sorted(mags, reverse=True)
    # the observable reach has thresholds -> attributed at its peak P(exceed)
    assert by["turbidity_proxy"].selection == "PEAK_EXCEEDANCE"


def test_attribution_driver_only_reach_says_it_is_not_an_exceedance(client: TestClient) -> None:
    body = AttributionResponse.model_validate(
        client.get(f"/api/reaches/{DRIVER_REACH}/attribution").json()
    )
    assert body.series and all(s.selection == "PEAK_P90" for s in body.series)


def test_attribution_without_stored_shap_is_503(client: TestClient) -> None:
    app.dependency_overrides[get_shap_source] = lambda: lambda *a: pd.DataFrame()
    assert client.get(f"/api/reaches/{OBS_REACH}/attribution").status_code == 503


def test_exposure_layer(client: TestClient) -> None:
    body = ExposureLayer.model_validate(client.get("/api/exposure?city=coimbra").json())
    assert body.features[0].properties.reaches == 2
    assert "no health outcome" in body.note
    assert client.get("/api/exposure?city=atlantis").status_code == 404


def test_validation_metrics_per_city(client: TestClient, results_dir: Path) -> None:
    (results_dir / "pune").mkdir()
    (results_dir / "pune" / "metrics.json").write_text(
        json.dumps({**METRICS, "city": "pune"}), encoding="utf-8"
    )
    assert client.get("/api/validation/metrics?city=pune").json()["metrics"]["city"] == "pune"
    assert client.get("/api/validation/metrics?city=coimbra").json()["metrics"]["city"] == "coimbra"
    assert client.get("/api/validation/metrics").json()["metrics"]["city"] == "coimbra"
    assert client.get("/api/validation/metrics?city=atlantis").status_code == 404
    # a Pune request never falls back to Coimbra's document
    (results_dir / "pune" / "metrics.json").unlink()
    assert client.get("/api/validation/metrics?city=pune").status_code == 503


def test_timeline_is_columnar_and_never_fills_gaps(
    client: TestClient, repo: FakeRepository
) -> None:
    body = TimelineResponse.model_validate(client.get("/api/timeline?city=coimbra").json())
    obs = body.observations
    assert (
        len(obs.reach_id) == len(obs.date) == len(obs.value) == len(repo.ok_observations("coimbra"))
    )
    assert all(v is not None for v in obs.value)  # OK rows only: a flagged date is absent
    fc = body.forecast
    assert len(fc.reach_id) == 2 * 2 * 10
    # the driver-only reach has no threshold -> NULL probability, never 0
    driver = [p for r, p in zip(fc.reach_id, fc.exceedance_prob, strict=True) if r == DRIVER_REACH]
    assert driver and all(p is None for p in driver)
    assert body.issued_date == ISSUED
    assert client.get("/api/timeline?city=atlantis").status_code == 404


def test_cors_allows_the_vite_dev_server_only(client: TestClient) -> None:
    pre = {"Access-Control-Request-Method": "GET"}
    ok = client.options("/api/cities", headers={"Origin": "http://localhost:5173", **pre})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/api/cities", headers={"Origin": "http://evil.example", **pre})
    assert "access-control-allow-origin" not in bad.headers


# ---------------------------------------------------------------------------
# cities + reaches
# ---------------------------------------------------------------------------
def test_cities(client: TestClient) -> None:
    body = CitiesResponse.model_validate(client.get("/api/cities").json())
    by = {c.city: c for c in body.cities}
    assert by["coimbra"].status == "READY" and by["coimbra"].driver_only == 1
    assert by["pune"].status == "NOT_BUILT" and by["pune"].reaches == 0


def test_reaches_geojson_with_null_not_zero_for_driver_only(client: TestClient) -> None:
    r = client.get("/api/reaches", params={"city": "coimbra"})
    assert r.status_code == 200
    body = ReachCollection.model_validate(r.json())
    assert body.type == "FeatureCollection" and body.forecast_issued_date == ISSUED
    props = {f.id: f.properties for f in body.features}
    obs, drv = props[OBS_REACH], props[DRIVER_REACH]
    assert obs.exceedance_status == "OK" and 0 <= obs.p_exceed_max <= 1
    assert drv.exceedance_status == "NO_THRESHOLD"
    assert drv.p_exceed_max is None and drv.p_exceed_turbidity_proxy is None
    assert drv.observability == "DRIVER_ONLY"
    assert obs.alert_severity == "ALERT" and obs.alert_id == ALERT_ID
    # MapLibre-safe: properties are flat scalars
    for f in r.json()["features"]:
        assert f["geometry"]["type"] == "LineString"
        assert not any(isinstance(v, dict | list) for v in f["properties"].values())


@pytest.mark.parametrize(
    ("params", "status"), [({"city": "atlantis"}, 404), ({"city": "pune"}, 404), ({}, 422)]
)
def test_reaches_errors(client: TestClient, params: dict[str, str], status: int) -> None:
    assert client.get("/api/reaches", params=params).status_code == status


def test_reach_detail(client: TestClient) -> None:
    r = client.get(f"/api/reaches/{OBS_REACH}")
    assert r.status_code == 200
    d = ReachDetail.model_validate(r.json())
    assert d.type == "Feature" and d.catchment.geometry.type == "Polygon"
    assert r.json()["catchment"]["type"] == "Feature"  # RFC 7946 foreign member
    flagged = [h for h in d.properties.history if h.quality_flag == "CLOUD"]
    assert flagged and flagged[0].turbidity_proxy is None  # a cloud day is NULL, never 0
    assert d.properties.thresholds and all(t.fit_end == ISSUED for t in d.properties.thresholds)
    assert d.properties.exposure.features["school"].count == 1
    assert d.properties.catchment_attributes["alan_radiance"] is None
    assert client.get("/api/reaches/CMB-9999").status_code == 404


def test_forecast(client: TestClient) -> None:
    r = client.get(f"/api/reaches/{OBS_REACH}/forecast", params={"horizon": 3})
    body = ForecastResponse.model_validate(r.json())
    assert {s.variable for s in body.series} == {"turbidity_proxy", "ndci"}
    for s in body.series:
        assert [d.horizon for d in s.days] == [1, 2, 3]
        for d in s.days:
            assert d.p05 <= d.p10 <= d.p50 <= d.p90 <= d.p95
            assert d.threshold is not None and 0 <= d.exceedance_prob <= 1
    assert any("no weather for the target day" in n for n in body.notes)


def test_forecast_for_driver_only_reach_has_no_probability(client: TestClient) -> None:
    body = ForecastResponse.model_validate(
        client.get(f"/api/reaches/{DRIVER_REACH}/forecast").json()
    )
    assert body.observability == "DRIVER_ONLY" and any("Driver-predicted" in n for n in body.notes)
    for s in body.series:
        assert len(s.days) == 10
        assert all(d.exceedance_prob is None and d.threshold is None for d in s.days)


@pytest.mark.parametrize("h", [0, 11])
def test_forecast_horizon_is_bounded(client: TestClient, h: int) -> None:
    assert (
        client.get(f"/api/reaches/{OBS_REACH}/forecast", params={"horizon": h}).status_code == 422
    )


# ---------------------------------------------------------------------------
# alerts + FHIR
# ---------------------------------------------------------------------------
def test_alerts_include_insufficient_evidence_sorted_by_severity(client: TestClient) -> None:
    body = AlertsResponse.model_validate(
        client.get("/api/alerts", params={"city": "coimbra"}).json()
    )
    assert body.alert_run == "OK" and body.alert_run_issued_date == ISSUED
    assert [a.severity for a in body.alerts] == ["ALERT", "INSUFFICIENT_EVIDENCE"]
    ie = body.alerts[1]
    assert ie.exceedance_prob is None and ie.suppressed_reason
    assert body.counts == {"ALERT": 1, "WATCH": 0, "INSUFFICIENT_EVIDENCE": 1}
    assert body.suppressed == {"confidence": 2}  # counted, not alert rows
    everything = client.get("/api/alerts", params={"active": "false"}).json()
    assert len(everything["alerts"]) == 2 and everything["city"] is None


def test_no_alert_run_is_said_not_passed_off_as_no_alerts(results_dir: Path) -> None:
    app.dependency_overrides[get_repository] = lambda: FakeRepository(with_alerts=False)
    try:
        body = AlertsResponse.model_validate(TestClient(app).get("/api/alerts").json())
    finally:
        app.dependency_overrides.clear()
    assert body.alert_run == "NO_ALERT_RUN" and body.alerts == [] and body.suppressed is None


def test_alert_detail(client: TestClient) -> None:
    a = AlertDetail.model_validate(client.get(f"/api/alerts/{ALERT_ID}").json())
    assert a.attribution[0].feature == "antecedent_dry_days"
    assert a.basis and a.guardrails and a.exposure and a.withheld_exceedance_prob is None
    ie = AlertDetail.model_validate(client.get(f"/api/alerts/{IE_ID}").json())
    assert ie.exceedance_prob is None and ie.withheld_exceedance_prob == 0.7
    assert client.get(f"/api/alerts/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/alerts/not-a-uuid").status_code == 422


def test_insufficient_evidence_with_a_probability_breaks_the_contract() -> None:
    with pytest.raises(ValueError):
        AlertDetail.model_validate(
            {**_alert(IE_ID, "INSUFFICIENT_EVIDENCE", 0.4, "x"), "attribution": []}
        )


def test_fhir_bundle(client: TestClient) -> None:
    r = client.get(f"/api/export/fhir/{ALERT_ID}")
    b = FhirBundle.model_validate(r.json())
    types = [e["resource"]["resourceType"] for e in b.entry]
    assert types == ["Observation", "Location", "Device"]
    obs, loc, dev = (e["resource"] for e in b.entry)
    assert obs["valueQuantity"]["value"] == 0.78
    assert obs["subject"]["reference"] == f"Location/{loc['id']}"
    assert obs["device"]["reference"] == f"Device/{dev['id']}"
    assert loc["position"] == {"longitude": -8.435, "latitude": 40.215}
    assert all(e["fullUrl"] == f"urn:uuid:{e['resource']['id']}" for e in b.entry)
    assert "Patient" not in json.dumps(r.json())
    # deterministic ids: the same alert exports to the same bundle
    assert client.get(f"/api/export/fhir/{ALERT_ID}").json() == r.json()


def test_fhir_insufficient_evidence_has_no_value(client: TestClient) -> None:
    obs = client.get(f"/api/export/fhir/{IE_ID}").json()["entry"][0]["resource"]
    assert "valueQuantity" not in obs
    assert obs["dataAbsentReason"]["text"].startswith("INSUFFICIENT_EVIDENCE")
    assert client.get(f"/api/export/fhir/{uuid.uuid4()}").status_code == 404


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
SCENARIO = {
    "name": "paving + sweeping",
    "interventions": [
        {"type": "permeable_paving", "reach_ids": [OBS_REACH, DRIVER_REACH], "extent": 5.0},
        {"type": "street_sweeping", "reach_ids": [OBS_REACH], "extent": 1.0},
    ],
}


def test_post_and_get_scenario(client: TestClient, repo: FakeRepository) -> None:
    r = client.post("/api/scenarios", json=SCENARIO)
    assert r.status_code == 201, r.text
    body = ScenarioResponse.model_validate(r.json())
    res = body.result
    assert body.city == "coimbra" and res.label == "planning estimate"
    assert res.caveat and res.citations and res.variant == "B"
    ok = [x for x in res.reaches if x.status is Status.OK]
    assert ok and all(x.scenario.interval.width > x.baseline.interval.width for x in ok)
    drv = [x for x in res.reaches if x.reach_id == DRIVER_REACH]
    assert all(x.status is Status.INSUFFICIENT_EVIDENCE for x in drv)
    assert "predict" not in json.dumps(r.json()["result"]).lower()
    got = client.get(f"/api/scenarios/{body.scenario_id}")
    assert got.status_code == 200 and got.json() == r.json()


@pytest.mark.parametrize(
    ("body", "status", "needle"),
    [
        (
            {
                "interventions": [
                    {"type": "permeable_paving", "reach_ids": [OBS_REACH], "magnitude": 0.5}
                ]
            },
            422,
            "extent",
        ),
        ({"interventions": []}, 422, ""),
        (
            {"interventions": [{"type": "daylighting", "reach_ids": [OBS_REACH]}]},
            422,
            "no cited coefficient",
        ),
        (
            {"interventions": [{"type": "street_sweeping", "reach_ids": [OBS_REACH], "extent": 3}]},
            422,
            "(0, 1]",
        ),
        (
            {
                "interventions": [
                    {"type": "street_sweeping", "reach_ids": ["CMB-0999"], "extent": 1}
                ]
            },
            404,
            "CMB-0999",
        ),
        (
            {
                "interventions": [
                    {"type": "street_sweeping", "reach_ids": ["XYZ-0001"], "extent": 1}
                ]
            },
            404,
            "XYZ-0001",
        ),
        (
            {
                "interventions": [
                    {"type": "street_sweeping", "reach_ids": [OBS_REACH, "PUN-0001"], "extent": 1}
                ]
            },
            422,
            "one city",
        ),
        (
            {
                "interventions": [
                    {
                        "type": "street_sweeping",
                        "reach_ids": [f"CMB-{i:04d}" for i in range(101)],
                        "extent": 1,
                    }
                ]
            },
            422,
            "max 100",
        ),
    ],
)
def test_scenario_request_errors(
    client: TestClient, body: dict[str, Any], status: int, needle: str
) -> None:
    r = client.post("/api/scenarios", json=body)
    assert r.status_code == status, r.text
    assert needle in r.text


def test_scenario_engine_unavailable_is_503(client: TestClient) -> None:
    def broken(city: str) -> Any:
        raise MissingResponseCheck("no response check on record - run `make response-check`")

    app.dependency_overrides[get_scenario_runner] = lambda: broken
    r = client.post("/api/scenarios", json=SCENARIO)
    assert r.status_code == 503 and "response-check" in r.text


def test_intervention_catalogue_carries_every_citation(client: TestClient) -> None:
    body = InterventionCatalogue.model_validate(
        client.get("/api/scenarios/interventions?city=coimbra").json()
    )
    table = load_intervention_coefficients()
    assert [i.id for i in body.interventions] == [c.id for c in table.interventions]
    for i in body.interventions:
        assert i.citations, i.id  # an uncited lever is never offered
        assert set(i.paths) == {"turbidity_proxy", "ndci"}
        assert all(p.reason for p in i.paths.values())
        if i.operation == "floor":
            assert i.extent_unit is None
    assert {n.id for n in body.not_quantified} == {n.id for n in table.not_quantified}
    assert body.caveat
    assert client.get("/api/scenarios/interventions?city=atlantis").status_code == 404


def test_unknown_scenario(client: TestClient) -> None:
    assert client.get(f"/api/scenarios/{uuid.uuid4()}").status_code == 404


def test_stored_scenario_that_breaks_the_contract_is_not_served(
    client: TestClient, repo: FakeRepository
) -> None:
    sid = ScenarioResponse.model_validate(
        client.post("/api/scenarios", json=SCENARIO).json()
    ).scenario_id
    repo.scenarios[sid]["result"]["citations"] = []
    with pytest.raises(ValueError):
        client.get(f"/api/scenarios/{sid}")


# ---------------------------------------------------------------------------
# priorities + validation
# ---------------------------------------------------------------------------
def test_priorities(client: TestClient) -> None:
    body = PrioritiesResponse.model_validate(
        client.get("/api/priorities", params={"city": "coimbra"}).json()
    )
    assert [r.reach_id for r in body.ranked] == [OBS_REACH]
    assert body.ranked[0].rank == 1 and body.ranked[0].information_value >= 0
    assert [(n.reach_id, n.reason) for n in body.not_ranked] == [(DRIVER_REACH, "NO_THRESHOLD")]
    assert body.weights["school"] == 3 and "planning choice" in body.weights_note
    assert client.get("/api/priorities", params={"city": "atlantis"}).status_code == 404


def test_validation_metrics_are_served_live_from_the_file(
    client: TestClient, results_dir: Path
) -> None:
    first = ValidationMetricsResponse.model_validate(client.get("/api/validation/metrics").json())
    assert first.metrics.losses and first.metrics.not_computed  # losses are served, not hidden
    assert first.metrics.model_extra["anomaly"] == {"scored_against": "PROXY"}
    assert first.scenario_response_check is None
    (results_dir / "metrics.json").write_text(
        json.dumps({**METRICS, "generated_at": "2026-09-24T00:00:00+00:00"}), encoding="utf-8"
    )
    (results_dir / "scenario_response_check_coimbra.json").write_text(
        json.dumps({"city": "coimbra", "checks": []}), encoding="utf-8"
    )
    second = ValidationMetricsResponse.model_validate(client.get("/api/validation/metrics").json())
    assert second.metrics.generated_at == "2026-09-24T00:00:00+00:00"
    assert second.source.sha256 != first.source.sha256
    assert second.scenario_response_check == {"city": "coimbra", "checks": []}


def test_validation_metrics_missing_or_broken(client: TestClient, results_dir: Path) -> None:
    (results_dir / "metrics.json").write_text(json.dumps({"city": "coimbra"}), encoding="utf-8")
    assert client.get("/api/validation/metrics").status_code == 500
    (results_dir / "metrics.json").unlink()
    r = client.get("/api/validation/metrics")
    assert r.status_code == 503 and "make evaluate" in r.text


def test_real_metrics_file_satisfies_the_schema() -> None:
    from core.settings import REPO_ROOT

    path = REPO_ROOT / "results" / "metrics.json"
    if not path.exists():
        pytest.skip("results/metrics.json not built")
    from api.schemas import MetricsDocument

    MetricsDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
