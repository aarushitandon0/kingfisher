"""The API against the real PostGIS database - the SQL in api/repository.py, end to end.

Marked `db`; skipped when the database is not reachable (`make db-up`). Needs the Coimbra
pipeline built (reaches, observations, exposure, production forecasts: `make
forecasts-latest`). Rows it writes - one synthetic alert pair and one scenario - are marked
(model_version / name TEST_FIXTURE) and deleted afterwards, pass or fail.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, timedelta
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.main import app
from api.routes.export import FhirBundle
from api.schemas import (
    AlertDetail,
    AlertsResponse,
    CitiesResponse,
    ForecastResponse,
    PrioritiesResponse,
    ReachCollection,
    ReachDetail,
    ScenarioResponse,
)

TEST_FIXTURE = "TEST_FIXTURE"


def _db_up() -> bool:
    try:
        from core.db import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1 FROM reaches LIMIT 1"))
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not _db_up(), reason="PostGIS not reachable (make db-up)"),
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def observable_reach() -> str:
    from core.db import session_scope

    with session_scope() as s:
        rid = s.execute(
            text(
                "SELECT r.reach_id FROM reaches r JOIN forecasts f USING (reach_id) "
                "WHERE r.observable AND f.fit = 'production' ORDER BY r.reach_id LIMIT 1"
            )
        ).scalar()
    if rid is None:
        pytest.skip("no production forecasts in the DB - run `make forecasts-latest`")
    return str(rid)


@pytest.fixture
def alerts(observable_reach: str) -> Iterator[tuple[UUID, UUID]]:
    """An ALERT and an INSUFFICIENT_EVIDENCE row written the way to_record shapes them."""
    from core.db import session_scope

    issued = date(2099, 1, 1)
    window = f"[{issued + timedelta(days=1)},{issued + timedelta(days=3)}]"
    ids = []
    try:
        with session_scope() as s:
            for severity, prob, withheld, reason in (
                ("ALERT", 0.81, None, None),
                ("INSUFFICIENT_EVIDENCE", None, 0.66, "STALE_EVIDENCE: test fixture"),
            ):
                ids.append(
                    s.execute(
                        text(
                            "INSERT INTO alerts (reach_id, issued_date, target_window, variable, "
                            "exceedance_prob, threshold_value, severity, attribution, exposure, "
                            "suppressed_reason, model_version, basis, guardrails, "
                            "withheld_exceedance_prob, threshold_derivation) VALUES "
                            "(:r, :d, CAST(:w AS daterange), 'turbidity_proxy', :p, 12.0, :sev, "
                            "CAST(:attr AS jsonb), CAST(:exp AS jsonb), :why, :mv, "
                            "CAST(:basis AS jsonb), CAST(:g AS jsonb), :wh, 'fixture') "
                            "RETURNING alert_id"
                        ),
                        {
                            "r": observable_reach,
                            "d": issued,
                            "w": window,
                            "p": prob,
                            "sev": severity,
                            "attr": json.dumps(
                                [
                                    {
                                        "feature": "api_7",
                                        "contribution": 0.2,
                                        "value": 3.0,
                                        "kind": "weather_driver",
                                    }
                                ]
                            ),
                            "exp": json.dumps({"buffer_m": 250.0, "features": {}}),
                            "why": reason,
                            "mv": TEST_FIXTURE,
                            "basis": json.dumps({"usable_observations_30d": 3}),
                            "g": json.dumps({"staleness": "pass"}),
                            "wh": withheld,
                        },
                    ).scalar_one()
                )
        yield ids[0], ids[1]
    finally:
        with session_scope() as s:
            s.execute(text("DELETE FROM alerts WHERE model_version = :mv"), {"mv": TEST_FIXTURE})


def test_cities_and_reaches(client: TestClient) -> None:
    cities = CitiesResponse.model_validate(client.get("/api/cities").json())
    coimbra = next(c for c in cities.cities if c.city == "coimbra")
    assert coimbra.status == "READY" and coimbra.reaches > 0
    fc = ReachCollection.model_validate(client.get("/api/reaches?city=coimbra").json())
    assert len(fc.features) == coimbra.reaches
    for f in fc.features:
        p = f.properties
        if p.observability == "DRIVER_ONLY":
            assert p.p_exceed_max is None and p.exceedance_status != "OK"
        if p.p_exceed_max is not None:
            assert 0.0 <= p.p_exceed_max <= 1.0


def test_reach_detail_and_forecast(client: TestClient, observable_reach: str) -> None:
    d = ReachDetail.model_validate(client.get(f"/api/reaches/{observable_reach}").json())
    assert d.catchment.geometry is not None and d.properties.history
    f = ForecastResponse.model_validate(
        client.get(f"/api/reaches/{observable_reach}/forecast?horizon=10").json()
    )
    assert f.variant == "A" and all(len(s.days) == 10 for s in f.series)


def test_alerts_detail_and_fhir(client: TestClient, alerts: tuple[UUID, UUID]) -> None:
    alert_id, ie_id = alerts
    body = AlertsResponse.model_validate(client.get("/api/alerts?city=coimbra").json())
    assert body.alert_run_issued_date == date(2099, 1, 1)
    assert [a.severity for a in body.alerts] == ["ALERT", "INSUFFICIENT_EVIDENCE"]
    a = AlertDetail.model_validate(client.get(f"/api/alerts/{alert_id}").json())
    # a closed daterange [d+1, d+3] is stored as [d+1, d+4) and read back inclusive
    assert (a.window_start, a.window_end) == (date(2099, 1, 2), date(2099, 1, 4))
    assert a.attribution[0].feature == "api_7" and a.guardrails == {"staleness": "pass"}
    ie = AlertDetail.model_validate(client.get(f"/api/alerts/{ie_id}").json())
    assert ie.exceedance_prob is None and ie.withheld_exceedance_prob == 0.66
    b = FhirBundle.model_validate(client.get(f"/api/export/fhir/{alert_id}").json())
    loc = b.entry[1]["resource"]
    assert {"longitude", "latitude"} <= set(loc["position"])


def test_insufficient_evidence_with_probability_is_refused_by_the_table(
    observable_reach: str,
) -> None:
    from sqlalchemy.exc import IntegrityError

    from core.db import session_scope

    with pytest.raises(IntegrityError), session_scope() as s:
        s.execute(
            text(
                "INSERT INTO alerts (reach_id, issued_date, target_window, variable, "
                "exceedance_prob, severity, suppressed_reason, model_version) VALUES "
                "(:r, '2099-01-01', '[2099-01-02,2099-01-03]', 'ndci', 0.5, "
                "'INSUFFICIENT_EVIDENCE', 'x', :mv)"
            ),
            {"r": observable_reach, "mv": TEST_FIXTURE},
        )


def test_priorities(client: TestClient) -> None:
    body = PrioritiesResponse.model_validate(client.get("/api/priorities?city=coimbra").json())
    values = [r.information_value for r in body.ranked]
    assert values == sorted(values, reverse=True)
    assert all(n.reason in ("NO_THRESHOLD", "NO_FORECAST", "NO_EXPOSURE") for n in body.not_ranked)


def test_scenario_round_trip_on_the_real_model(client: TestClient, observable_reach: str) -> None:
    from core.db import session_scope
    from models.scenario_model import check_path

    if not check_path("coimbra").exists():
        pytest.skip("run `make response-check` first")
    body = {
        "name": TEST_FIXTURE,
        "variables": ["turbidity_proxy"],
        "interventions": [
            {"type": "street_sweeping", "reach_ids": [observable_reach], "extent": 1.0},
            {"type": "permeable_paving", "reach_ids": [observable_reach], "extent": 1.0},
        ],
    }
    try:
        r = client.post("/api/scenarios", json=body)
        assert r.status_code == 201, r.text
        res = ScenarioResponse.model_validate(r.json())
        assert res.result.variant == "B"
        paths = {lv.intervention_id: lv.path for lv in res.result.levers}
        assert paths["street_sweeping"] == "MODEL_PERTURBATION"  # driver lever
        again = client.get(f"/api/scenarios/{res.scenario_id}")
        assert again.status_code == 200 and again.json() == r.json()
        with session_scope() as s:
            rows = s.execute(
                text(
                    "SELECT status, ci_high - ci_low AS w, baseline_ci_high - baseline_ci_low AS "
                    "bw, citations FROM scenario_results WHERE scenario_id = :id"
                ),
                {"id": res.scenario_id},
            ).fetchall()
        assert rows
        for row in rows:
            if row.status == "OK":
                assert row.w > row.bw and row.citations
    finally:
        with session_scope() as s:
            s.execute(text("DELETE FROM scenarios WHERE name = :n"), {"n": TEST_FIXTURE})
