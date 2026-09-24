"""FHIR export: alert -> Observation + Location + Device bundle (MASTERSPEC section 14).

A pure function. FHIR R4 `Bundle` of type `collection`:

  Location     the reach (name, midpoint position, reach_id identifier)
  Device       the model that produced the probability (model_version) - provenance
  Observation  the alert: exceedance probability of the variable above the reach's own
               seasonal threshold, the severity as interpretation, the threshold and window
               as components, exposure context as a note.

Codes: Kingfisher's own concepts use a local code system (LOCAL_SYSTEM, a URN) - nothing
here claims a LOINC or SNOMED code it has not been assigned. The Observation is about an
optical water-state index, not a person: no Patient, no health outcome, and no statement
that water is safe or unsafe (CLAUDE.md #8). An INSUFFICIENT_EVIDENCE alert carries no
value - it has a dataAbsentReason and the reason as text.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

LOCAL_SYSTEM = "urn:kingfisher:codes"
DATA_ABSENT_SYSTEM = "http://terminology.hl7.org/CodeSystem/data-absent-reason"
UCUM = "http://unitsofmeasure.org"
_NS = uuid.UUID("5b2f6c1e-7a0b-4f55-9a43-2f1b8d0f3c11")  # Kingfisher FHIR id namespace


def _id(kind: str, key: str) -> str:
    return str(uuid.uuid5(_NS, f"{kind}:{key}"))


def _iso(v: date | datetime | str | None) -> str | None:
    if v is None:
        return None
    return v.isoformat() if isinstance(v, date | datetime) else str(v)


def alert_to_fhir_bundle(
    alert: dict[str, Any], *, display_name: str | None = None
) -> dict[str, Any]:
    """`alert`: a repository alert row (api.repository.PostgresRepository.alert)."""
    alert_id = str(alert["alert_id"])
    reach_id = str(alert["reach_id"])
    variable = str(alert["variable"])
    label = display_name or variable
    severity = str(alert["severity"])
    loc_id = _id("Location", reach_id)
    dev_id = _id("Device", str(alert.get("model_version") or "unknown"))
    obs_id = _id("Observation", alert_id)

    location: dict[str, Any] = {
        "resourceType": "Location",
        "id": loc_id,
        "identifier": [{"system": f"{LOCAL_SYSTEM}:reach", "value": reach_id}],
        "name": alert.get("reach_name") or reach_id,
        "description": "Urban stream reach (Kingfisher reach network)",
        "mode": "instance",
        "physicalType": {"text": "stream reach"},
    }
    mid = alert.get("midpoint")
    if isinstance(mid, dict) and mid.get("type") == "Point":
        lon, lat = mid["coordinates"][:2]
        location["position"] = {"longitude": float(lon), "latitude": float(lat)}

    device = {
        "resourceType": "Device",
        "id": dev_id,
        "identifier": [
            {"system": f"{LOCAL_SYSTEM}:model-version", "value": alert.get("model_version")}
        ],
        "deviceName": [{"name": "Kingfisher forecast model", "type": "model-name"}],
        "version": [{"value": str(alert.get("model_version") or "unknown")}],
        "note": [
            {
                "text": "Statistical forecast model; exceedance probabilities are calibrated "
                "against the reach's own seasonal threshold (results/metrics.json)."
            }
        ],
    }

    observation: dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_id,
        "identifier": [{"system": f"{LOCAL_SYSTEM}:alert", "value": alert_id}],
        "status": "preliminary",
        "category": [
            {
                "coding": [
                    {"system": LOCAL_SYSTEM, "code": "environmental", "display": "Environmental"}
                ],
                "text": "environmental",
            }
        ],
        "code": {
            "coding": [
                {
                    "system": LOCAL_SYSTEM,
                    "code": f"exceedance-probability-{variable}",
                    "display": f"Probability that the {label} exceeds the reach seasonal threshold",
                }
            ],
            "text": f"P({label} > reach seasonal threshold)",
        },
        "subject": {"reference": f"Location/{loc_id}", "display": location["name"]},
        "effectivePeriod": {
            "start": _iso(alert.get("window_start")),
            "end": _iso(alert.get("window_end")),
        },
        "issued": f"{_iso(alert.get('issued_date'))}T00:00:00Z",
        "device": {"reference": f"Device/{dev_id}"},
        "interpretation": [
            {"coding": [{"system": f"{LOCAL_SYSTEM}:severity", "code": severity}], "text": severity}
        ],
        "component": [],
        "note": [
            {
                "text": "Exposure pathways only - no health outcome is predicted and no water "
                "is declared safe or unsafe."
            }
        ],
    }
    p = alert.get("exceedance_prob")
    if severity == "INSUFFICIENT_EVIDENCE" or p is None:
        observation["dataAbsentReason"] = {
            "coding": [{"system": DATA_ABSENT_SYSTEM, "code": "unknown"}],
            "text": f"INSUFFICIENT_EVIDENCE: {alert.get('suppressed_reason')}",
        }
    else:
        observation["valueQuantity"] = {
            "value": float(p),
            "unit": "probability",
            "system": UCUM,
            "code": "1",
        }
    thr = alert.get("threshold_value")
    if thr is not None:
        observation["component"].append(
            {
                "code": {
                    "coding": [{"system": LOCAL_SYSTEM, "code": "threshold"}],
                    "text": "threshold",
                },
                "valueQuantity": {
                    "value": float(thr),
                    "unit": "index",
                    "system": UCUM,
                    "code": "1",
                },
            }
        )
    if alert.get("threshold_derivation"):
        observation["note"].append({"text": f"Threshold: {alert['threshold_derivation']}"})
    if alert.get("suppressed_reason") and severity != "INSUFFICIENT_EVIDENCE":
        observation["note"].append({"text": f"Guardrail: {alert['suppressed_reason']}"})
    exposure = alert.get("exposure")
    if isinstance(exposure, dict) and exposure:
        feats = exposure.get("features") or {}
        parts = [f"{k}: {v.get('count')}" for k, v in sorted(feats.items()) if isinstance(v, dict)]
        observation["note"].append(
            {
                "text": f"Exposure within {exposure.get('buffer_m')} m - "
                + ", ".join(parts)
                + f"; population {exposure.get('population')}"
            }
        )
    if not observation["component"]:
        del observation["component"]

    return {
        "resourceType": "Bundle",
        "id": _id("Bundle", alert_id),
        "type": "collection",
        "timestamp": f"{_iso(alert.get('issued_date'))}T00:00:00Z",
        "entry": [
            {"fullUrl": f"urn:uuid:{r['id']}", "resource": r}
            for r in (observation, location, device)
        ],
    }
