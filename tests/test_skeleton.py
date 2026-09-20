"""Skeleton checks. The real suite (guardrails, leakage, quality flags) lands with
the code it guards — see CLAUDE.md, Testing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.logging import get_logger, stage
from core.settings import CONFIG_DIR, REPO_ROOT


def test_repo_layout() -> None:
    for path in ["config", "data/reference", "pipeline", "models", "engine", "api", "migrations"]:
        assert (REPO_ROOT / path).is_dir(), f"missing {path}"


def test_coimbra_config_has_bbox_and_crs() -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "cities" / "coimbra.yaml").read_text(encoding="utf-8"))
    assert cfg["city"] == "coimbra"
    assert cfg["reach_id_prefix"] == "CMB"
    assert cfg["bbox"]["min_lon"] < cfg["bbox"]["max_lon"]
    assert cfg["bbox"]["min_lat"] < cfg["bbox"]["max_lat"]
    assert cfg["crs"]["storage"] == "EPSG:4326"
    assert cfg["crs"]["metric"].startswith("EPSG:")
    assert cfg["reaches"] == []


def test_no_uncited_intervention_coefficients() -> None:
    """Every merged coefficient carries a citation (CLAUDE.md #6). Vacuously true
    while the table is empty; binding from Day 6."""
    cfg = yaml.safe_load((CONFIG_DIR / "intervention_coefficients.yaml").read_text(encoding="utf-8"))
    required = set(cfg["schema"]["required_fields"])
    for entry in cfg["interventions"] or []:
        missing = required - set(entry)
        assert not missing, f"{entry.get('name', entry)} is missing {sorted(missing)}"


def test_gitignore_excludes_secrets_and_raw_data_but_keeps_reference() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in [".env", "data/raw/", "data/interim/", "data/processed/"]:
        assert pattern in lines, f"{pattern} must be gitignored"
    assert "data/reference/" not in lines
    assert Path(REPO_ROOT / "data" / "reference" / "incidents.csv").exists()


def test_stage_logger_records_drops_with_reasons() -> None:
    log = get_logger("test")
    with stage(log, "unit", city="coimbra") as counters:
        counters.record(rows_in=100, rows_out=80)
        counters.drop(15, "NO_WATER_PIXELS")
        counters.drop(5, "CLOUD")
    assert counters.rows_dropped == 20
    assert counters.drop_reasons == {"NO_WATER_PIXELS": 15, "CLOUD": 5}
