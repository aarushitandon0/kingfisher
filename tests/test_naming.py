"""The turbidity index is an uncalibrated optical index on L2A reflectance. It is shown as
"turbidity index" everywhere a person reads it, and never in a turbidity unit."""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.config import load_config
from core.settings import REPO_ROOT

UNIT = re.compile(r"\b(FNU|NTU|FTU)\b")
USER_FACING = [REPO_ROOT / "README.md", *(REPO_ROOT / "frontend").rglob("*.*")]


def test_display_name_is_turbidity_index() -> None:
    thr = load_config("thresholds")["variables"]["turbidity_proxy"]
    assert thr["display_name"] == "turbidity index"
    assert not UNIT.search(thr["units"])


def test_no_turbidity_unit_in_readme_or_frontend() -> None:
    hits = [
        str(p.relative_to(REPO_ROOT))
        for p in USER_FACING
        if p.is_file()
        and "node_modules" not in p.parts
        and UNIT.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not hits, f"turbidity unit in user-facing files: {hits}"


def test_metrics_label_the_variable_turbidity_index_and_carry_no_unit() -> None:
    path = REPO_ROOT / "results" / "metrics.json"
    if not path.exists():
        return  # nothing published yet; the evaluate code path is covered by its tests
    text = path.read_text(encoding="utf-8")
    assert not UNIT.search(text)
    assert json.loads(text)["variables"]["turbidity_proxy"]["display_name"] == "turbidity index"


def test_figures_are_titled_with_display_names() -> None:
    src = Path(REPO_ROOT / "models" / "evaluate.py").read_text(encoding="utf-8")
    assert 'f"{var} - ' not in src, "a figure title uses the raw column name"
