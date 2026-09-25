"""Static, server-less build of the demo -> dist/snapshot-site/ (make snapshot).

The hosted demo has no server: the frontend is built in snapshot mode
(`npm run build:snapshot`, VITE_SNAPSHOT=1) and reads every API response from a JSON file
saved here from the REAL API. Nothing is recomputed or composed: each file is the API's own
response body, byte for byte.

Scenario runs cannot be enumerated, so a fixed set is run through the real engine now and
listed in snapshot/api/scenarios/index.json: every lever alone at the workbench's default
extent, and all levers together, on the city's top-priority observable reaches; plus, per
lever, the observable reaches where the engine returns an estimate (estimable_presets). The
workbench says it is a snapshot and offers only these.

    docker run -d --name kf-space --user 1000 -p 7860:7860 kingfisher-space   (make space first)
    python scripts/build_snapshot.py [--api http://127.0.0.1:7860]

Point --api at the deploy container, not the dev API: scenario runs are written to the
database of whatever serves them. Fails loudly on any non-200 response.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "snapshot-site"
SNAP = OUT / "snapshot"
PRESET_REACHES = 25
MIN_SHOWCASE_DAYS = 0.5  # exceedance days/yr; below this a lever gets no extra preset
HISTORY_DAYS = 3650  # the frontend's default in api.reach()

SPACE_README = """---
title: Kingfisher
emoji: 🐦
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
short_description: Early warning and resilience planning for urban streams
---

Kingfisher - hosted snapshot of the production system (Coimbra and Pune). Every number was
produced by the real API and models; scenario runs are the precomputed set listed in the
workbench. Planning estimates, not predictions; no health outcome is predicted.
"""


def snapshot_file(path: str) -> Path:
    """Must match snapshotFile() in frontend/src/api/client.ts."""
    p, _, query = path.partition("?")
    return SNAP / f"{p.lstrip('/')}{'~' + query if query else ''}.json"


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.saved = 0

    def call(self, path: str, body: dict[str, Any] | None = None) -> bytes:
        data = json.dumps(body).encode() if body is not None else None
        req = Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=600) as r:
                return r.read()
        except HTTPError as e:
            raise RuntimeError(f"{path}: HTTP {e.code} {e.read()[:300]!r}") from None

    def save(self, path: str) -> Any:
        raw = self.call(path)
        f = snapshot_file(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(raw)
        self.saved += 1
        return json.loads(raw)

    def save_many(self, paths: list[str], label: str) -> None:
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(self.save, paths))
        print(f"  {label}: {len(paths)} files in {time.monotonic() - t0:.0f} s")


def default_extent(i: dict[str, Any]) -> float | None:
    """Must match defaultExtent() in frontend/src/views/ScenarioView.tsx."""
    if i["operation"] == "floor":
        return None
    if i["operation"] == "scale_down":
        return 1
    return 5


def presets(api: Api, city: str, catalogue: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = api.save(f"/api/priorities?city={city}")["ranked"]
    reach_ids = [r["reach_id"] for r in ranked[:PRESET_REACHES]]
    if not reach_ids:
        raise RuntimeError(f"{city}: no ranked reaches to run preset scenarios on")
    levers = catalogue["interventions"]
    combos = [[i] for i in levers] + ([levers] if len(levers) > 1 else [])
    out = []
    for n, combo in enumerate(combos):
        interventions = [{"type": i["id"], "reach_ids": reach_ids, "extent": default_extent(i)} for i in combo]
        label = "All levers" if len(combo) > 1 else combo[0]["name"]
        name = f"{label} · top {len(reach_ids)} priority reaches"
        pid = f"{city}-{n:02d}"
        t0 = time.monotonic()
        raw = api.call("/api/scenarios", {"name": name, "interventions": interventions})
        file = f"/api/scenarios/{pid}"
        snapshot_file(file).parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(file).write_bytes(raw)
        print(f"  scenario {pid}: {name} ({time.monotonic() - t0:.1f} s)")
        out.append({"id": pid, "city": city, "name": name, "interventions": interventions, "file": file})
    return out + estimable_presets(api, city, catalogue, start=len(combos))


def estimable_presets(api: Api, city: str, catalogue: dict[str, Any], start: int) -> list[dict[str, Any]]:
    """One preset per lever on the observable reaches where the engine returns an estimate.

    The top-priority presets can come back all INSUFFICIENT_EVIDENCE / NOT_ESTIMABLE (e.g.
    Pune: no monsoon-season threshold on most reaches). This probes each lever over every
    observable reach and keeps the reaches whose result is OK for any variable - chosen by the
    engine's own status, never by the size or sign of the change, so reaches that get worse
    stay in. A lever is offered only if its estimate moves at least MIN_SHOWCASE_DAYS on some
    reach; otherwise the top-priority preset already shows it."""
    geo = json.loads(snapshot_file(f"/api/reaches?city={city}").read_bytes())["features"]
    observable = [f["properties"]["reach_id"] for f in geo if f["properties"].get("observable")][:100]
    out = []
    for i in catalogue["interventions"]:
        req = [{"type": i["id"], "reach_ids": observable, "extent": default_extent(i)}]
        probe = json.loads(api.call("/api/scenarios", {"name": "snapshot probe", "interventions": req}))["result"]
        ok = [r for r in probe["reaches"] if r["status"] == "OK"]
        if not any(abs(r["delta_days"] or 0) >= MIN_SHOWCASE_DAYS for r in ok):
            print(f"  estimable {i['id']}: {len(ok)} OK reach-variables, no change >= {MIN_SHOWCASE_DAYS} d - skipped")
            continue
        reach_ids = sorted({r["reach_id"] for r in ok})
        interventions = [{"type": i["id"], "reach_ids": reach_ids, "extent": default_extent(i)}]
        name = f"{i['name']} · {len(reach_ids)} reaches the engine can estimate"
        pid = f"{city}-{start + len(out):02d}"
        raw = api.call("/api/scenarios", {"name": name, "interventions": interventions})
        file = f"/api/scenarios/{pid}"
        snapshot_file(file).write_bytes(raw)
        print(f"  scenario {pid}: {name}")
        out.append({"id": pid, "city": city, "name": name, "interventions": interventions, "file": file})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", default="http://127.0.0.1:7860", help="the deploy container")
    args = ap.parse_args()
    api = Api(args.api)

    if OUT.exists():
        shutil.rmtree(OUT)
    print("frontend (snapshot mode)")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    subprocess.run([npm, "run", "build:snapshot"], cwd=ROOT / "frontend", check=True)
    shutil.copytree(ROOT / "frontend" / "dist-snapshot", OUT)
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")

    cities = [c["city"] for c in api.save("/api/cities")["cities"]]
    all_presets: list[dict[str, Any]] = []
    for city in cities:
        print(city)
        api.save_many(
            [f"/api/reaches?city={city}", f"/api/exposure?city={city}", f"/api/timeline?city={city}",
             f"/api/validation/metrics?city={city}"],
            "city layers",
        )
        catalogue = api.save(f"/api/scenarios/interventions?city={city}")
        reaches = [f["properties"]["reach_id"] for f in json.loads(snapshot_file(f"/api/reaches?city={city}").read_bytes())["features"]]
        api.save_many(
            [p for r in reaches for p in (f"/api/reaches/{r}?history_days={HISTORY_DAYS}", f"/api/reaches/{r}/forecast",
                                          f"/api/reaches/{r}/catchment", f"/api/reaches/{r}/attribution")],
            f"{len(reaches)} reaches",
        )
        alerts = [a["alert_id"] for a in api.save(f"/api/alerts?city={city}&active=true")["alerts"]]
        api.save_many([p for a in alerts for p in (f"/api/alerts/{a}", f"/api/export/fhir/{a}")], f"{len(alerts)} alerts")
        all_presets += presets(api, city, catalogue)

    index = snapshot_file("/api/scenarios/index")
    index.write_text(json.dumps(all_presets, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"snapshot ready: {OUT} ({api.saved} API files, {len(all_presets)} scenarios, {size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
