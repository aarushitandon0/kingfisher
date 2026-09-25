"""Assemble the Hugging Face Space bundle for the live demo -> dist/space/.

    python scripts/build_space.py            (make space)

The bundle is deploy/ (Dockerfile, start.sh, requirements, Space README) plus:
  app/       the Python packages the API imports, config/, results/, the built frontend
             (app/web), the production models and the frames/SHAP the API reads
  db/        a pg_dump of the local PostGIS without drivers_daily (the API never reads
             it) and without forecasts, which ship separately as production rows only

Needs the local PostGIS running (docker compose up -d db) and frontend dependencies
installed. Fails loudly if anything the API needs is missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "space"
DB_CONTAINER = "kingfisher-db"
PSQL = ["docker", "exec", DB_CONTAINER]
CODE = ["api", "core", "engine", "models", "pipeline", "config", "results"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def run(cmd: list[str], **kw: object) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)  # type: ignore[call-overload]


def copy_code(app: Path) -> None:
    for name in CODE:
        shutil.copytree(ROOT / name, app / name, ignore=IGNORE)
    ref = ROOT / "data" / "reference"
    if ref.exists():
        shutil.copytree(ref, app / "data" / "reference", ignore=IGNORE)


def copy_processed(app: Path, cities: list[str]) -> None:
    dst = app / "data" / "processed"
    dst.mkdir(parents=True)
    for city in cities:
        for name in (f"frame_{city}.parquet", f"frame_{city}.meta.json", f"gbm_shap_{city}.parquet"):
            src = ROOT / "data" / "processed" / name
            if not src.exists():
                raise FileNotFoundError(f"{src} missing - the API needs it for {city}")
            shutil.copy2(src, dst / name)


def copy_models(app: Path) -> None:
    src = ROOT / "artifacts" / "models"
    dst = app / "artifacts" / "models"
    dst.mkdir(parents=True)
    pointers = sorted(src.glob("*.production.json"))
    if not pointers:
        raise FileNotFoundError(f"no production model pointers in {src}")
    for p in pointers:
        version = json.loads(p.read_text(encoding="utf-8"))["version"]
        shutil.copy2(p, dst / p.name)
        shutil.copytree(src / version, dst / version)


def build_frontend(app: Path) -> None:
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    run([npm, "run", "build"], cwd=ROOT / "frontend")
    shutil.copytree(ROOT / "frontend" / "dist", app / "web")


def dump_db(db: Path) -> None:
    db.mkdir(parents=True)
    run(PSQL + ["pg_dump", "-U", "kingfisher", "-d", "kingfisher", "-Fc", "-Z", "9",
                "--exclude-table-data=drivers_daily", "--exclude-table-data=forecasts",
                "-f", "/tmp/kingfisher.dump"])
    run(["docker", "cp", f"{DB_CONTAINER}:/tmp/kingfisher.dump", str(db / "kingfisher.dump")])
    copy = "\\copy (SELECT * FROM forecasts WHERE fit = 'production') TO '/tmp/fc.csv' WITH (FORMAT csv, HEADER true)"
    run(PSQL + ["psql", "-U", "kingfisher", "-d", "kingfisher", "-v", "ON_ERROR_STOP=1", "-c", copy])
    run(PSQL + ["gzip", "-f", "-9", "/tmp/fc.csv"])
    run(["docker", "cp", f"{DB_CONTAINER}:/tmp/fc.csv.gz", str(db / "forecasts_production.csv.gz")])
    run(PSQL + ["rm", "-f", "/tmp/kingfisher.dump", "/tmp/fc.csv.gz"])


def main() -> None:
    cities = sorted(p.stem for p in (ROOT / "config" / "cities").glob("*.yaml"))
    if OUT.exists():
        shutil.rmtree(OUT)
    app = OUT / "app"
    app.mkdir(parents=True)
    for name in ("Dockerfile", "start.sh", "requirements.txt", "README.md"):
        # LF only: a CRLF start.sh does not run on Linux (Windows checkouts can add CRLF).
        data = (ROOT / "deploy" / name).read_bytes().replace(b"
", b"
")
        (OUT / name).write_bytes(data)
    print("code + config + results"); copy_code(app)
    print(f"frames + SHAP for {cities}"); copy_processed(app, cities)
    print("production models"); copy_models(app)
    print("frontend"); build_frontend(app)
    print("database"); dump_db(OUT / "db")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"bundle ready: {OUT} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
