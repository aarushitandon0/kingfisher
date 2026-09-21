"""Zip exactly what the EA-LSTM training needs, for Google Colab.

    python scripts/make_colab_bundle.py          -> dist/kingfisher_colab.zip

Contents: the code the training imports (core/, models/, pipeline/, engine/), config/,
and the NeuralHydrology export (data/processed/nh/). Never .env, never data/raw/, never
credentials - the bundle is checked for them before it is written.

On Colab follow notebooks/ealstm_colab.ipynb. Bring back `ealstm_runs.zip` and unpack it
with `python scripts/make_colab_bundle.py --unpack path/to/ealstm_runs.zip`.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CODE_DIRS = ("core", "models", "pipeline", "engine", "config")
DATA_DIR = Path("data/processed/nh")
FORBIDDEN = (".env", "data/raw", ".pem", ".cdsapirc", "credentials")
OUT = REPO / "dist" / "kingfisher_colab.zip"


def files() -> list[Path]:
    out: list[Path] = []
    for d in CODE_DIRS:
        out += [
            p
            for p in (REPO / d).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        ]
    out += [p for p in (REPO / DATA_DIR).rglob("*") if p.is_file()]
    return sorted(out)


def build() -> Path:
    fs = files()
    rel = [p.relative_to(REPO).as_posix() for p in fs]
    leaked = [r for r in rel if any(f in r for f in FORBIDDEN)]
    if leaked:
        raise SystemExit(f"refusing to bundle secrets / raw data: {leaked}")
    manifest = REPO / DATA_DIR / "export_manifest.json"
    if not manifest.exists():
        raise SystemExit("no NH export - run `python -m pipeline.export_neuralhydrology` first")
    n_nc = sum(1 for r in rel if r.endswith(".nc"))
    if n_nc == 0:
        raise SystemExit("the NH export has no time series")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p, r in zip(fs, rel, strict=True):
            z.write(p, r)
    m = json.loads(manifest.read_text())
    print(
        f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB): {len(rel)} files, {n_nc} reach "
        f"series, {len(m['basin_lists']['train_reaches'])} training reaches"
    )
    return OUT


def unpack(zip_path: Path) -> None:
    """Unzip Colab's artifacts/ealstm into this repo and list what arrived."""
    dest = REPO / "artifacts" / "ealstm"
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        bad = [n for n in names if not n.startswith("artifacts/ealstm/") or ".." in n]
        if bad:
            raise SystemExit(f"unexpected paths in {zip_path}: {bad[:5]}")
        z.extractall(REPO)
    pointers = sorted(p.name for p in dest.glob("*__seed*.json"))
    print(f"unpacked {len(names)} files; {len(pointers)} trained members: {pointers}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unpack", type=Path, help="unzip Colab's ealstm_runs.zip into artifacts/")
    a = ap.parse_args()
    if a.unpack:
        unpack(a.unpack)
    else:
        build()


if __name__ == "__main__":
    main()
