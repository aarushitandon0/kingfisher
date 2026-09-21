"""Exposure layer I/O: OSM + GHS-POP -> engine.exposure -> exposure_features.

Run:

    python -m pipeline.exposure_features --city coimbra

All logic is in engine/exposure.py (pure); this module only loads the inputs already on
disk (the Overpass extract and the GHS-POP tile listed in DATA_INVENTORY.md) and writes
the rows. Nothing is fetched from the network here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.ops import transform as shapely_transform

from core.logging import get_logger, stage
from core.settings import DATA_DIR
from engine.exposure import (
    DEFAULT_BUFFER_M,
    PopulationGrid,
    compute_exposure,
    osm_exposure_features,
)
from pipeline.l0_network import load_city_config

log = get_logger(__name__)

OSM_PATH = {"coimbra": DATA_DIR / "raw" / "osm" / "coimbra_overpass.json"}
GHS_ZIP = {
    "coimbra": DATA_DIR
    / "raw"
    / "population"
    / "GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R5_C18.zip"
}
GHS_SOURCE = "GHS-POP E2020 R2023A 3ss (JRC, CC-BY-4.0)"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# The Day-0 Overpass extract did not ask for water access points, so every reach would
# read "0 water access points" - an artefact of the query, not a fact about the river.
# This supplementary query is fetched once, cached, and merged with the extract.
WATER_ACCESS_QUERY = """
[out:json][timeout:120];
(
  node["ford"]({s},{w},{n},{e});
  way["ford"]({s},{w},{n},{e});
  node["leisure"~"^(slipway|swimming_area)$"]({s},{w},{n},{e});
  way["leisure"~"^(slipway|swimming_area)$"]({s},{w},{n},{e});
  node["waterway"="access_point"]({s},{w},{n},{e});
);
out body geom;
"""


def load_water_access(bbox: dict[str, float]) -> list[dict[str, Any]]:
    import requests

    from core.cache import DiskCache

    query = WATER_ACCESS_QUERY.format(
        s=bbox["min_lat"], w=bbox["min_lon"], n=bbox["max_lat"], e=bbox["max_lon"]
    )

    def fetch() -> dict[str, Any]:
        r = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "kingfisher-hackathon/0.1 (urban stream monitoring)"},
            timeout=180,
        )
        r.raise_for_status()
        payload: dict[str, Any] = r.json()
        if "elements" not in payload:
            raise RuntimeError(f"Overpass returned no `elements`: {str(payload)[:300]}")
        return payload

    entry = DiskCache("osm").get_or_fetch(
        {"query": query},
        fetch,
        slug="water_access",
        url=OVERPASS_URL,
        source="OpenStreetMap via Overpass (ODbL)",
    )
    elements: list[dict[str, Any]] = entry.payload["elements"]
    return elements


def load_population(city: str, bbox: dict[str, float], pad_deg: float = 0.05) -> PopulationGrid:
    import rasterio
    from rasterio.windows import from_bounds

    path = GHS_ZIP[city]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/fetch_datasets.py")
    tif = f"/vsizip/{path.as_posix()}/{path.stem}.tif"
    with rasterio.open(tif) as src:
        window = (
            from_bounds(
                bbox["min_lon"] - pad_deg,
                bbox["min_lat"] - pad_deg,
                bbox["max_lon"] + pad_deg,
                bbox["max_lat"] + pad_deg,
                transform=src.transform,
            )
            .round_offsets()
            .round_lengths()
        )
        values = src.read(1, window=window).astype("float64")
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
        t = src.window_transform(window)
        return PopulationGrid(
            values=values,
            transform=(t.a, t.b, t.c, t.d, t.e, t.f),
            crs=str(src.crs),
            source=GHS_SOURCE,
        )


def load_reaches(city: str) -> pd.DataFrame:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        rows = session.execute(
            text("SELECT reach_id, ST_AsText(geom) AS wkt FROM reaches WHERE city = :c"),
            {"c": city},
        ).all()
    if not rows:
        raise RuntimeError(f"No reaches for {city} - run L0 first")
    return pd.DataFrame(
        {"reach_id": [r[0] for r in rows], "geometry": [shapely_wkt.loads(r[1]) for r in rows]}
    )


def write(rows: pd.DataFrame, reach_ids: list[str], to_wgs: Any) -> int:
    from sqlalchemy import text

    from core.db import session_scope

    records = [
        {
            "reach_id": r.reach_id,
            "feature_type": r.feature_type,
            "count": None if pd.isna(r.count) else int(r.count),
            "nearest_distance_m": None
            if r.nearest_distance_m is None or pd.isna(r.nearest_distance_m)
            else float(r.nearest_distance_m),
            "geom": None if r.geom is None else shapely_transform(to_wgs, r.geom).wkt,
            "buffer_m": float(r.buffer_m),
            "source": r.source,
        }
        for r in rows.itertuples(index=False)
    ]
    with session_scope() as session:
        session.execute(
            text("DELETE FROM exposure_features WHERE reach_id = ANY(:ids)"), {"ids": reach_ids}
        )
        session.execute(
            text(
                "INSERT INTO exposure_features "
                "(reach_id, feature_type, count, nearest_distance_m, geom, buffer_m, source) "
                "VALUES (:reach_id, :feature_type, :count, :nearest_distance_m, "
                "ST_GeomFromText(:geom, 4326), :buffer_m, :source)"
            ),
            records,
        )
    return len(records)


def build(city: str, *, write_db: bool = True) -> dict[str, Any]:
    cfg = load_city_config(city)
    metric = cfg["crs"]["metric"]
    buffer_m = float(cfg.get("exposure", {}).get("buffer_m", DEFAULT_BUFFER_M))
    to_metric = Transformer.from_crs("EPSG:4326", metric, always_xy=True).transform
    to_wgs = Transformer.from_crs(metric, "EPSG:4326", always_xy=True).transform

    with stage(log, "exposure", city=city) as counters:
        osm = json.loads(OSM_PATH[city].read_text(encoding="utf-8"))
        extra = load_water_access(cfg["bbox"])
        seen = {(e["type"], e["id"]) for e in osm["elements"]}
        elements = osm["elements"] + [e for e in extra if (e["type"], e["id"]) not in seen]
        feats = osm_exposure_features(elements)
        if feats.empty:
            raise RuntimeError("no exposure features in the OSM extract - refusing to write zeros")
        if feats.attrs["dropped_no_geometry"]:
            counters.drop(feats.attrs["dropped_no_geometry"], "NO_GEOMETRY")
        feats["geometry"] = [shapely_transform(to_metric, g) for g in feats["geometry"]]
        reaches = load_reaches(city)
        reaches["geometry"] = [shapely_transform(to_metric, g) for g in reaches["geometry"]]
        grid = load_population(city, cfg["bbox"])
        rows = compute_exposure(
            reaches,
            feats,
            buffer_m=buffer_m,
            population=grid,
            to_population_crs=Transformer.from_crs(metric, grid.crs, always_xy=True).transform,
        )
        pop = rows[rows["feature_type"] == "population"]
        counters.record(
            rows_in=len(elements),
            rows_out=len(rows),
            reaches=len(reaches),
            osm_features=dict(Counter(feats["feature_type"])),
            buffer_m=buffer_m,
            population_null=int(pop["count"].isna().sum()),
        )
        if write_db:
            write(rows, list(reaches["reach_id"]), to_wgs)
    by_type = (
        rows[rows["feature_type"] != "population"]
        .groupby("feature_type")
        .agg(reaches_with_any=("count", lambda s: int((s > 0).sum())), total=("count", "sum"))
    )
    return {
        "reaches": len(reaches),
        "rows": len(rows),
        "buffer_m": buffer_m,
        "by_type": by_type.astype(int).to_dict("index"),
        "population_total_in_buffers": int(pop["count"].sum()),
        "population_null": int(pop["count"].isna().sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exposure features within a buffer of each reach")
    parser.add_argument("--city", default="coimbra")
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.city, write_db=not args.no_db), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
