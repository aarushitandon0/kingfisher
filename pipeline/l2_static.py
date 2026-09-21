"""L2 - static catchment attributes, aggregated over each reach's UPSTREAM CATCHMENT
polygon (never a buffer around the reach). Writes catchment_attributes. These are the
scenario levers (MASTERSPEC 6.3).

Run:

    python -m pipeline.l2_static --city coimbra
    python -m pipeline.l2_static --city coimbra --strict     # refuse any fallback

| attribute            | source                                                        |
|----------------------|---------------------------------------------------------------|
| imperviousness_pct   | land-cover adapter (CLMS IMD, nearest year / WorldCover proxy) |
| urban_fraction       | land-cover adapter (Urban Atlas or CORINE / WorldCover built-up)|
| riparian_width_m     | land-cover adapter (CLMS Riparian Zones / WorldCover corridor) |
| riparian_ndvi_mean   | Sentinel-2 L1 riparian NDVI (all adapters - see below)         |
| road_density_km_km2  | OSM highway=<road_classes>, Portugal/India pbf                 |
| alan_radiance        | VIIRS DNB monthly composites (EOG), nearest year               |
| population           | GHS-POP (GHSL), 3 arc-second                                   |

THE ADAPTER INTERFACE (Pune)
----------------------------
Everything land-cover-specific sits behind `LandCoverAdapter`: imperviousness_pct,
urban_fraction, riparian_width_m. `AdapterChain` implements the same interface and, per
attribute, asks each adapter in turn - the first that has the data answers, and the row
records which one it was. Coimbra's chain is [clms, worldcover]; Pune's is [worldcover].
Calling code only ever sees the interface, so adding Pune is a YAML line.

While the CLMS registration is pending, Coimbra's land-cover attributes come from ESA
WorldCover. That fallback is logged as a WARNING, written into every row's `sources`,
and flagged PROXY_* - never passed off as CLMS. `--strict` refuses it instead.

WHY riparian_ndvi_mean COMES FROM SENTINEL-2 FOR EVERY ADAPTER
------------------------------------------------------------
CLMS Riparian Zones delineates riparian zones and their land cover; it has no NDVI band,
and neither does WorldCover. NDVI is an optical measurement, and L1 already measures it
on every reach's 30 m riparian buffer. The catchment value is the length-weighted mean,
over the reaches inside the catchment, of each reach's climatological NDVI (mean of
monthly means, dates <= splits.train_end only so no test-year observation leaks into a
static feature).

NULLS
-----
A value that cannot be computed is NULL with a reason in `flags` - SOURCE_UNAVAILABLE
(dataset not on disk yet), NO_PIXELS (catchment smaller than a raster cell), PARTIAL_
COVERAGE (catchment runs off the raster), NO_S2_RIPARIAN_OBS. Never zero. A source that
IS on disk but returns nothing at all over the study area raises.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from core.cache import DiskCache, cache_key
from core.config import as_date, load_config
from core.logging import get_logger, stage
from core.settings import DATA_DIR
from pipeline.l0_network import bbox_tuple, load_city_config

log = get_logger(__name__)

RAW = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

ATTRIBUTES = [
    "imperviousness_pct",
    "riparian_ndvi_mean",
    "riparian_width_m",
    "road_density_km_km2",
    "alan_radiance",
    "population",
    "urban_fraction",
]

# ESA WorldCover v200 classes
WC_BUILT_UP = 50
WC_VEGETATED = (10, 20, 30, 90, 95, 100)  # tree, shrub, grass, wetland, mangrove, moss
WC_NODATA = 0


class SourceUnavailable(RuntimeError):
    """The dataset for an attribute is not on disk. Carries what to do about it."""


# ---------------------------------------------------------------------------
# values with provenance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AttributeValue:
    value: float | None
    source: str | None
    flag: str | None = None  # reason when value is None, or a caveat (PROXY_*) when not
    year: int | None = None

    @staticmethod
    def missing(flag: str, source: str | None = None) -> AttributeValue:
        return AttributeValue(None, source, flag)


@dataclass
class CatchmentContext:
    """Everything an adapter may need. Geometries are EPSG:4326."""

    catchments: dict[str, BaseGeometry]
    area_km2: dict[str, float | None]
    metric_crs: str
    reference_year: int
    network: list[LineString] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_metric(self, geom: BaseGeometry) -> BaseGeometry:
        return shapely_transform(_transformer("EPSG:4326", self.metric_crs).transform, geom)

    def to_wgs(self, geom: BaseGeometry) -> BaseGeometry:
        return shapely_transform(_transformer(self.metric_crs, "EPSG:4326").transform, geom)

    def to_metric_many(self, geoms: Sequence[BaseGeometry]) -> np.ndarray:
        """Vectorised reprojection of many geometries in one pyproj call."""
        t = _transformer("EPSG:4326", self.metric_crs)

        def fn(xy: np.ndarray) -> np.ndarray:
            x, y = t.transform(xy[:, 0], xy[:, 1])
            return np.column_stack([x, y])

        return np.asarray(shapely.transform(np.asarray(geoms, dtype=object), fn))


@lru_cache(maxsize=16)
def _transformer(src: str, dst: str) -> Transformer:
    # Building a Transformer costs milliseconds; per-geometry that was hours.
    return Transformer.from_crs(src, dst, always_xy=True)


# ---------------------------------------------------------------------------
# raster zonal statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZonalSample:
    values: np.ndarray  # valid pixel values inside the polygon
    n_inside: int  # pixels inside the polygon (valid + nodata)
    fully_covered: bool  # polygon lies within the raster footprint

    @property
    def n_nodata(self) -> int:
        return self.n_inside - len(self.values)


def zip_path(path: Path) -> str:
    """rasterio path for a GeoTIFF inside a .zip (GHS-POP ships zipped)."""
    import zipfile

    if path.suffix.lower() != ".zip":
        return str(path)
    with zipfile.ZipFile(path) as z:
        tifs = [n for n in z.namelist() if n.lower().endswith(".tif")]
    if not tifs:
        raise SourceUnavailable(f"{path} contains no GeoTIFF")
    return f"zip://{path.as_posix()}!/{tifs[0]}"


class RasterSource:
    """One or more GeoTIFF tiles (disjoint) sampled polygon-by-polygon.

    Tiles are opened once and kept open. Each polygon is reprojected into the raster CRS,
    read through a window around its bounds, optionally supersampled (for extensive
    quantities like population, so a catchment smaller than one cell still gets its
    share), and masked by pixel centre.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        nodata: float | None = None,
        negative_is_nodata: bool = False,
    ) -> None:
        import rasterio

        if not paths:
            raise SourceUnavailable("no raster files given")
        self.paths = list(paths)
        self.datasets = [rasterio.open(zip_path(p)) for p in self.paths]
        self.nodata = nodata
        self.negative_is_nodata = negative_is_nodata

    def close(self) -> None:
        for ds in self.datasets:
            ds.close()

    def __enter__(self) -> RasterSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def sample(
        self, geom: BaseGeometry, *, supersample: int = 1, all_touched: bool = False
    ) -> ZonalSample:
        from rasterio.features import geometry_mask
        from rasterio.transform import Affine
        from rasterio.windows import Window, from_bounds

        values: list[np.ndarray] = []
        n_inside = 0
        covered_area = 0.0
        total_area = 0.0
        for ds in self.datasets:
            fn = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform
            g = shapely_transform(fn, geom)
            total_area = g.area
            footprint = box(*ds.bounds)
            if not g.intersects(footprint):
                continue
            covered_area += g.intersection(footprint).area
            win = from_bounds(*g.bounds, transform=ds.transform)
            win = Window(
                math.floor(win.col_off) - 1,
                math.floor(win.row_off) - 1,
                math.ceil(win.width) + 3,
                math.ceil(win.height) + 3,
            ).intersection(Window(0, 0, ds.width, ds.height))
            data = ds.read(1, window=win).astype("float64")
            transform = ds.window_transform(win)
            nodata = self.nodata if self.nodata is not None else ds.nodata
            invalid = np.zeros(data.shape, dtype=bool)
            if nodata is not None:
                invalid |= data == nodata
            if self.negative_is_nodata:
                invalid |= data < 0
            invalid |= ~np.isfinite(data)
            if supersample > 1:
                s = supersample
                data = np.kron(data, np.ones((s, s))) / (s * s)
                invalid = np.kron(invalid, np.ones((s, s), dtype=bool))
                transform = transform @ Affine.scale(1 / s)
            inside = geometry_mask(
                [mapping(g)],
                out_shape=data.shape,
                transform=transform,
                invert=True,
                all_touched=all_touched,
            )
            n_inside += int(inside.sum())
            values.append(data[inside & ~invalid])
        fully = total_area > 0 and covered_area >= 0.999 * total_area
        return ZonalSample(np.concatenate(values) if values else np.array([]), n_inside, fully)


def _nearest_year_files(paths: Iterable[Path], year: int) -> tuple[list[Path], int | None]:
    by_year: dict[int, list[Path]] = {}
    for p in paths:
        m = re.search(r"(19|20)\d{2}", p.name)
        if m:
            by_year.setdefault(int(m.group(0)), []).append(p)
    if not by_year:
        return [], None
    best = min(by_year, key=lambda y: (abs(y - year), -y))
    return sorted(by_year[best]), best


def _fraction(sample: ZonalSample, classes: Sequence[int]) -> float | None:
    if len(sample.values) == 0:
        return None
    return float(np.isin(sample.values, classes).mean())


def _coverage_flag(sample: ZonalSample) -> str | None:
    if sample.n_inside == 0 or len(sample.values) == 0:
        return "NO_PIXELS"
    if not sample.fully_covered:
        return "PARTIAL_COVERAGE"
    return None


# ---------------------------------------------------------------------------
# the adapter interface
# ---------------------------------------------------------------------------
class LandCoverAdapter(ABC):
    """Land-cover-specific attributes. Implementations raise SourceUnavailable for any
    attribute whose data is not on disk; AdapterChain then asks the next adapter."""

    name: ClassVar[str]
    LAND_COVER_ATTRIBUTES: ClassVar[tuple[str, ...]] = (
        "imperviousness_pct",
        "urban_fraction",
        "riparian_width_m",
    )

    @abstractmethod
    def imperviousness_pct(self, ctx: CatchmentContext) -> dict[str, AttributeValue]: ...

    @abstractmethod
    def urban_fraction(self, ctx: CatchmentContext) -> dict[str, AttributeValue]: ...

    @abstractmethod
    def riparian_width_m(self, ctx: CatchmentContext) -> dict[str, AttributeValue]: ...

    def compute(self, attribute: str, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        if attribute not in self.LAND_COVER_ATTRIBUTES:
            raise KeyError(attribute)
        result: dict[str, AttributeValue] = getattr(self, attribute)(ctx)
        return result


class AdapterChain(LandCoverAdapter):
    """Per attribute, the first adapter with data answers. Same interface as one adapter."""

    name = "chain"

    def __init__(self, adapters: Sequence[LandCoverAdapter], *, strict: bool = False) -> None:
        if not adapters:
            raise ValueError("AdapterChain needs at least one adapter")
        self.adapters = list(adapters)
        self.strict = strict
        self.used: dict[str, str] = {}

    def _first(self, attribute: str, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        reasons = []
        for i, adapter in enumerate(self.adapters):
            try:
                result = adapter.compute(attribute, ctx)
            except SourceUnavailable as exc:
                reasons.append(f"{adapter.name}: {exc}")
                if self.strict:
                    raise
                continue
            self.used[attribute] = adapter.name
            if i > 0:
                log.warning(
                    "l2_static.adapter_fallback",
                    attribute=attribute,
                    used=adapter.name,
                    unavailable=reasons,
                )
            return result
        raise SourceUnavailable(f"{attribute}: no adapter has data - " + " | ".join(reasons))

    def imperviousness_pct(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        return self._first("imperviousness_pct", ctx)

    def urban_fraction(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        return self._first("urban_fraction", ctx)

    def riparian_width_m(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        return self._first("riparian_width_m", ctx)


def riparian_corridor(
    catchment: BaseGeometry,
    network: Sequence[LineString],
    tree: STRtree | None,
    width_m: float,
    ctx: CatchmentContext,
) -> tuple[BaseGeometry | None, float]:
    """(corridor polygon in WGS84, channel length in m) for the network inside a
    catchment. Buffering happens in the metric CRS."""
    candidates = network if tree is None else [network[i] for i in tree.query(catchment)]
    lines = [ln.intersection(catchment) for ln in candidates]
    lines = [ln for ln in lines if not ln.is_empty and ln.length > 0]
    if not lines:
        return None, 0.0
    parts: list[LineString] = []
    for ln in lines:
        parts.extend(getattr(ln, "geoms", [ln]))
    metric = ctx.to_metric(MultiLineString([p for p in parts if isinstance(p, LineString)]))
    corridor = metric.buffer(width_m).intersection(ctx.to_metric(catchment))
    return ctx.to_wgs(corridor), float(metric.length)


class WorldCoverAdapter(LandCoverAdapter):
    """ESA WorldCover 10 m. Global - the Pune adapter, and Coimbra's fallback.

    All three attributes are PROXIES and are flagged so:
      imperviousness_pct  100 x share of built-up (class 50) pixels. WorldCover built-up
                          is not an imperviousness density; it is collinear with
                          urban_fraction by construction. Replace with CLMS IMD (EU) or
                          GHSL built-up surface when available.
      urban_fraction      share of built-up pixels.
      riparian_width_m    vegetated share of a `riparian_corridor_m`-wide corridor either
                          side of every channel in the catchment, times that width: the
                          equivalent width of vegetated bank per side.
    """

    name = "worldcover"
    SOURCE = "ESA WorldCover 10 m v200 (2021), CC-BY-4.0"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or RAW / "landcover"
        # imperviousness_pct and urban_fraction are the same pixel count; sample once.
        self._built_up_share: dict[tuple[int, str], tuple[float | None, str | None, int]] = {}

    def _source(self) -> tuple[RasterSource, int]:
        paths = sorted(self.root.glob("ESA_WorldCover_10m_*_Map.tif"))
        if not paths:
            raise SourceUnavailable(
                f"no ESA_WorldCover_10m_*_Map.tif under {self.root} - run scripts/fetch_datasets.py"
            )
        year = int(re.search(r"_(20\d{2})_", paths[0].name).group(1))  # type: ignore[union-attr]
        return RasterSource(paths, nodata=WC_NODATA), year

    def _built_up(
        self, ctx: CatchmentContext, scale: float, flag: str
    ) -> dict[str, AttributeValue]:
        src, year = self._source()
        out = {}
        with src:
            for rid, geom in ctx.catchments.items():
                key = (id(ctx), rid)
                if key not in self._built_up_share:
                    sample = src.sample(geom)
                    coverage = _coverage_flag(sample)
                    frac = None if coverage == "NO_PIXELS" else _fraction(sample, [WC_BUILT_UP])
                    self._built_up_share[key] = (frac, coverage, year)
                frac, coverage, year = self._built_up_share[key]
                if coverage == "NO_PIXELS" or frac is None:
                    out[rid] = AttributeValue.missing("NO_PIXELS", self.SOURCE)
                    continue
                out[rid] = AttributeValue(
                    frac * scale,
                    f"{self.SOURCE}; class 50 (built-up) share",
                    coverage or flag,
                    year,
                )
        return out

    def imperviousness_pct(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        return self._built_up(ctx, 100.0, "PROXY_BUILT_UP_SHARE")

    def urban_fraction(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        return self._built_up(ctx, 1.0, "PROXY_BUILT_UP_SHARE")

    def riparian_width_m(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        width = float(ctx.settings.get("riparian_corridor_m", 100))
        if not ctx.network:
            raise SourceUnavailable("no channel network loaded for the riparian corridor")
        src, year = self._source()
        tree = STRtree(ctx.network)
        out = {}
        with src:
            for rid, geom in ctx.catchments.items():
                corridor, length = riparian_corridor(geom, ctx.network, tree, width, ctx)
                if corridor is None or corridor.is_empty:
                    out[rid] = AttributeValue.missing("NO_CHANNEL_IN_CATCHMENT", self.SOURCE)
                    continue
                sample = src.sample(corridor)
                frac = _fraction(sample, WC_VEGETATED)
                if frac is None:
                    out[rid] = AttributeValue.missing("NO_PIXELS", self.SOURCE)
                    continue
                out[rid] = AttributeValue(
                    frac * width,
                    f"{self.SOURCE}; vegetated share of a {width:g} m corridor per bank",
                    _coverage_flag(sample) or "PROXY_VEGETATED_CORRIDOR",
                    year,
                )
        return out


class ClmsAdapter(LandCoverAdapter):
    """Copernicus Land Monitoring Service (EU only).

    Expected layout (drop the CLMS downloads here; filenames only need the year):
      data/raw/clms/imperviousness/*IMD*<year>*.tif   Imperviousness Density, 0-100
      data/raw/clms/urban_atlas/*.gpkg                Urban Atlas (code_2018 / code_2012)
      data/raw/clms/corine/*<year>*.tif               CORINE raster (codes 1-44)
      data/raw/clms/riparian_zones/*.gpkg             Riparian Zones delineation polygons
    """

    name = "clms"
    IMD_NODATA = (254, 255)
    CORINE_ARTIFICIAL = tuple(range(1, 12))  # CLC raster codes 1-11 = classes 111-142

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or RAW / "clms"

    def _require(self, sub: str, pattern: str) -> list[Path]:
        paths = sorted((self.root / sub).glob(pattern))
        if not paths:
            raise SourceUnavailable(
                f"no {pattern} under {self.root / sub} (CLMS registration pending? "
                "see DATA_INVENTORY.md)"
            )
        return paths

    def imperviousness_pct(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        paths, year = _nearest_year_files(
            self._require("imperviousness", "*.tif"), ctx.reference_year
        )
        if not paths:
            raise SourceUnavailable("imperviousness rasters carry no year in their names")
        source = f"CLMS Imperviousness Density {year}"
        out = {}
        with RasterSource(paths) as src:
            for rid, geom in ctx.catchments.items():
                sample = src.sample(geom)
                valid = sample.values[~np.isin(sample.values, self.IMD_NODATA)]
                if len(valid) == 0:
                    out[rid] = AttributeValue.missing("NO_PIXELS", source)
                    continue
                flag = None if sample.fully_covered else "PARTIAL_COVERAGE"
                out[rid] = AttributeValue(float(valid.mean()), source, flag, year)
        return out

    def urban_fraction(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        ua = sorted((self.root / "urban_atlas").glob("*.gpkg"))
        if ua:
            return self._urban_atlas(ua, ctx)
        paths, year = _nearest_year_files(self._require("corine", "*.tif"), ctx.reference_year)
        source = f"CLMS CORINE Land Cover {year}; artificial surfaces (1xx) share"
        out = {}
        with RasterSource(paths) as src:
            for rid, geom in ctx.catchments.items():
                sample = src.sample(geom)
                frac = _fraction(sample, self.CORINE_ARTIFICIAL)
                out[rid] = (
                    AttributeValue.missing("NO_PIXELS", source)
                    if frac is None
                    else AttributeValue(frac, source, _coverage_flag(sample), year)
                )
        return out

    def _urban_atlas(self, paths: list[Path], ctx: CatchmentContext) -> dict[str, AttributeValue]:
        import geopandas as gpd

        frames = [gpd.read_file(p) for p in paths]
        ua = pd.concat(frames, ignore_index=True)
        code_col = next(
            (c for c in ("code_2018", "code_2012", "CODE2012") if c in ua.columns), None
        )
        if code_col is None:
            raise SourceUnavailable(f"Urban Atlas files lack a code column: {list(ua.columns)}")
        ua = gpd.GeoDataFrame(ua, geometry="geometry").to_crs(ctx.metric_crs)
        # Artificial surfaces (1xxxx) minus green urban / sports and leisure (14xxx).
        urban = ua[ua[code_col].astype(str).str.match(r"^1[123]")]
        tree = STRtree(list(urban.geometry))
        year_match = re.search(r"(20\d{2})", code_col)
        year = int(year_match.group(1)) if year_match else None
        source = f"CLMS Urban Atlas ({code_col}); classes 11-13 share"
        out = {}
        for rid, geom in ctx.catchments.items():
            g = ctx.to_metric(geom)
            hits = [urban.geometry.iloc[i] for i in tree.query(g)]
            area = sum(h.intersection(g).area for h in hits)
            out[rid] = AttributeValue(area / g.area if g.area else None, source, None, year)
        return out

    def riparian_width_m(self, ctx: CatchmentContext) -> dict[str, AttributeValue]:
        import geopandas as gpd

        paths = self._require("riparian_zones", "*.gpkg")
        rz = pd.concat([gpd.read_file(p) for p in paths], ignore_index=True)
        rz = gpd.GeoDataFrame(rz, geometry="geometry").to_crs(ctx.metric_crs)
        tree = STRtree(list(rz.geometry))
        net_tree = STRtree(ctx.network) if ctx.network else None
        source = "CLMS Riparian Zones delineation; RZ area / (2 x channel length)"
        out = {}
        for rid, geom in ctx.catchments.items():
            _, length = riparian_corridor(geom, ctx.network, net_tree, 1.0, ctx)
            if length <= 0:
                out[rid] = AttributeValue.missing("NO_CHANNEL_IN_CATCHMENT", source)
                continue
            g = ctx.to_metric(geom)
            area = sum(rz.geometry.iloc[i].intersection(g).area for i in tree.query(g))
            out[rid] = AttributeValue(area / (2 * length), source, None, None)
        return out


ADAPTERS: dict[str, type[LandCoverAdapter]] = {
    "clms": ClmsAdapter,
    "worldcover": WorldCoverAdapter,
}


def resolve_adapter(static_cfg: dict[str, Any], *, strict: bool = False) -> AdapterChain:
    names = [static_cfg["land_cover_adapter"]]
    if static_cfg.get("land_cover_fallback") and not strict:
        names.append(static_cfg["land_cover_fallback"])
    unknown = [n for n in names if n not in ADAPTERS]
    if unknown:
        raise ValueError(f"unknown land-cover adapter(s) {unknown}; known: {sorted(ADAPTERS)}")
    return AdapterChain([ADAPTERS[n]() for n in names], strict=strict)


# ---------------------------------------------------------------------------
# adapter-independent attributes
# ---------------------------------------------------------------------------
def population(ctx: CatchmentContext, root: Path | None = None) -> dict[str, AttributeValue]:
    root = root or RAW / "population"
    paths = sorted(root.glob("GHS_POP_*.zip")) + sorted(root.glob("GHS_POP_*.tif"))
    if not paths:
        raise SourceUnavailable(f"no GHS_POP_* under {root}")
    year_match = re.search(r"_E(\d{4})_", paths[0].name)
    year = int(year_match.group(1)) if year_match else None
    source = f"GHSL GHS-POP E{year} 3 arc-second (JRC), CC-BY-4.0"
    out = {}
    # Supersample x3: a 90 m cell is split into nine 30 m cells carrying 1/9 of its
    # people, so small catchments get a pro-rata share instead of all-or-nothing.
    with RasterSource(paths, negative_is_nodata=True) as src:
        for rid, geom in ctx.catchments.items():
            sample = src.sample(geom, supersample=3)
            flag = _coverage_flag(sample)
            if flag == "NO_PIXELS":
                out[rid] = AttributeValue.missing(flag, source)
            elif flag == "PARTIAL_COVERAGE" or sample.n_nodata:
                out[rid] = AttributeValue.missing("PARTIAL_COVERAGE", source)
            else:
                out[rid] = AttributeValue(float(round(sample.values.sum())), source, None, year)
    return out


def alan_radiance(ctx: CatchmentContext, root: Path | None = None) -> dict[str, AttributeValue]:
    """Mean VIIRS DNB radiance over the catchment, averaged over the monthly composites of
    the year nearest reference_year. all_touched: a 500 m cell covering a small catchment
    is the right sample of an intensive quantity."""
    root = root or RAW / "viirs"
    paths, year = _nearest_year_files(sorted(root.glob("*.tif")), ctx.reference_year)
    if not paths:
        raise SourceUnavailable(
            f"no VIIRS DNB composites under {root} (EOG registration pending? see "
            "DATA_INVENTORY.md)"
        )
    source = f"VIIRS DNB monthly composites {year} (EOG, Colorado School of Mines), nW/cm2/sr"
    sources = [RasterSource([p], negative_is_nodata=True) for p in paths]
    out = {}
    try:
        for rid, geom in ctx.catchments.items():
            means = []
            for src in sources:
                sample = src.sample(geom, all_touched=True)
                if len(sample.values):
                    means.append(float(sample.values.mean()))
            out[rid] = (
                AttributeValue(float(np.mean(means)), source, None, year)
                if means
                else AttributeValue.missing("NO_PIXELS", source)
            )
    finally:
        for src in sources:
            src.close()
    return out


def road_density(ctx: CatchmentContext, roads: Sequence[LineString]) -> dict[str, AttributeValue]:
    """Motorised road length (km) inside the catchment / catchment area (km2)."""
    if not roads:
        raise SourceUnavailable("no OSM roads loaded")
    classes = ctx.settings.get("road_classes", [])
    source = f"OSM highway in {{{', '.join(classes)}}} (ODbL)"
    # Everything in metres once, then clip and measure as arrays.
    metric_roads = ctx.to_metric_many(list(roads))
    road_len = shapely.length(metric_roads)
    tree = STRtree(metric_roads)
    out = {}
    for rid, geom in ctx.catchments.items():
        area = ctx.area_km2.get(rid)
        if not area:
            out[rid] = AttributeValue.missing("NO_CATCHMENT_AREA", source)
            continue
        poly = ctx.to_metric(geom)
        shapely.prepare(poly)
        # Only roads that touch the polygon; half the bbox hits lie wholly outside it.
        hits = tree.query(poly, predicate="intersects")
        inside = shapely.contains(poly, metric_roads[hits])
        length_m = float(road_len[hits[inside]].sum())
        edge = hits[~inside]
        if len(edge):
            length_m += float(shapely.length(shapely.intersection(metric_roads[edge], poly)).sum())
        out[rid] = AttributeValue(length_m / 1000.0 / area, source, None, None)
    return out


def reach_riparian_climatology(obs: pd.DataFrame, train_end: Any) -> pd.Series:
    """Per reach: mean of monthly means of OK riparian NDVI, dates <= train_end.
    Monthly first so cloud-free summers do not dominate the average."""
    ok = obs[(obs["riparian_flag"] == "OK") & obs["riparian_ndvi"].notna()].copy()
    ok = ok[pd.to_datetime(ok["obs_date"]) <= pd.Timestamp(train_end)]
    if ok.empty:
        return pd.Series(dtype=float)
    ok["month"] = pd.to_datetime(ok["obs_date"]).dt.month
    monthly = ok.groupby(["reach_id", "month"])["riparian_ndvi"].mean()
    return monthly.groupby(level="reach_id").mean()


def riparian_ndvi_mean(
    ctx: CatchmentContext,
    reach_ndvi: pd.Series,
    reach_lines: dict[str, LineString],
    study_bbox: tuple[float, float, float, float],
) -> dict[str, AttributeValue]:
    source = "Sentinel-2 L2A riparian NDVI (L1, 30 m buffer, water excluded); length-weighted"
    points = {rid: line.representative_point() for rid, line in reach_lines.items()}
    lengths = {rid: ctx.to_metric(line).length for rid, line in reach_lines.items()}
    study = box(*study_bbox)
    out = {}
    for rid, geom in ctx.catchments.items():
        inside = [r for r, p in points.items() if geom.contains(p) or r == rid]
        observed = [r for r in inside if r in reach_ndvi.index and not math.isnan(reach_ndvi[r])]
        if not observed:
            out[rid] = AttributeValue.missing("NO_S2_RIPARIAN_OBS", source)
            continue
        w = np.array([lengths[r] for r in observed])
        v = np.array([reach_ndvi[r] for r in observed])
        flag = None
        if not study.contains(geom):
            flag = "NETWORK_WITHIN_STUDY_BBOX_ONLY"  # upstream reaches outside L0's bbox
        elif len(observed) < len(inside):
            flag = "PARTIAL_REACH_COVERAGE"
        out[rid] = AttributeValue(float((w * v).sum() / w.sum()), source, flag, None)
    return out


# ---------------------------------------------------------------------------
# OSM lines (one cached pass over the pbf)
# ---------------------------------------------------------------------------
def _read_osm_lines(
    pbf: Path,
    bounds: tuple[float, float, float, float],
    road_classes: list[str],
    waterway_tags: list[str],
) -> dict[str, Any]:
    import osmium
    import osmium.filter

    window = box(*bounds)
    roads: list[list[tuple[float, float]]] = []
    channels: list[list[tuple[float, float]]] = []
    wanted_roads, wanted_water = set(road_classes), set(waterway_tags)
    processor = (
        osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("highway", "waterway"))
    )
    for way in processor:
        if not isinstance(way, osmium.osm.Way):  # the EntityFilter guarantees this
            continue
        highway, waterway = way.tags.get("highway"), way.tags.get("waterway")
        target = (
            roads if highway in wanted_roads else channels if waterway in wanted_water else None
        )
        if target is None:
            continue
        coords = [(n.location.lon, n.location.lat) for n in way.nodes if n.location.valid()]
        if len(coords) >= 2 and LineString(coords).intersects(window):
            target.append(coords)
    return {"source": pbf.name, "bounds": list(bounds), "roads": roads, "channels": channels}


def load_osm_lines(
    city_cfg: dict[str, Any], bounds: tuple[float, float, float, float]
) -> tuple[list[LineString], list[LineString]]:
    static = city_cfg["static_attributes"]
    road_classes = list(static["road_classes"])
    waterway_tags = list(city_cfg["network"]["waterway_tags"])
    pbfs = sorted((RAW / "osm").glob("*-latest.osm.pbf"))
    if not pbfs:
        raise SourceUnavailable(f"no *-latest.osm.pbf under {RAW / 'osm'}")
    params = {
        "stage": "l2_static",
        "pbf": pbfs[0].name,
        "pbf_bytes": pbfs[0].stat().st_size,
        "bounds": [round(b, 5) for b in bounds],
        "road_classes": sorted(road_classes),
        "waterway_tags": sorted(waterway_tags),
    }
    entry = DiskCache("osm").get_or_fetch(
        params,
        lambda: _read_osm_lines(pbfs[0], bounds, road_classes, waterway_tags),
        slug=f"static_lines_{city_cfg['city']}",
        source=f"OSM {pbfs[0].name} (ODbL)",
    )
    payload = entry.payload
    roads = [LineString(c) for c in payload["roads"]]
    channels = [LineString(c) for c in payload["channels"]]
    if not roads or not channels:
        raise RuntimeError(
            f"OSM pass over {bounds} found {len(roads)} roads and {len(channels)} channels - "
            "an empty result is a failure, not an attribute of zero."
        )
    return roads, channels


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def input_fingerprint() -> list[tuple[str, int]]:
    """(path, size) of every raw input l2_static can read - a new CLMS or VIIRS file
    changes the fingerprint and invalidates the derived-result cache."""
    files = []
    for sub in ("landcover", "clms", "population", "viirs", "osm"):
        root = RAW / sub
        if root.exists():
            files += [
                (str(p.relative_to(RAW)), p.stat().st_size)
                for p in sorted(root.rglob("*"))
                if p.is_file() and not p.name.endswith(".meta.json")
            ]
    return files


def load_catchments(city: str) -> tuple[dict[str, Any], dict[str, float | None], dict[str, Any]]:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        rows = (
            session.execute(
                text(
                    "SELECT reach_id, catchment_area_km2, "
                    "ST_AsGeoJSON(catchment_geom) AS catchment, "
                    "ST_AsGeoJSON(geom) AS line FROM reaches WHERE city = :city ORDER BY reach_id"
                ),
                {"city": city},
            )
            .mappings()
            .all()
        )
    if not rows:
        raise RuntimeError(f"No reaches in PostGIS for city={city}. Run L0 first.")
    catchments = {r["reach_id"]: shape(json.loads(r["catchment"])) for r in rows if r["catchment"]}
    area = {r["reach_id"]: r["catchment_area_km2"] for r in rows}
    lines = {r["reach_id"]: shape(json.loads(r["line"])) for r in rows}
    return catchments, area, lines


def load_l0_flags(city: str) -> dict[str, list[str]]:
    path = INTERIM_DIR / f"catchments_{city}.geojson"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        f["properties"]["reach_id"]: f["properties"].get("flags", []) for f in payload["features"]
    }


def assemble_rows(
    reach_ids: Sequence[str],
    results: dict[str, dict[str, AttributeValue]],
    reference_year: int,
    adapter_used: dict[str, str],
    l0_flags: dict[str, list[str]],
) -> pd.DataFrame:
    """Per-attribute results -> catchment_attributes rows, with sources and flags JSON."""
    rows = []
    for rid in reach_ids:
        row: dict[str, Any] = {"reach_id": rid}
        sources: dict[str, Any] = {}
        flags: dict[str, Any] = {}
        years = []
        for attribute in ATTRIBUTES:
            av = results.get(attribute, {}).get(rid) or AttributeValue.missing("NOT_COMPUTED")
            value = av.value
            if attribute == "population" and value is not None:
                value = int(value)
            row[attribute] = value
            sources[attribute] = av.source
            if av.flag:
                flags[attribute] = av.flag
            if attribute in LandCoverAdapter.LAND_COVER_ATTRIBUTES and av.year:
                years.append(av.year)
        catchment_flags = [
            f
            for f in l0_flags.get(rid, [])
            if f in ("CATCHMENT_TRUNCATED", "SNAP_OFF_STREAM_NETWORK", "MONOTONICITY_VIOLATION")
        ]
        if catchment_flags:
            # A truncated catchment's extensive attributes (population, road length) are
            # lower bounds; carried on the row, not buried in the L0 GeoJSON.
            flags["_catchment"] = catchment_flags
        row["year"] = max(years) if years else reference_year
        row["adapter"] = ",".join(f"{a}={n}" for a, n in sorted(adapter_used.items()))
        row["sources"] = sources
        row["flags"] = flags
        rows.append(row)
    return pd.DataFrame(rows)


def build_static_attributes(
    city: str, *, strict: bool = False, write_db: bool = True
) -> dict[str, Any]:
    city_cfg = load_city_config(city)
    static = city_cfg["static_attributes"]
    reference_year = int(static.get("reference_year", 2021))
    train_end = as_date(load_config("modelling")["splits"]["train_end"])

    catchments, area, lines = load_catchments(city)
    all_bounds = np.array([g.bounds for g in catchments.values()])
    lo, hi = all_bounds.min(axis=0), all_bounds.max(axis=0)
    union_bounds = (float(lo[0]), float(lo[1]), float(hi[2]), float(hi[3]))
    ctx = CatchmentContext(
        catchments=catchments,
        area_km2=area,
        metric_crs=city_cfg["crs"]["metric"],
        reference_year=reference_year,
        settings=static,
    )
    results: dict[str, dict[str, AttributeValue]] = {}
    unavailable: dict[str, str] = {}

    def attempt(attribute: str, fn: Any) -> None:
        key = {
            "attribute": attribute,
            "catchments": catchment_key,
            "settings": static,
            "reference_year": reference_year,
            "inputs": input_fingerprint(),
            "strict": strict,
        }
        cached = result_cache.get(key, slug=f"{city}_{attribute}")
        if cached is not None:
            payload = cached.payload
            results[attribute] = {r: AttributeValue(**v) for r, v in payload["values"].items()}
            if payload.get("adapter"):
                chain.used[attribute] = payload["adapter"]
            log.info("l2_static.cached_result", attribute=attribute)
            return
        try:
            results[attribute] = fn()
            result_cache.put(
                key,
                {
                    "values": {r: v.__dict__ for r, v in results[attribute].items()},
                    "adapter": chain.used.get(attribute),
                },
                slug=f"{city}_{attribute}",
                source="derived: pipeline.l2_static",
            )
        except SourceUnavailable as exc:
            if strict:
                raise
            unavailable[attribute] = str(exc)
            log.warning("l2_static.source_unavailable", attribute=attribute, reason=str(exc))
            results[attribute] = {
                rid: AttributeValue.missing("SOURCE_UNAVAILABLE") for rid in catchments
            }

    # Derived per-attribute results are cached like network responses: the riparian
    # corridor pass alone takes ~25 min, and re-running one attribute must not redo it.
    # The key covers the catchment geometries, the settings and every input file.
    result_cache = DiskCache("l2_static_results", root=INTERIM_DIR)
    catchment_key = cache_key({r: g.wkt for r, g in sorted(catchments.items())})
    chain = resolve_adapter(static, strict=strict)

    with stage(log, "l2_static", city=city) as counters:
        roads, channels = load_osm_lines(city_cfg, union_bounds)
        ctx.network = channels
        for attribute in LandCoverAdapter.LAND_COVER_ATTRIBUTES:
            attempt(attribute, lambda a=attribute: chain.compute(a, ctx))
        attempt("population", lambda: population(ctx))
        attempt("alan_radiance", lambda: alan_radiance(ctx))
        attempt("road_density_km_km2", lambda: road_density(ctx, roads))

        obs = load_riparian_observations(city)
        reach_ndvi = reach_riparian_climatology(obs, train_end)
        results["riparian_ndvi_mean"] = riparian_ndvi_mean(
            ctx, reach_ndvi, lines, bbox_tuple(city_cfg)
        )

        # Fail loudly: a source that answered but produced no value anywhere is broken.
        for attribute, values in results.items():
            if attribute in unavailable:
                continue
            if values and all(v.value is None for v in values.values()):
                reasons = Counter(v.flag for v in values.values())
                if attribute == "riparian_ndvi_mean" and set(reasons) == {"NO_S2_RIPARIAN_OBS"}:
                    log.warning(
                        "l2_static.riparian_ndvi_empty", note="run L1 first", reasons=dict(reasons)
                    )
                    continue
                raise RuntimeError(f"{attribute}: every catchment is NULL ({dict(reasons)})")

        frame = assemble_rows(
            sorted(catchments), results, reference_year, chain.used, load_l0_flags(city)
        )
        null_counts = {a: int(frame[a].isna().sum()) for a in ATTRIBUTES}
        flag_counts = {
            a: dict(Counter(f.get(a) for f in frame["flags"] if f.get(a))) for a in ATTRIBUTES
        }
        counters.record(
            rows_in=len(lines),
            rows_out=len(frame),
            adapters=chain.used,
            unavailable=sorted(unavailable),
            null_counts=null_counts,
            flag_counts=flag_counts,
        )
        dropped = len(lines) - len(frame)
        if dropped:
            counters.drop(dropped, "NO_CATCHMENT")
        if write_db:
            from core.db import bulk_upsert

            bulk_upsert(frame, "catchment_attributes", ["reach_id"])

    return {
        "rows": len(frame),
        "adapters": chain.used,
        "unavailable": unavailable,
        "null_counts": null_counts,
        "flag_counts": flag_counts,
        "summary": frame[ATTRIBUTES].describe().round(3).to_dict(),
    }


def load_riparian_observations(city: str) -> pd.DataFrame:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        return pd.read_sql(
            text(
                "SELECT o.reach_id, o.obs_date, o.riparian_ndvi, o.riparian_flag "
                "FROM observations o "
                "JOIN reaches r USING (reach_id) WHERE r.city = :city AND o.source = 'S2'"
            ),
            session.connection(),
            params={"city": city},
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L2 static catchment attributes")
    parser.add_argument("--city", default="coimbra")
    parser.add_argument(
        "--strict", action="store_true", help="no fallback adapters, no NULL sources"
    )
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args(argv)
    summary = build_static_attributes(args.city, strict=args.strict, write_db=not args.no_db)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
