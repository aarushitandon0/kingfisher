"""L1 - observables: Sentinel-2 L2A per reach via the Sentinel Hub Statistical API.

Run:

    python -m pipeline.l1_satellite --city coimbra --estimate          # PU forecast, no spend
    python -m pipeline.l1_satellite --city coimbra --limit 3            # plumbing check
    python -m pipeline.l1_satellite --city coimbra --max-pu 5000        # full run, capped

Writes `observations` (source='S2') and data/interim/s2_<city>.parquet (the same rows
plus pixel counts and water-pixel band means, for QA).

WHAT ONE ACQUISITION BECOMES
----------------------------
Two Statistical API requests per (reach, calendar year), both aggregated per
acquisition day (P1D) at 10 m in the reach's UTM zone:

  water     centreline buffered 15 m each side - the Day-1 observability corridor, so
            water_pixel_count here and reaches.observable describe the same ground.
            Per pixel: water = usable AND MNDWI > 0 AND SCL = WATER. MNDWI, NDCI and the
            Nechad turbidity proxy are averaged over water pixels only.
  riparian  centreline buffered 30 m each side. NDVI averaged over usable pixels that
            are NOT water (MNDWI > 0 OR SCL = WATER excluded).

Masked means are reconstructed as mean(x*m) / mean(m) over data pixels, from the
evalscript's 0/1 masks and masked products. That makes the statistics independent of how
the service treats NaN, and gives exact pixel counts for free.

Per acquisition, quality_flag is decided in this order:

  no data pixels in the corridor        -> row dropped, logged NO_DATA_COVERAGE
                                           (the reach was not imaged; not an observation)
  cloud_fraction > max_cloud_fraction   -> CLOUD
  water pixels < min_water_pixels       -> NO_WATER_PIXELS
  Nechad diverges / index off-domain    -> OUT_OF_RANGE
  otherwise                             -> OK

Only OK rows carry index values. Every other row stores NULL values plus the flag, and
still stores water_pixel_count and cloud_fraction, because those were measured. The one
exception is a fully-clouded corridor: no pixel was usable, so the water pixel count is
unknown and is NULL - "the satellite could not see" is not "the satellite saw no water".

COST
----
Processing units are read from each response's `x-processingunits-spent` header and
stored in the cache meta sidecar, so the cache directory is the all-time spend ledger.
Every batch logs PU spent this run and all-time. `--max-pu` stops the run before a batch
that would exceed it, and fails loudly. Observable reaches are fetched first, so a
budget-capped run spends on the reaches that carry signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from shapely.geometry import LineString, shape
from shapely.ops import transform as shapely_transform

from core.cache import DiskCache, cache_key
from core.config import load_config
from core.logging import get_logger, stage
from core.settings import DATA_DIR, get_settings
from pipeline.l0_network import load_city_config

log = get_logger(__name__)

SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
SH_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
STATISTICS_URL = f"{SH_BASE_URL}/api/v1/statistics"
SOURCE = "S2"
CACHE_NAMESPACE = "sentinelhub"
INTERIM_DIR = DATA_DIR / "interim"

WATER, RIPARIAN = "water", "riparian"

# Columns written to `observations`.
OBS_COLUMNS = [
    "reach_id",
    "obs_date",
    "source",
    "turbidity_proxy",
    "ndci",
    "mndwi",
    "riparian_ndvi",
    "water_pixel_count",
    "cloud_fraction",
    "quality_flag",
    "riparian_flag",
]

# Band order of the water request's masked-product output ("wx").
WATER_VALUE_BANDS = ["mndwi", "ndci", "turbidity", "B02", "B03", "B04", "B05", "B08", "B11"]


class BudgetExceeded(RuntimeError):
    """The next batch would push this run past --max-pu. Raised, never swallowed."""


class NonRetryableRequestError(RuntimeError):
    """A 4xx that retrying cannot fix (auth, bad request). Aborts the run."""


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class S2Settings:
    """Everything that decides what an acquisition becomes. Built from
    config/sentinel2.yaml plus the city's `observability` block."""

    bands: tuple[str, ...]
    scl_cloud: tuple[int, ...]
    scl_invalid: tuple[int, ...]
    scl_water: int
    water_half_width_m: float
    riparian_half_width_m: float
    nechad_a: float
    nechad_c: float
    max_rho_fraction_of_c: float
    min_water_pixels: int
    max_cloud_fraction: float
    min_turbidity_valid_fraction: float
    riparian_min_pixels: int
    resolution_m: float = 10.0
    collection: str = "sentinel-2-l2a"
    batch_size: int = 16
    max_threads: int = 4
    max_attempts: int = 6
    backoff_base_s: float = 2.0
    backoff_max_s: float = 120.0

    @classmethod
    def from_config(cls, s2: dict[str, Any], city_cfg: dict[str, Any]) -> S2Settings:
        obs = city_cfg.get("observability", {})
        req = s2.get("requests", {})
        return cls(
            bands=tuple(s2["bands"]),
            scl_cloud=tuple(s2["scl"]["cloud"]),
            scl_invalid=tuple(s2["scl"]["invalid"]),
            scl_water=int(s2["scl"]["water"]),
            water_half_width_m=float(s2["geometry"]["water_half_width_m"]),
            riparian_half_width_m=float(s2["geometry"]["riparian_half_width_m"]),
            nechad_a=float(s2["nechad"]["A"]),
            nechad_c=float(s2["nechad"]["C"]),
            max_rho_fraction_of_c=float(s2["nechad"]["max_rho_fraction_of_C"]),
            min_water_pixels=int(obs["min_water_pixels"]),
            max_cloud_fraction=float(obs["max_cloud_fraction"]),
            min_turbidity_valid_fraction=float(s2["quality"]["min_turbidity_valid_fraction"]),
            riparian_min_pixels=int(s2["quality"]["riparian_min_pixels"]),
            resolution_m=float(s2.get("resolution_m", 10)),
            collection=str(s2.get("collection", "sentinel-2-l2a")),
            batch_size=int(req.get("batch_size", 16)),
            max_threads=int(req.get("max_threads", 4)),
            max_attempts=int(req.get("max_attempts", 6)),
            backoff_base_s=float(req.get("backoff_base_s", 2.0)),
            backoff_max_s=float(req.get("backoff_max_s", 120.0)),
        )


def load_settings(city: str) -> S2Settings:
    return S2Settings.from_config(load_config("sentinel2"), load_city_config(city))


# ---------------------------------------------------------------------------
# evalscripts (compiled from config; their hash is part of every cache key)
# ---------------------------------------------------------------------------
def _js_list(values: Sequence[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def water_evalscript(s: S2Settings) -> str:
    bands = ", ".join(f'"{b}"' for b in (*s.bands, "dataMask"))
    return f"""//VERSION=3
// Kingfisher L1 water request. px = 0/1 pixel classes; wx = water-masked products.
// Means over water pixels are mean(wx) / mean(px.water), reconstructed client-side.
var CLOUD = {_js_list(s.scl_cloud)};
var INVALID = {_js_list(s.scl_invalid)};
var SCL_WATER = {s.scl_water};
var NECHAD_A = {s.nechad_a!r};
var NECHAD_C = {s.nechad_c!r};
var RHO_MAX = {s.max_rho_fraction_of_c!r} * NECHAD_C;

function setup() {{
  return {{
    input: [{{ bands: [{bands}] }}],
    output: [
      {{ id: "px", bands: 4, sampleType: "FLOAT32" }},
      {{ id: "wx", bands: 9, sampleType: "FLOAT32" }},
      {{ id: "dataMask", bands: 1 }}
    ]
  }};
}}

function ratio(a, b) {{ return (a + b) === 0 ? 0 : (a - b) / (a + b); }}

function evaluatePixel(s) {{
  var cloud = CLOUD.indexOf(s.SCL) >= 0 ? 1 : 0;
  var usable = (cloud === 0 && INVALID.indexOf(s.SCL) < 0) ? 1 : 0;
  var mndwi = ratio(s.B03, s.B11);
  var ndci = ratio(s.B05, s.B04);
  var water = (usable === 1 && mndwi > 0 && s.SCL === SCL_WATER) ? 1 : 0;
  var turbValid = (water === 1 && s.B04 >= 0 && s.B04 < RHO_MAX) ? 1 : 0;
  var turb = turbValid === 1 ? NECHAD_A * s.B04 / (1 - s.B04 / NECHAD_C) : 0;
  return {{
    px: [cloud, usable, water, turbValid],
    wx: [water * mndwi, water * ndci, turb,
         water * s.B02, water * s.B03, water * s.B04, water * s.B05, water * s.B08,
         water * s.B11],
    dataMask: [s.dataMask]
  }};
}}
"""


def riparian_evalscript(s: S2Settings) -> str:
    return f"""//VERSION=3
// Kingfisher L1 riparian request. NDVI over usable, non-water pixels of the buffer.
var CLOUD = {_js_list(s.scl_cloud)};
var INVALID = {_js_list(s.scl_invalid)};
var SCL_WATER = {s.scl_water};

function setup() {{
  return {{
    input: [{{ bands: ["B03", "B04", "B08", "B11", "SCL", "dataMask"] }}],
    output: [
      {{ id: "px", bands: 3, sampleType: "FLOAT32" }},
      {{ id: "rx", bands: 1, sampleType: "FLOAT32" }},
      {{ id: "dataMask", bands: 1 }}
    ]
  }};
}}

function ratio(a, b) {{ return (a + b) === 0 ? 0 : (a - b) / (a + b); }}

function evaluatePixel(s) {{
  var cloud = CLOUD.indexOf(s.SCL) >= 0 ? 1 : 0;
  var usable = (cloud === 0 && INVALID.indexOf(s.SCL) < 0) ? 1 : 0;
  var waterish = (ratio(s.B03, s.B11) > 0 || s.SCL === SCL_WATER) ? 1 : 0;
  var land = (usable === 1 && waterish === 0) ? 1 : 0;
  return {{
    px: [cloud, usable, land],
    rx: [land * ratio(s.B08, s.B04)],
    dataMask: [s.dataMask]
  }};
}}
"""


def evalscript_hash(script: str) -> str:
    return hashlib.sha256(script.encode("utf-8")).hexdigest()


def nechad_pixel(rho: float, s: S2Settings) -> float | None:
    """Python mirror of the evalscript's per-pixel turbidity index (tests pin the two
    together): T = A*rho / (1 - rho/C), valid only for 0 <= rho < max_rho_fraction_of_C*C.
    As rho -> C the denominator -> 0 and T diverges, so those pixels are excluded
    (None) rather than averaged in; a date where too few survive is OUT_OF_RANGE."""
    if not (0.0 <= rho < s.max_rho_fraction_of_c * s.nechad_c):
        return None
    return s.nechad_a * rho / (1.0 - rho / s.nechad_c)


# ---------------------------------------------------------------------------
# chunking + geometry
# ---------------------------------------------------------------------------
def year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar-year chunks covering [start, end], both inclusive. A completed year is a
    stable cache key; only the chunk containing `end` changes as `end` moves."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    chunks = []
    year = start.year
    while year <= end.year:
        lo = max(start, date(year, 1, 1))
        hi = min(end, date(year, 12, 31))
        chunks.append((lo, hi))
        year += 1
    return chunks


def geometry_hash(line: LineString, half_width_m: float) -> str:
    """Cache-key component: if L0 moves a centreline, the old response describes other
    ground and must not be served for the new one."""
    return cache_key({"wkt": line.wkt, "half_width_m": half_width_m})[:16]


def corridor(line: LineString, half_width_m: float) -> Any:
    """Buffer the centreline in its UTM zone and wrap it as a sentinelhub Geometry.

    UTM, not WGS84: the Statistical API reads `resolution` in the request CRS's units,
    and in degrees (10, 10) would collapse a reach into one enormous pixel.
    """
    from pyproj import Transformer
    from sentinelhub import CRS, Geometry  # type: ignore[attr-defined]

    centroid = line.centroid
    utm = CRS.get_utm_from_wgs84(centroid.x, centroid.y)
    to_utm = Transformer.from_crs("EPSG:4326", utm.ogc_string(), always_xy=True).transform
    return Geometry(shapely_transform(to_utm, line).buffer(half_width_m, cap_style=2), crs=utm)


# ---------------------------------------------------------------------------
# response parsing (pure - unit tested without a network)
# ---------------------------------------------------------------------------
def _num(value: Any) -> float | None:
    """Statistical API numbers; "NaN" (a string in the JSON) becomes None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _band_stats(output: dict[str, Any], index: int) -> dict[str, Any]:
    stats: dict[str, Any] = output["bands"][f"B{index}"]["stats"]
    return stats


def _counts(output: dict[str, Any], n_bands: int) -> tuple[int, list[int]]:
    """(data pixels, per-band count of pixels where the 0/1 mask is 1)."""
    first = _band_stats(output, 0)
    data = int(first["sampleCount"]) - int(first["noDataCount"])
    if data <= 0:
        return 0, [0] * n_bands
    counts = []
    for b in range(n_bands):
        mean = _num(_band_stats(output, b).get("mean"))
        counts.append(int(round((mean or 0.0) * data)))
    return data, counts


def _masked_mean(output: dict[str, Any], index: int, data: int, count: int) -> float | None:
    """mean over masked pixels = mean(x*m) over data pixels * data / count."""
    if count <= 0:
        return None
    mean = _num(_band_stats(output, index).get("mean"))
    if mean is None:
        return None
    return mean * data / count


def _interval_day(item: dict[str, Any]) -> str:
    return str(item.get("interval", {}).get("from", "?"))[:10]


def parse_water_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Statistical API water response -> one raw row per acquisition day.

    Returns (rows, failed_days). Rows carry pixel counts and water-pixel means; they are
    not yet classified. A day whose processing failed server-side is returned separately
    so the caller can count it as dropped with a reason.
    """
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for item in payload.get("data", []):
        day = _interval_day(item)
        if item.get("error"):
            failed.append(day)
            continue
        outputs = item.get("outputs") or {}
        if "px" not in outputs or "wx" not in outputs:
            rows.append({"date": day, "data_pixels": 0})
            continue
        data, (cloud, usable, water, turb_valid) = _counts(outputs["px"], 4)
        row: dict[str, Any] = {
            "date": day,
            "data_pixels": data,
            "cloud_pixels": cloud,
            "usable_pixels": usable,
            "water_pixels": water,
            "turbidity_valid_pixels": turb_valid,
        }
        for index, name in enumerate(WATER_VALUE_BANDS):
            denominator = turb_valid if name == "turbidity" else water
            row[f"w_{name}"] = _masked_mean(outputs["wx"], index, data, denominator)
        rows.append(row)
    for item in payload.get("failedIntervals", []) or []:
        failed.append(_interval_day(item))
    return rows, failed


def parse_riparian_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for item in payload.get("data", []):
        day = _interval_day(item)
        if item.get("error"):
            failed.append(day)
            continue
        outputs = item.get("outputs") or {}
        if "px" not in outputs or "rx" not in outputs:
            rows.append({"date": day, "r_data_pixels": 0})
            continue
        data, (cloud, usable, land) = _counts(outputs["px"], 3)
        rows.append(
            {
                "date": day,
                "r_data_pixels": data,
                "r_cloud_pixels": cloud,
                "r_usable_pixels": usable,
                "r_land_pixels": land,
                "r_ndvi": _masked_mean(outputs["rx"], 0, data, land),
            }
        )
    for item in payload.get("failedIntervals", []) or []:
        failed.append(_interval_day(item))
    return rows, failed


def _in_domain(value: float | None, lo: float, hi: float, *, lo_open: bool = False) -> bool:
    if value is None or not math.isfinite(value):
        return False
    return (value > lo if lo_open else value >= lo) and value <= hi


def classify_water(row: dict[str, Any], s: S2Settings) -> dict[str, Any] | None:
    """Raw water row -> observation values + quality_flag. None = not an observation
    (no data pixels: the corridor was outside the acquisition's footprint)."""
    data = int(row.get("data_pixels", 0))
    if data <= 0:
        return None
    cloud = int(row["cloud_pixels"])
    usable = int(row["usable_pixels"])
    water = int(row["water_pixels"])
    cloud_fraction = cloud / data
    out: dict[str, Any] = {
        "cloud_fraction": cloud_fraction,
        # Unknown, not zero, when nothing in the corridor was usable.
        "water_pixel_count": water if usable > 0 else None,
        "turbidity_proxy": None,
        "ndci": None,
        "mndwi": None,
    }
    if usable == 0 or cloud_fraction > s.max_cloud_fraction:
        out["quality_flag"] = "CLOUD"
        return out
    if water < s.min_water_pixels:
        out["quality_flag"] = "NO_WATER_PIXELS"
        return out

    turbidity, ndci, mndwi = row.get("w_turbidity"), row.get("w_ndci"), row.get("w_mndwi")
    valid_fraction = int(row["turbidity_valid_pixels"]) / water
    in_range = (
        valid_fraction >= s.min_turbidity_valid_fraction
        and _in_domain(mndwi, 0.0, 1.0, lo_open=True)
        and _in_domain(ndci, -1.0, 1.0)
        and _in_domain(turbidity, 0.0, math.inf)
        and _in_domain(row.get("w_B04"), 0.0, math.inf)
    )
    if not in_range:
        out["quality_flag"] = "OUT_OF_RANGE"
        return out
    out.update({"turbidity_proxy": turbidity, "ndci": ndci, "mndwi": mndwi, "quality_flag": "OK"})
    return out


def classify_riparian(row: dict[str, Any] | None, s: S2Settings) -> dict[str, Any]:
    """Raw riparian row -> (riparian_ndvi, riparian_flag). A missing row (the riparian
    buffer returned no interval for a day the water corridor did) is no land observed."""
    if row is None or int(row.get("r_data_pixels", 0)) <= 0:
        return {"riparian_ndvi": None, "riparian_flag": "NO_LAND_PIXELS"}
    data = int(row["r_data_pixels"])
    if int(row["r_usable_pixels"]) == 0 or int(row["r_cloud_pixels"]) / data > s.max_cloud_fraction:
        return {"riparian_ndvi": None, "riparian_flag": "CLOUD"}
    if int(row["r_land_pixels"]) < s.riparian_min_pixels:
        return {"riparian_ndvi": None, "riparian_flag": "NO_LAND_PIXELS"}
    ndvi = row.get("r_ndvi")
    if not _in_domain(ndvi, -1.0, 1.0):
        return {"riparian_ndvi": None, "riparian_flag": "OUT_OF_RANGE"}
    return {"riparian_ndvi": ndvi, "riparian_flag": "OK"}


@dataclass
class ReachParse:
    rows: list[dict[str, Any]]
    returned: int = 0
    dropped: Counter[str] = field(default_factory=Counter)


def build_reach_rows(
    reach_id: str,
    water_rows: list[dict[str, Any]],
    riparian_rows: list[dict[str, Any]] | None,
    failed_days: list[str],
    s: S2Settings,
) -> ReachParse:
    """Classify and merge one reach's acquisitions. `riparian_rows=None` means the
    riparian request was not made (flag NOT_REQUESTED), which is different from made
    and empty."""
    result = ReachParse(rows=[], returned=len(water_rows) + len(failed_days))
    if failed_days:
        result.dropped["FAILED_INTERVAL"] += len(failed_days)
    riparian_by_day = {r["date"]: r for r in (riparian_rows or [])}
    seen: set[str] = set()
    for raw in sorted(water_rows, key=lambda r: r["date"]):
        day = raw["date"]
        if day in seen:
            result.dropped["DUPLICATE_DATE"] += 1
            continue
        seen.add(day)
        water = classify_water(raw, s)
        if water is None:
            result.dropped["NO_DATA_COVERAGE"] += 1
            continue
        if riparian_rows is None:
            riparian = {"riparian_ndvi": None, "riparian_flag": "NOT_REQUESTED"}
        else:
            riparian = classify_riparian(riparian_by_day.get(day), s)
        result.rows.append(
            {
                "reach_id": reach_id,
                "obs_date": date.fromisoformat(day),
                "source": SOURCE,
                **water,
                **riparian,
                # QA extras - kept in the interim parquet, not in the observations table.
                "data_pixels": raw["data_pixels"],
                "usable_pixels": raw["usable_pixels"],
                "turbidity_valid_pixels": raw["turbidity_valid_pixels"],
                **{f"w_{b}": raw.get(f"w_{b}") for b in WATER_VALUE_BANDS[3:]},
            }
        )
    return result


# ---------------------------------------------------------------------------
# network: one request, retried with exponential backoff
# ---------------------------------------------------------------------------
def sh_config(client_id: str, client_secret: str) -> Any:
    from sentinelhub import SHConfig  # type: ignore[attr-defined]

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = SH_BASE_URL
    config.sh_token_url = SH_TOKEN_URL
    # Retries are ours (below), with backoff and logging; don't stack the library's.
    config.max_download_attempts = 1
    return config


def _status_code(exc: BaseException) -> int | None:
    for candidate in (exc, getattr(exc, "request_exception", None), exc.__cause__):
        response = getattr(candidate, "response", None)
        if response is not None and getattr(response, "status_code", None) is not None:
            return int(response.status_code)
    return None


def is_retryable(exc: BaseException) -> bool:
    """429 and 5xx and connection trouble retry; any other 4xx is our fault and won't
    get better by asking again."""
    code = _status_code(exc)
    if code is None:
        return True
    return code == 429 or code >= 500


def with_backoff(
    fn: Callable[[], Any],
    *,
    max_attempts: int,
    base_s: float,
    max_s: float,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call `fn` until it succeeds. Waits base*2^n (+ jitter, capped) between attempts."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified right here
            if not is_retryable(exc):
                raise NonRetryableRequestError(
                    f"{label}: HTTP {_status_code(exc)} - not retrying: {exc}"
                ) from exc
            if attempt == max_attempts:
                raise
            wait = min(max_s, base_s * 2 ** (attempt - 1)) * (1 + 0.25 * random.random())
            log.warning(
                "l1.request_retry",
                request=label,
                attempt=attempt,
                wait_s=round(wait, 1),
                status=_status_code(exc),
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            sleep(wait)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Task:
    reach_id: str
    kind: str  # WATER | RIPARIAN
    start: date
    end: date
    line: LineString

    def half_width(self, s: S2Settings) -> float:
        return s.water_half_width_m if self.kind == WATER else s.riparian_half_width_m


class StatisticalFetcher:
    """Cache-first Statistical API access. Knows nothing about quality flags."""

    def __init__(self, s: S2Settings, *, refresh: bool = False) -> None:
        self.s = s
        self.refresh = refresh
        self.cache = DiskCache(CACHE_NAMESPACE)
        self.scripts = {WATER: water_evalscript(s), RIPARIAN: riparian_evalscript(s)}
        self.hashes = {k: evalscript_hash(v) for k, v in self.scripts.items()}
        self._config: Any = None
        self._client: Any = None

    # -- keys ---------------------------------------------------------------
    def key(self, task: Task) -> dict[str, Any]:
        """(reach_id, date_range, evalscript_hash) plus what else would change the answer."""
        return {
            "stage": "l1",
            "kind": task.kind,
            "reach_id": task.reach_id,
            "start": task.start.isoformat(),
            "end": task.end.isoformat(),
            "evalscript_hash": self.hashes[task.kind],
            "geometry_hash": geometry_hash(task.line, task.half_width(self.s)),
            "collection": self.s.collection,
            "resolution_m": self.s.resolution_m,
            "aggregation": "P1D",
        }

    @staticmethod
    def slug(task: Task) -> str:
        return f"l1-{task.kind}_{task.reach_id}_{task.start.isoformat()}_{task.end.isoformat()}"

    def cached(self, task: Task) -> dict[str, Any] | None:
        if self.refresh:
            return None
        entry = self.cache.get(self.key(task), slug=self.slug(task))
        if entry is None:
            return None
        payload: dict[str, Any] = entry.payload
        return payload

    # -- network ------------------------------------------------------------
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from sentinelhub import SentinelHubDownloadClient  # type: ignore[attr-defined]

        client_id, client_secret = get_settings().require_sentinel_hub()
        self._config = sh_config(client_id, client_secret)
        self._client = SentinelHubDownloadClient(config=self._config)

    def _request(self, task: Task) -> Any:
        from sentinelhub import (  # type: ignore[attr-defined]
            DataCollection,
            SentinelHubStatistical,
        )

        return SentinelHubStatistical(
            aggregation=SentinelHubStatistical.aggregation(
                evalscript=self.scripts[task.kind],
                time_interval=(task.start.isoformat(), task.end.isoformat()),
                aggregation_interval="P1D",
                resolution=(self.s.resolution_m, self.s.resolution_m),
            ),
            input_data=[
                SentinelHubStatistical.input_data(
                    DataCollection.SENTINEL2_L2A.define_from("s2l2a_cdse", service_url=SH_BASE_URL)
                )
            ],
            geometry=corridor(task.line, task.half_width(self.s)),
            config=self._config,
        )

    def fetch(self, task: Task) -> tuple[dict[str, Any], float | None]:
        """Network fetch -> cache -> (payload, processing units spent or None)."""
        self._ensure_client()
        request = self._request(task)

        def once() -> Any:
            return self._client.download(request.download_list, decode_data=False)[0]

        response = with_backoff(
            once,
            max_attempts=self.s.max_attempts,
            base_s=self.s.backoff_base_s,
            max_s=self.s.backoff_max_s,
            label=self.slug(task),
        )
        payload: dict[str, Any] = response.decode()
        if not isinstance(payload, dict) or "data" not in payload:
            raise RuntimeError(f"{self.slug(task)}: Statistical API returned no `data` block")
        header = (response.headers or {}).get("x-processingunits-spent")
        pu = _num(header)
        self.cache.put(
            self.key(task),
            payload,
            slug=self.slug(task),
            url=STATISTICS_URL,
            source="Copernicus Sentinel-2 L2A via Sentinel Hub Statistical API (CDSE)",
            extra_meta={"processing_units": pu, "kind": task.kind},
        )
        return payload, pu

    # -- ledger -------------------------------------------------------------
    def all_time_pu(self) -> float:
        return sum(
            float(m.get("processing_units") or 0.0) for m in self.cache.iter_meta("l1-*.meta.json")
        )

    def pu_rates(self, plan: RunPlan | None = None) -> dict[str, dict[str, Any]]:
        """PU per chunk-day by request kind, spend-weighted over the ledger (total PU /
        total days requested), else FALLBACK_PU_PER_YEAR / 365.

        Spend-weighted, not a mean of per-request rates: on 2026-09-21 two cheap
        pre-2018 probe years (one satellite, fewer acquisitions) pulled a plain mean
        ~25% below the 2024-2025 cost, and the first full-run estimate undershot."""
        out: dict[str, dict[str, Any]] = {}
        metas = self.cache.iter_meta("l1-*.meta.json")
        for kind in (WATER, RIPARIAN):
            spent, days = 0.0, 0
            samples = 0
            for m in metas:
                if m.get("kind") != kind or m.get("processing_units") is None:
                    continue
                span = date.fromisoformat(m["params"]["end"]) - date.fromisoformat(
                    m["params"]["start"]
                )
                spent += float(m["processing_units"])
                days += span.days + 1
                samples += 1
            if samples:
                out[kind] = {"pu_per_day": spent / days, "source": "ledger", "samples": samples}
            else:
                out[kind] = {
                    "pu_per_day": FALLBACK_PU_PER_YEAR[kind] / 365,
                    "source": "fallback",
                    "samples": 0,
                }
        return out

    @staticmethod
    def task_pu(task: Task, rates: dict[str, dict[str, Any]]) -> float:
        return float(rates[task.kind]["pu_per_day"]) * ((task.end - task.start).days + 1)


# Used only when the ledger has no sample yet: measured on 2026-09-21 as 3.41 PU per
# reach-year for the water request (8 input bands) and 2.43 PU per reach-year for the
# riparian request (6 bands). Every reach corridor is far below Sentinel Hub's minimum
# billed area, so a request costs (minimum area) x (bands / 3) x (acquisitions): the rate
# is per acquisition, uniform across reaches, and a window costs its share of a year.
FALLBACK_PU_PER_YEAR = {WATER: 3.4, RIPARIAN: 2.43}


# ---------------------------------------------------------------------------
# run plan (config/sentinel2.yaml -> plan)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunPlan:
    water_years: tuple[int, ...]
    water_observable_only: bool
    riparian_years: tuple[int, ...]
    riparian_window: tuple[str, str]  # ("MM-DD", "MM-DD"), inclusive
    probe_years: tuple[int, ...]

    @classmethod
    def from_config(cls, s2: dict[str, Any]) -> RunPlan:
        p = s2["plan"]
        return cls(
            water_years=tuple(int(y) for y in p["water"]["years"]),
            water_observable_only=bool(p["water"]["observable_only"]),
            riparian_years=tuple(int(y) for y in p["riparian"]["years"]),
            riparian_window=(str(p["riparian"]["window"][0]), str(p["riparian"]["window"][1])),
            probe_years=tuple(int(y) for y in p["pre_2018_probe"]["years"]),
        )

    def window_days(self) -> int:
        lo, hi = (date.fromisoformat(f"2001-{md}") for md in self.riparian_window)
        return (hi - lo).days + 1


def water_chunk(year: int, today: date) -> tuple[date, date] | None:
    lo, hi = date(year, 1, 1), min(date(year, 12, 31), today)
    return (lo, hi) if lo <= hi else None


def riparian_chunk(year: int, window: tuple[str, str], today: date) -> tuple[date, date] | None:
    lo = date.fromisoformat(f"{year}-{window[0]}")
    hi = date.fromisoformat(f"{year}-{window[1]}")
    return (lo, hi) if hi <= today else None  # an unfinished window is not requested


def plan_tasks(
    reaches: list[dict[str, Any]], plan: RunPlan, today: date, *, riparian: bool = True
) -> tuple[list[Task], dict[str, int]]:
    """Tasks in run order: water for observable reaches year by year in plan order, then
    the riparian windows. Returns (tasks, skipped counts by reason)."""
    skipped: Counter[str] = Counter()
    water_reaches = reaches
    if plan.water_observable_only:
        water_reaches = [r for r in reaches if r["observable"] is True]
        skipped["WATER_UNOBSERVABLE_REACH"] = len(reaches) - len(water_reaches)
    tasks: list[Task] = []
    for year in plan.water_years:
        chunk = water_chunk(year, today)
        if chunk is None:
            skipped["WATER_YEAR_IN_FUTURE"] += len(water_reaches)
            continue
        tasks += [Task(r["reach_id"], WATER, *chunk, r["line"]) for r in water_reaches]
    if riparian:
        for year in plan.riparian_years:
            chunk = riparian_chunk(year, plan.riparian_window, today)
            if chunk is None:
                skipped["RIPARIAN_WINDOW_NOT_COMPLETE"] += len(reaches)
                continue
            tasks += [Task(r["reach_id"], RIPARIAN, *chunk, r["line"]) for r in reaches]
    return tasks, dict(skipped)


def summarise_riparian_window(
    reach_id: str, lo: date, hi: date, rows: list[dict[str, Any]], s: S2Settings
) -> dict[str, Any]:
    """One reach's midsummer window -> one row: median NDVI over the window's clear
    acquisitions. No clear acquisition -> NULL with a flag, never a fill."""
    classified = [
        classify_riparian(r, s)
        for r in sorted(rows, key=lambda r: r["date"])
        if int(r.get("r_data_pixels", 0)) > 0
    ]
    ok = [float(c["riparian_ndvi"]) for c in classified if c["riparian_flag"] == "OK"]
    if ok:
        flag = "OK"
    elif classified:
        flag = "NO_CLEAR_ACQUISITION"
    else:
        flag = "NO_ACQUISITION"
    return {
        "reach_id": reach_id,
        "year": lo.year,
        "window_start": lo,
        "window_end": hi,
        "riparian_ndvi_median": float(pd.Series(ok).median()) if ok else None,
        "n_acquisitions": len(classified),
        "n_clear": len(ok),
        "flag": flag,
        "flag_counts": dict(Counter(c["riparian_flag"] for c in classified)),
    }


# ---------------------------------------------------------------------------
# reaches in, rows out
# ---------------------------------------------------------------------------
def load_reaches(city: str) -> list[dict[str, Any]]:
    """Reaches from PostGIS, observable first (by median water pixels)."""
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        rows = session.execute(
            text(
                "SELECT reach_id, observable, median_water_pixels, ST_AsGeoJSON(geom) AS geom "
                "FROM reaches WHERE city = :city "
                "ORDER BY observable DESC NULLS LAST, median_water_pixels DESC NULLS LAST, "
                "reach_id"
            ),
            {"city": city},
        ).mappings()
        reaches = [
            {
                "reach_id": r["reach_id"],
                "observable": r["observable"],
                "median_water_pixels": r["median_water_pixels"],
                "line": shape(json.loads(r["geom"])),
            }
            for r in rows
        ]
    if not reaches:
        raise RuntimeError(f"No reaches in PostGIS for city={city}. Run L0 first.")
    return reaches


def write_observations(rows: pd.DataFrame, reach_ids: list[str], start: date, end: date) -> int:
    """Replace the S2 rows for these reaches and dates. Delete-then-insert rather than
    upsert, so an acquisition that is no longer an observation (e.g. a changed rule)
    does not linger."""
    from sqlalchemy import text

    from core.db import bulk_upsert, session_scope

    with session_scope() as session:
        session.execute(
            text(
                "DELETE FROM observations WHERE source = :source AND reach_id = ANY(:ids) "
                "AND obs_date BETWEEN :start AND :end"
            ),
            {"source": SOURCE, "ids": reach_ids, "start": start, "end": end},
        )
    return bulk_upsert(
        rows, "observations", ["reach_id", "obs_date", "source"], columns=OBS_COLUMNS
    )


def write_riparian(rows: pd.DataFrame) -> int:
    from core.db import bulk_upsert

    return bulk_upsert(rows, "riparian_ndvi_window", ["reach_id", "year"])


def write_interim(frame: pd.DataFrame, city: str, name: str, keys: list[str]) -> Any:
    """Merge into data/interim/<name>_<city>.parquet, replacing previous rows that share
    `keys` with the new frame (a partial run must not erase what an earlier run wrote)."""
    path = INTERIM_DIR / f"{name}_{city}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = pd.read_parquet(path)
        if set(keys) <= set(previous.columns):
            new_keys = frame[keys].drop_duplicates()
            merged = previous.merge(new_keys, on=keys, how="left", indicator=True)
            previous = previous[(merged["_merge"] == "left_only").to_numpy()]
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.sort_values(keys).to_parquet(path, index=False)
    return path


def fetch_observations(
    city: str,
    *,
    reach_ids: Sequence[str] | None = None,
    limit: int | None = None,
    max_pu: float | None = None,
    riparian: bool = True,
    estimate_only: bool = False,
    write_db: bool = True,
    refresh: bool = False,
    today: date | None = None,
) -> dict[str, Any]:
    """The planned L1 run (config/sentinel2.yaml -> plan). See the module docstring."""
    s2 = load_config("sentinel2")
    s = load_settings(city)
    plan = RunPlan.from_config(s2)
    today = today or datetime.now(UTC).date()

    reaches = load_reaches(city)
    if reach_ids:
        wanted = set(reach_ids)
        reaches = [r for r in reaches if r["reach_id"] in wanted]
        missing = wanted - {r["reach_id"] for r in reaches}
        if missing:
            raise ValueError(f"Unknown reach ids for {city}: {sorted(missing)}")
    if limit:
        reaches = reaches[:limit]

    fetcher = StatisticalFetcher(s, refresh=refresh)
    all_tasks, skipped = plan_tasks(reaches, plan, today, riparian=riparian)
    uncached = [t for t in all_tasks if fetcher.cached(t) is None]
    rates = fetcher.pu_rates(plan)
    estimate_by: dict[str, float] = {}
    for t in uncached:
        k = f"{t.kind}:{t.start.year}"
        estimate_by[k] = estimate_by.get(k, 0.0) + fetcher.task_pu(t, rates)
    estimate = float(sum(estimate_by.values()))
    all_time = fetcher.all_time_pu()
    plan_out = {
        "reaches": len(reaches),
        "water_reaches": len({t.reach_id for t in all_tasks if t.kind == WATER}),
        "riparian_reaches": len({t.reach_id for t in all_tasks if t.kind == RIPARIAN}),
        "water_years": list(plan.water_years),
        "riparian_window": list(plan.riparian_window),
        "requests_total": len(all_tasks),
        "requests_cached": len(all_tasks) - len(uncached),
        "requests_to_fetch": len(uncached),
        "requests_to_fetch_by_kind": dict(Counter(t.kind for t in uncached)),
        "skipped": skipped,
        "pu_rates": rates,
        "pu_estimate": round(estimate, 1),
        "pu_estimate_by_kind_year": {k: round(v, 1) for k, v in sorted(estimate_by.items())},
        "pu_all_time_l1": round(all_time, 2),
    }
    log.info("l1.plan", city=city, **{k: v for k, v in plan_out.items() if k != "pu_rates"})
    if estimate_only:
        return plan_out

    obs_flags: Counter[str] = Counter()
    rip_flags: Counter[str] = Counter()
    water_frames: list[pd.DataFrame] = []
    rip_rows: list[dict[str, Any]] = []
    pu_run = 0.0
    pu_missing_header = 0
    network_calls = 0
    failed: list[str] = []
    step = s.batch_size * 2

    with stage(log, "l1_satellite", city=city) as counters:
        rows_in = 0
        for b in range(0, len(all_tasks), step):
            tasks = all_tasks[b : b + step]
            to_fetch = [t for t in tasks if fetcher.cached(t) is None]
            batch_estimate = sum(fetcher.task_pu(t, rates) for t in to_fetch)
            if max_pu is not None and pu_run + batch_estimate > max_pu:
                raise BudgetExceeded(
                    f"Stopping before task {b + 1}/{len(all_tasks)}: {pu_run:.1f} PU spent this "
                    f"run + ~{batch_estimate:.1f} estimated > --max-pu {max_pu:g}. Everything "
                    "processed so far is written; re-run with a higher cap to continue - cached "
                    "chunks cost nothing."
                )
            payloads: dict[Task, dict[str, Any]] = {}
            for t in tasks:
                cached = fetcher.cached(t)
                if cached is not None:
                    payloads[t] = cached

            def run(task: Task) -> tuple[Task, dict[str, Any] | None, float | None, str]:
                try:
                    payload, pu = fetcher.fetch(task)
                    return task, payload, pu, ""
                except NonRetryableRequestError:
                    raise
                except Exception as exc:  # noqa: BLE001 - counted and raised after the run
                    return task, None, None, f"{type(exc).__name__}: {exc}"

            with ThreadPoolExecutor(max_workers=s.max_threads) as pool:
                for task, payload, pu, error in pool.map(run, to_fetch):
                    network_calls += 1
                    if payload is None:
                        failed.append(f"{fetcher.slug(task)}: {error}")
                        log.error("l1.request_failed", request=fetcher.slug(task), error=error)
                        continue
                    payloads[task] = payload
                    if pu is None:
                        pu_missing_header += 1
                    pu_run += pu or 0.0

            batch_water: dict[tuple[date, date], list[pd.DataFrame]] = {}
            batch_rip: list[dict[str, Any]] = []
            for t in tasks:
                if t not in payloads:
                    counters.drop(1, f"{t.kind.upper()}_CHUNK_FAILED")
                    continue
                if t.kind == WATER:
                    w, f = parse_water_payload(payloads[t])
                    parsed = build_reach_rows(t.reach_id, w, None, f, s)
                    rows_in += parsed.returned
                    for reason, n in parsed.dropped.items():
                        counters.drop(n, reason)
                    frame = pd.DataFrame(parsed.rows)
                    if not frame.empty:
                        obs_flags.update(frame["quality_flag"])
                    batch_water.setdefault((t.start, t.end), []).append(
                        frame if not frame.empty else pd.DataFrame(columns=OBS_COLUMNS)
                    )
                else:
                    r, _ = parse_riparian_payload(payloads[t])
                    row = summarise_riparian_window(t.reach_id, t.start, t.end, r, s)
                    rip_flags[row["flag"]] += 1
                    batch_rip.append(row)

            for (lo, hi), frames in batch_water.items():
                chunk = pd.concat([f for f in frames if not f.empty] or frames, ignore_index=True)
                ids = sorted({t.reach_id for t in tasks if t.kind == WATER and t.start == lo})
                if write_db:
                    # Delete-then-insert per chunk: a reach-year with no acquisitions left
                    # after re-classification must not keep stale rows.
                    write_observations(chunk, ids, lo, hi)
                if not chunk.empty:
                    water_frames.append(chunk)
            if batch_rip:
                if write_db:
                    write_riparian(pd.DataFrame(batch_rip))
                rip_rows += batch_rip

            log.info(
                "l1.batch",
                tasks_done=min(b + step, len(all_tasks)),
                of=len(all_tasks),
                network_calls=network_calls,
                pu_batch_estimate=round(batch_estimate, 2),
                pu_run=round(pu_run, 2),
                pu_all_time=round(fetcher.all_time_pu(), 2),
            )

        if not water_frames and not rip_rows:
            raise RuntimeError(
                f"L1 produced nothing for {city} ({len(failed)} failed requests). "
                "Refusing to report success on nothing."
            )
        interim = {}
        if water_frames:
            water = pd.concat(water_frames, ignore_index=True)
            water["year"] = pd.to_datetime(water["obs_date"]).dt.year
            interim["water"] = str(write_interim(water, city, "s2", ["reach_id", "year"]))
        if rip_rows:
            interim["riparian"] = str(
                write_interim(pd.DataFrame(rip_rows), city, "s2_riparian", ["reach_id", "year"])
            )
        counters.record(
            rows_in=rows_in,
            rows_out=sum(len(f) for f in water_frames),
            quality_flags=dict(obs_flags),
            riparian_window_flags=dict(rip_flags),
            network_calls=network_calls,
            pu_run=round(pu_run, 2),
            pu_all_time=round(fetcher.all_time_pu(), 2),
            pu_missing_header=pu_missing_header,
            failed_requests=len(failed),
        )

    if pu_missing_header:
        log.warning(
            "l1.pu_header_missing",
            responses=pu_missing_header,
            note="PU totals undercount by these responses",
        )
    if failed:
        raise RuntimeError(
            f"{len(failed)} Statistical API requests failed after retries; their chunks were "
            f"not written. First: {failed[0]}"
        )
    return {
        **plan_out,
        "observation_rows": sum(len(f) for f in water_frames),
        "quality_flags": dict(obs_flags),
        "riparian_windows": len(rip_rows),
        "riparian_window_flags": dict(rip_flags),
        "pu_run": round(pu_run, 2),
        "pu_all_time_l1": round(fetcher.all_time_pu(), 2),
        "interim": interim,
    }


def probe_pre_2018(city: str, *, reach_id: str | None = None) -> dict[str, Any]:
    """Fetch the plan's pre-2018 years for ONE reach (default: the most observable) and
    report usable rows per year. Written to nothing - the decision to extend the series
    backwards is made on this report."""
    s2 = load_config("sentinel2")
    s = load_settings(city)
    plan = RunPlan.from_config(s2)
    reaches = [r for r in load_reaches(city) if r["observable"] is True]
    reach = (
        next((r for r in reaches if r["reach_id"] == reach_id), None) if reach_id else reaches[0]
    )
    if reach is None:
        raise ValueError(f"{reach_id} is not an observable {city} reach")
    fetcher = StatisticalFetcher(s)
    pu = 0.0
    years: dict[str, Any] = {}
    for year in plan.probe_years:
        task = Task(reach["reach_id"], WATER, date(year, 1, 1), date(year, 12, 31), reach["line"])
        payload = fetcher.cached(task)
        if payload is None:
            payload, spent = fetcher.fetch(task)
            pu += spent or 0.0
        w, f = parse_water_payload(payload)
        parsed = build_reach_rows(task.reach_id, w, None, f, s)
        flags = Counter(r["quality_flag"] for r in parsed.rows)
        years[str(year)] = {
            "acquisitions_returned": parsed.returned,
            "rows": len(parsed.rows),
            "usable_OK": flags.get("OK", 0),
            "quality_flags": dict(flags),
            "dropped": dict(parsed.dropped),
        }
    return {"reach_id": reach["reach_id"], "years": years, "pu_spent": round(pu, 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L1 Sentinel-2 observations")
    parser.add_argument("--city", default="coimbra")
    parser.add_argument("--reach", action="append", dest="reach_ids", help="repeatable")
    parser.add_argument("--limit", type=int, default=None, help="first N reaches")
    parser.add_argument("--max-pu", type=float, default=None, help="stop before exceeding")
    parser.add_argument("--no-riparian", action="store_true", help="water requests only")
    parser.add_argument("--estimate", action="store_true", help="print the PU plan and exit")
    parser.add_argument("--probe-pre-2018", action="store_true", help="one reach, 2016-2017")
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache (spends PUs)")
    args = parser.parse_args(argv)

    if args.probe_pre_2018:
        rid = args.reach_ids[0] if args.reach_ids else None
        print(json.dumps(probe_pre_2018(args.city, reach_id=rid), indent=2, default=str))
        return 0
    summary = fetch_observations(
        args.city,
        reach_ids=args.reach_ids,
        limit=args.limit,
        max_pu=args.max_pu,
        riparian=not args.no_riparian,
        estimate_only=args.estimate,
        write_db=not args.no_db,
        refresh=args.refresh,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
