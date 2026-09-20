"""GATE: the Sentinel Hub Statistical API returns real numbers for a Coimbra polygon.

    python scripts/smoke_test_sentinelhub.py

This is the project's go/no-go gate (MASTERSPEC section 16, Day 0). If this does not print
per-date numbers, nothing downstream works - fix the credentials before anything else.

ONE Statistical API request, one hard-coded polygon on the Mondego at Coimbra, one
month of dates, bands B03/B04/B05/B08/B11/SCL, MNDWI and NDCI computed server-side in
the evalscript. We receive kilobytes of JSON, never a granule.

The response is cached under data/raw/sentinelhub/ so re-running this costs no
processing units.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Windows consoles default to cp1252; gate output must never come back as mojibake.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.cache import DiskCache, cache_key  # noqa: E402
from core.settings import get_settings  # noqa: E402

SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
SH_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)

# A bounding box on the Mondego channel through central Coimbra (Ponte de Santa Clara
# to Ponte Rainha Santa Isabel). Deliberately a wide main-channel box, not a 200-500m
# reach polygon: the point of the gate is to prove the API path works where water
# pixels certainly exist. The narrow-reach observability question is Day 1's problem.
POLYGON_BBOX = (-8.4450, 40.2030, -8.4150, 40.2120)  # min_lon, min_lat, max_lon, max_lat

START_DATE = "2026-06-01"
END_DATE = "2026-06-30"
BANDS = ["B03", "B04", "B05", "B08", "B11", "SCL"]

# MNDWI = (B03 - B11) / (B03 + B11)   water extent; water where MNDWI > 0
# NDCI  = (B05 - B04) / (B05 + B04)   chlorophyll-a / eutrophication
# NDVI  = (B08 - B04) / (B08 + B04)   riparian vegetation
# SCL classes treated as unusable: 0 no-data, 1 saturated, 3 cloud shadow,
# 8/9 cloud medium+high probability, 10 thin cirrus.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B04", "B05", "B08", "B11", "SCL", "dataMask"] }],
    output: [
      { id: "indices", bands: 3, sampleType: "FLOAT32" },
      { id: "water", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function isUsable(scl) {
  return !(scl === 0 || scl === 1 || scl === 3 || scl === 8 || scl === 9 || scl === 10);
}

function evaluatePixel(s) {
  var mndwi = (s.B03 - s.B11) / (s.B03 + s.B11);
  var ndci  = (s.B05 - s.B04) / (s.B05 + s.B04);
  var ndvi  = (s.B08 - s.B04) / (s.B08 + s.B04);
  var usable = s.dataMask === 1 && isUsable(s.SCL) ? 1 : 0;
  var isWater = usable === 1 && mndwi > 0 ? 1 : 0;
  return {
    indices: [mndwi, ndci, ndvi],
    water: [isWater],
    dataMask: [usable]
  };
}
"""

RULE = "=" * 78


def credentials_error(detail: str) -> str:
    return (
        "\n  SENTINEL HUB AUTHENTICATION FAILED\n"
        f"  {detail}\n\n"
        "  Fix, in order:\n"
        "    1. Register at https://dataspace.copernicus.eu and confirm the email.\n"
        "    2. Create an OAuth client at\n"
        "       https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings\n"
        "    3. cp .env.example .env  and set:\n"
        "         SH_CLIENT_ID=<client id>\n"
        "         SH_CLIENT_SECRET=<client secret, shown once at creation>\n"
        "    4. Re-run:  python scripts/smoke_test_sentinelhub.py\n\n"
        "  Common causes: the secret was regenerated (it is shown only once), the\n"
        "  client was created on services.sentinel-hub.com instead of CDSE, or the\n"
        "  .env values carry stray quotes or whitespace.\n"
    )


def build_request(client_id: str, client_secret: str):  # noqa: ANN201
    from sentinelhub import (
        CRS,
        BBox,
        DataCollection,
        SentinelHubStatistical,
        SHConfig,
    )

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = SH_BASE_URL
    config.sh_token_url = SH_TOKEN_URL

    bbox = BBox(POLYGON_BBOX, crs=CRS.WGS84)

    return SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=EVALSCRIPT,
            time_interval=(START_DATE, END_DATE),
            aggregation_interval="P1D",  # per acquisition date
            resolution=(10, 10),
        ),
        input_data=[
            SentinelHubStatistical.input_data(
                DataCollection.SENTINEL2_L2A.define_from(
                    "s2l2a_cdse", service_url=SH_BASE_URL
                )
            )
        ],
        bbox=bbox,
        config=config,
    )


def run_request(request) -> tuple[dict, float | None]:  # noqa: ANN001
    """Execute the request. Returns (payload, processing_units_or_None).

    PU spend is reported by CDSE in the `x-processingunits-spent` response header, so
    we ask sentinelhub-py for the undecoded response first and fall back if this
    version does not expose headers.
    """
    try:
        responses = request.get_data(decode_data=False)
        raw = responses[0]
        payload = raw.decode() if hasattr(raw, "decode") else raw
        headers = getattr(raw, "headers", {}) or {}
        spent = headers.get("x-processingunits-spent") or headers.get(
            "X-ProcessingUnits-Spent"
        )
        return payload, float(spent) if spent is not None else None
    except TypeError:
        # Older sentinelhub-py: get_data() takes no decode_data argument.
        return request.get_data()[0], None


def summarise(payload: dict) -> list[dict]:
    """Flatten the Statistical API response into one row per acquisition date.

    Every number below comes from the API response. Nothing is estimated here.
    """
    rows: list[dict] = []
    for interval in payload.get("data", []):
        day = interval["interval"]["from"][:10]
        outputs = interval.get("outputs", {})
        if not outputs:
            rows.append({"date": day, "status": "NO_USABLE_PIXELS"})
            continue

        indices = outputs["indices"]["bands"]
        water_band = outputs["water"]["bands"]["B0"]["stats"]
        usable = water_band["sampleCount"] - water_band["noDataCount"]
        water_pixels = round(water_band["mean"] * usable) if usable else 0

        rows.append(
            {
                "date": day,
                "status": "OK" if usable else "NO_USABLE_PIXELS",
                "usable_pixels": usable,
                "water_pixels": water_pixels,
                "mndwi": indices["B0"]["stats"]["mean"],
                "ndci": indices["B1"]["stats"]["mean"],
                "ndvi": indices["B2"]["stats"]["mean"],
            }
        )

    for failed in payload.get("failedIntervals", []):
        rows.append({"date": failed.get("interval", {}).get("from", "?")[:10], "status": "FAILED"})
    return sorted(rows, key=lambda r: r["date"])


def main() -> int:
    print(RULE)
    print("KINGFISHER - SENTINEL HUB STATISTICAL API SMOKE TEST  (Day 0 go/no-go gate)")
    print(RULE)
    print(f"  base url   : {SH_BASE_URL}")
    print(f"  token url  : {SH_TOKEN_URL}")
    print(f"  polygon    : bbox {POLYGON_BBOX}  (Mondego at Coimbra)")
    print(f"  dates      : {START_DATE} .. {END_DATE}   aggregation P1D")
    print(f"  bands      : {', '.join(BANDS)}")
    print("  indices    : MNDWI (B03,B11), NDCI (B05,B04), NDVI (B08,B04)")

    settings = get_settings()
    try:
        client_id, client_secret = settings.require_sentinel_hub()
    except RuntimeError as exc:
        print(credentials_error(str(exc)))
        print(f"{RULE}\n  GATE: FAILED - no credentials. Nothing downstream can run.\n{RULE}")
        return 1

    print(f"  client_id  : {client_id[:8]}...{client_id[-4:]}  (secret loaded, not shown)")

    try:
        from sentinelhub import __version__ as sh_version
    except ImportError:
        print("\n  sentinelhub-py is not installed. Run:  pip install -e \".[dev]\"")
        print(f"{RULE}\n  GATE: FAILED\n{RULE}")
        return 1
    print(f"  sentinelhub: {sh_version}")

    params = {
        "bbox": POLYGON_BBOX,
        "start": START_DATE,
        "end": END_DATE,
        "evalscript_hash": cache_key({"evalscript": EVALSCRIPT}),
        "collection": "sentinel-2-l2a",
    }
    cache = DiskCache("sentinelhub")
    cached = cache.get(params, slug="smoke_coimbra_mondego")

    processing_units: float | None = None
    if cached is not None:
        payload = cached.payload
        print(f"\n  served from cache: {cached.path.relative_to(REPO_ROOT)}"
              f"  (fetched {cached.fetched_at[:19]}Z, 0 processing units)")
    else:
        try:
            request = build_request(client_id, client_secret)
            payload, processing_units = run_request(request)
        except Exception as exc:  # noqa: BLE001 - the gate must explain every failure
            text = f"{type(exc).__name__}: {exc}"
            lowered = text.lower()
            if any(
                token in lowered
                for token in ("401", "invalid_client", "unauthorized", "invalid client", "403")
            ):
                print(credentials_error(text))
            else:
                print(f"\n  REQUEST FAILED\n  {text}\n")
                print("  Check, in order: network access to sh.dataspace.copernicus.eu,")
                print("  the processing-unit quota on the CDSE dashboard, and the evalscript.")
            print(f"{RULE}\n  GATE: FAILED\n{RULE}")
            return 1
        cache.put(
            params,
            payload,
            slug="smoke_coimbra_mondego",
            url=f"{SH_BASE_URL}/api/v1/statistics",
            source="Copernicus Sentinel-2 L2A via Sentinel Hub Statistical API (CDSE)",
        )

    rows = summarise(payload)
    if not rows:
        print("\n  The API responded but returned no intervals at all.")
        print("  Widen the date range, or check the polygon falls inside a Sentinel-2 tile.")
        print(f"{RULE}\n  GATE: FAILED - response contained zero acquisitions.\n{RULE}")
        return 1

    print(f"\n  ACQUISITIONS ({len(rows)} intervals returned)")
    print(f"  {'date':<12} {'status':<18} {'usable px':>10} {'water px':>9}"
          f" {'MNDWI':>8} {'NDCI':>8} {'NDVI':>8}")
    print("  " + "-" * 74)
    for row in rows:
        if row["status"] != "OK":
            print(f"  {row['date']:<12} {row['status']:<18}"
                  f" {'-':>10} {'-':>9} {'-':>8} {'-':>8} {'-':>8}")
            continue
        print(
            f"  {row['date']:<12} {row['status']:<18} {row['usable_pixels']:>10,}"
            f" {row['water_pixels']:>9,} {row['mndwi']:>8.4f} {row['ndci']:>8.4f}"
            f" {row['ndvi']:>8.4f}"
        )

    usable_dates = [r for r in rows if r["status"] == "OK" and r.get("usable_pixels", 0) > 0]
    with_water = [r for r in usable_dates if r["water_pixels"] > 0]

    print("\n  PROCESSING UNITS")
    if processing_units is not None:
        print(f"    spent on this request: {processing_units}")
    elif cached is not None:
        print("    0 - response served from disk cache")
    else:
        print("    not reported in the response headers by this sentinelhub-py version;")
        print("    check the quota at https://shapps.dataspace.copernicus.eu/dashboard/")

    print(f"\n{RULE}")
    if with_water:
        print("  GATE: PASSED - real numbers returned.")
        print(f"    {len(rows)} intervals, {len(usable_dates)} with usable pixels, "
              f"{len(with_water)} with water pixels (MNDWI > 0)")
        best = max(with_water, key=lambda r: r["water_pixels"])
        print(f"    best date {best['date']}: {best['water_pixels']:,} water pixels, "
              f"MNDWI {best['mndwi']:.4f}, NDCI {best['ndci']:.4f}")
    elif usable_dates:
        print("  GATE: PARTIAL - the API works, but no pixel passed the MNDWI water test.")
        print("    Auth and plumbing are fine. Move the polygon onto open channel and re-run.")
    else:
        print("  GATE: FAILED - intervals returned but every one was cloud or no-data.")
        print("    Try a different month before touching credentials.")
    print(RULE)

    debug_path = REPO_ROOT / "data" / "raw" / "sentinelhub" / "smoke_last_response.json"
    debug_path.write_text(json.dumps(payload, indent=2)[:2_000_000], encoding="utf-8")
    print(f"  full response: {debug_path.relative_to(REPO_ROOT)}")
    return 0 if with_water else 1


if __name__ == "__main__":
    raise SystemExit(main())
