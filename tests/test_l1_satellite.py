"""L1 quality-flag tests (CLAUDE.md Testing #3): missing data never becomes zero and
never becomes an unlabelled fill.

Payloads are hand-built in the exact shape the Statistical API returns (a cached
response from data/raw/sentinelhub/ has the same structure), so every expected number
below can be checked with a calculator.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from pipeline.l1_satellite import (
    NonRetryableRequestError,
    S2Settings,
    build_reach_rows,
    classify_water,
    parse_riparian_payload,
    parse_water_payload,
    riparian_evalscript,
    water_evalscript,
    with_backoff,
    year_chunks,
)

S = S2Settings(
    bands=("B02", "B03", "B04", "B05", "B08", "B11", "SCL"),
    scl_cloud=(3, 8, 9, 10),
    scl_invalid=(0, 1),
    scl_water=6,
    water_half_width_m=15,
    riparian_half_width_m=30,
    nechad_a=228.1,
    nechad_c=0.1641,
    max_rho_fraction_of_c=0.9,
    min_water_pixels=5,
    max_cloud_fraction=0.2,
    min_turbidity_valid_fraction=0.5,
    riparian_min_pixels=10,
)


def _stats(mean: float | str, samples: int, nodata: int) -> dict[str, Any]:
    return {"stats": {"mean": mean, "sampleCount": samples, "noDataCount": nodata}}


def _water_item(
    day: str,
    *,
    samples: int = 200,
    nodata: int = 100,
    cloud: float = 0.0,
    usable: float = 1.0,
    water: float = 0.5,
    turb_valid: float = 0.5,
    wx: list[float] | None = None,
) -> dict[str, Any]:
    """One interval. Means are over the (samples - nodata) data pixels, as the API does."""
    wx = wx if wx is not None else [0.2, 0.01, 5.0, 0.01, 0.02, 0.01, 0.015, 0.01, 0.005]
    return {
        "interval": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"},
        "outputs": {
            "px": {
                "bands": {
                    f"B{i}": _stats(v, samples, nodata)
                    for i, v in enumerate([cloud, usable, water, turb_valid])
                }
            },
            "wx": {"bands": {f"B{i}": _stats(v, samples, nodata) for i, v in enumerate(wx)}},
        },
    }


def _riparian_item(
    day: str, *, cloud: float = 0.0, usable: float = 1.0, land: float = 0.6, rx: float = 0.3
) -> dict[str, Any]:
    return {
        "interval": {"from": f"{day}T00:00:00Z"},
        "outputs": {
            "px": {
                "bands": {f"B{i}": _stats(v, 400, 100) for i, v in enumerate([cloud, usable, land])}
            },
            "rx": {"bands": {"B0": _stats(rx, 400, 100)}},
        },
    }


# ---------------------------------------------------------------------------
# masked means are reconstructed exactly
# ---------------------------------------------------------------------------
def test_masked_mean_reconstruction_is_exact() -> None:
    # 100 data pixels, 50 water. mean(water * mndwi) over data pixels = 0.2, so the mean
    # MNDWI over water pixels is 0.2 * 100 / 50 = 0.4.
    rows, failed = parse_water_payload({"data": [_water_item("2023-06-03")]})
    assert failed == []
    row = rows[0]
    assert row["data_pixels"] == 100
    assert row["water_pixels"] == 50
    assert row["w_mndwi"] == pytest.approx(0.4)
    assert row["w_ndci"] == pytest.approx(0.02)
    # turbidity is divided by the turbidity-valid count (also 50 here): 5.0 * 100 / 50.
    assert row["w_turbidity"] == pytest.approx(10.0)


def test_clear_date_with_water_is_ok_and_carries_values() -> None:
    rows, _ = parse_water_payload({"data": [_water_item("2023-06-03")]})
    out = classify_water(rows[0], S)
    assert out is not None
    assert out["quality_flag"] == "OK"
    assert out["water_pixel_count"] == 50
    assert out["cloud_fraction"] == 0.0
    assert out["mndwi"] == pytest.approx(0.4)
    assert out["turbidity_proxy"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# the three missing-data flags - values NULL, never zero
# ---------------------------------------------------------------------------
def test_fully_clouded_corridor_is_cloud_with_null_count_not_zero() -> None:
    rows, _ = parse_water_payload(
        {
            "data": [
                _water_item(
                    "2023-01-14", cloud=1.0, usable=0.0, water=0.0, turb_valid=0.0, wx=[0.0] * 9
                )
            ]
        }
    )
    out = classify_water(rows[0], S)
    assert out is not None
    assert out["quality_flag"] == "CLOUD"
    # Nothing was usable: the count is unknown. 0 would claim "saw no water".
    assert out["water_pixel_count"] is None
    for column in ("turbidity_proxy", "ndci", "mndwi"):
        assert out[column] is None


def test_partly_cloudy_above_threshold_is_cloud_but_keeps_measured_count() -> None:
    rows, _ = parse_water_payload(
        {"data": [_water_item("2023-02-01", cloud=0.3, usable=0.7, water=0.4, turb_valid=0.4)]}
    )
    out = classify_water(rows[0], S)
    assert out is not None
    assert out["quality_flag"] == "CLOUD"
    assert out["cloud_fraction"] == pytest.approx(0.3)
    assert out["water_pixel_count"] == 40  # measured over the usable pixels - kept
    assert out["turbidity_proxy"] is None


def test_clear_date_without_water_is_no_water_pixels_with_real_zero_count() -> None:
    rows, _ = parse_water_payload(
        {"data": [_water_item("2023-07-01", water=0.0, turb_valid=0.0, wx=[0.0] * 9)]}
    )
    out = classify_water(rows[0], S)
    assert out is not None
    assert out["quality_flag"] == "NO_WATER_PIXELS"
    # Here 0 IS a measurement: the corridor was clear and no pixel was water.
    assert out["water_pixel_count"] == 0
    assert out["turbidity_proxy"] is None and out["ndci"] is None and out["mndwi"] is None


def test_below_min_water_pixels_is_no_water_pixels() -> None:
    rows, _ = parse_water_payload(
        {"data": [_water_item("2023-07-02", water=0.04, turb_valid=0.04)]}
    )
    out = classify_water(rows[0], S)
    assert out is not None
    assert out["water_pixel_count"] == 4
    assert out["quality_flag"] == "NO_WATER_PIXELS"


def test_diverging_nechad_is_out_of_range_not_a_huge_number() -> None:
    # 50 water pixels but only 10 with a valid Nechad value (B04 near C).
    rows, _ = parse_water_payload({"data": [_water_item("2023-08-01", turb_valid=0.1)]})
    out = classify_water(rows[0], S)
    assert out is not None
    assert out["quality_flag"] == "OUT_OF_RANGE"
    assert out["turbidity_proxy"] is None


def test_no_data_pixels_is_not_an_observation() -> None:
    rows, _ = parse_water_payload(
        {
            "data": [
                _water_item(
                    "2023-01-19",
                    samples=200,
                    nodata=200,
                    cloud="NaN",
                    usable="NaN",
                    water="NaN",
                    turb_valid="NaN",
                    wx=["NaN"] * 9,
                )
            ]
        }  # type: ignore[list-item]
    )
    assert classify_water(rows[0], S) is None


def test_nan_strings_never_become_zero_values() -> None:
    rows, _ = parse_water_payload(
        {"data": [_water_item("2023-01-19", wx=["NaN"] * 9)]}  # type: ignore[list-item]
    )
    assert rows[0]["w_mndwi"] is None
    out = classify_water(rows[0], S)
    assert out is not None and out["quality_flag"] == "OUT_OF_RANGE"
    assert out["mndwi"] is None


# ---------------------------------------------------------------------------
# merged rows: every row has a flag, OK rows have values, nothing is zero-filled
# ---------------------------------------------------------------------------
def _reach_rows() -> Any:
    water = {
        "data": [
            _water_item("2023-06-03"),
            _water_item(
                "2023-06-08", cloud=1.0, usable=0.0, water=0.0, turb_valid=0.0, wx=[0.0] * 9
            ),
            _water_item("2023-06-13", water=0.0, turb_valid=0.0, wx=[0.0] * 9),
            _water_item("2023-06-18", samples=200, nodata=200),
            {"interval": {"from": "2023-06-23T00:00:00Z"}, "error": {"type": "EXECUTION_ERROR"}},
        ]
    }
    riparian = {
        "data": [
            _riparian_item("2023-06-03"),
            _riparian_item("2023-06-08", cloud=1.0, usable=0.0, land=0.0, rx=0.0),
            _riparian_item("2023-06-13", land=0.01, rx=0.001),
        ]
    }
    w_rows, failed = parse_water_payload(water)
    r_rows, _ = parse_riparian_payload(riparian)
    return build_reach_rows("TST-0001", w_rows, r_rows, failed, S)


def test_reach_rows_drops_are_counted_with_reasons() -> None:
    parsed = _reach_rows()
    assert parsed.returned == 5
    assert parsed.dropped == {"NO_DATA_COVERAGE": 1, "FAILED_INTERVAL": 1}
    assert [r["quality_flag"] for r in parsed.rows] == ["OK", "CLOUD", "NO_WATER_PIXELS"]


def test_every_non_ok_row_has_null_values_and_every_ok_row_has_values() -> None:
    for row in _reach_rows().rows:
        values = [row["turbidity_proxy"], row["ndci"], row["mndwi"]]
        if row["quality_flag"] == "OK":
            assert all(v is not None for v in values)
        else:
            assert all(v is None for v in values), row
        assert row["quality_flag"] in {"OK", "CLOUD", "NO_WATER_PIXELS", "OUT_OF_RANGE"}
        assert row["obs_date"] == date.fromisoformat(str(row["obs_date"]))


def test_riparian_flags_are_independent_of_water_flags() -> None:
    rows = _reach_rows().rows
    assert rows[0]["riparian_flag"] == "OK"
    # 0.3 * 300 / 180 = 0.5
    assert rows[0]["riparian_ndvi"] == pytest.approx(0.5)
    assert rows[1]["riparian_flag"] == "CLOUD" and rows[1]["riparian_ndvi"] is None
    # 3 land pixels < riparian_min_pixels
    assert rows[2]["riparian_flag"] == "NO_LAND_PIXELS" and rows[2]["riparian_ndvi"] is None


def test_riparian_not_requested_is_labelled_as_such() -> None:
    w_rows, _ = parse_water_payload({"data": [_water_item("2023-06-03")]})
    rows = build_reach_rows("TST-0001", w_rows, None, [], S).rows
    assert rows[0]["riparian_flag"] == "NOT_REQUESTED"
    assert rows[0]["riparian_ndvi"] is None


# ---------------------------------------------------------------------------
# chunking, evalscripts, retry
# ---------------------------------------------------------------------------
def test_year_chunks_cover_the_window_without_overlap() -> None:
    chunks = year_chunks(date(2016, 1, 1), date(2018, 3, 5))
    assert chunks == [
        (date(2016, 1, 1), date(2016, 12, 31)),
        (date(2017, 1, 1), date(2017, 12, 31)),
        (date(2018, 1, 1), date(2018, 3, 5)),
    ]


def test_evalscripts_embed_config_so_the_hash_tracks_it() -> None:
    script = water_evalscript(S)
    assert "228.1" in script and "0.1641" in script
    for band in S.bands:
        assert f'"{band}"' in script
    other = water_evalscript(S2Settings(**{**S.__dict__, "nechad_a": 300.0}))
    assert other != script
    assert "B08" in riparian_evalscript(S)


class _HttpError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.response = type("R", (), {"status_code": code})()


def test_backoff_retries_429_then_succeeds_with_growing_waits() -> None:
    calls, waits = [], []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _HttpError(429)
        return "ok"

    assert (
        with_backoff(flaky, max_attempts=5, base_s=1.0, max_s=60, label="t", sleep=waits.append)
        == "ok"
    )
    assert len(calls) == 3
    assert len(waits) == 2 and 1.0 <= waits[0] < waits[1]


def test_backoff_does_not_retry_auth_failures() -> None:
    calls = []

    def unauthorised() -> None:
        calls.append(1)
        raise _HttpError(401)

    with pytest.raises(NonRetryableRequestError):
        with_backoff(
            unauthorised, max_attempts=5, base_s=0, max_s=0, label="t", sleep=lambda _: None
        )
    assert len(calls) == 1


def test_backoff_gives_up_loudly_after_max_attempts() -> None:
    def down() -> None:
        raise _HttpError(503)

    with pytest.raises(_HttpError):
        with_backoff(down, max_attempts=3, base_s=0, max_s=0, label="t", sleep=lambda _: None)
