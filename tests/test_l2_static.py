"""L2 static-attribute tests on synthetic rasters.

The rasters are tiny GeoTIFFs written into tmp_path with known pixel values, so every
expected fraction and sum is countable by hand. The CLMS adapter is exercised here even
though no CLMS data is on disk yet - that is the point of the adapter interface.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from pipeline.l2_static import (
    AdapterChain,
    AttributeValue,
    CatchmentContext,
    ClmsAdapter,
    LandCoverAdapter,
    RasterSource,
    SourceUnavailable,
    WorldCoverAdapter,
    assemble_rows,
    population,
    reach_riparian_climatology,
    resolve_adapter,
    road_density,
)

# A 10 x 10 raster of 0.01 deg cells from (0, 0.1) to (0.1, 0). Catchment "A" covers the
# left half (50 cells), "B" the top-left quarter (25 cells).
TRANSFORM = from_origin(0.0, 0.1, 0.01, 0.01)
A = box(0.0, 0.0, 0.05, 0.1)
B = box(0.0, 0.05, 0.05, 0.1)


def _write(path: Path, data: np.ndarray, *, nodata: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=TRANSFORM,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def _ctx(**kw: object) -> CatchmentContext:
    base = dict(
        catchments={"A": A, "B": B},
        area_km2={"A": 60.0, "B": 30.0},
        metric_crs="EPSG:3857",
        reference_year=2021,
    )
    base.update(kw)
    return CatchmentContext(**base)  # type: ignore[arg-type]


def _worldcover(root: Path) -> Path:
    data = np.full((10, 10), 10, dtype="uint8")  # tree cover
    data[:, 0] = 50  # built-up: first column -> 10 of A's 50 cells, 5 of B's 25
    return _write(root / "ESA_WorldCover_10m_2021_v200_TEST_Map.tif", data, nodata=0)


def test_raster_source_counts_pixels_by_centre(tmp_path: Path) -> None:
    path = _worldcover(tmp_path)
    with RasterSource([path], nodata=0) as src:
        sample = src.sample(A)
    assert sample.n_inside == 50
    assert len(sample.values) == 50
    assert sample.fully_covered


def test_worldcover_built_up_share_is_flagged_as_proxy(tmp_path: Path) -> None:
    _worldcover(tmp_path)
    out = WorldCoverAdapter(tmp_path).imperviousness_pct(_ctx())
    assert out["A"].value == pytest.approx(20.0)  # 10 / 50 * 100
    assert out["B"].value == pytest.approx(20.0)  # 5 / 25 * 100
    assert out["A"].flag == "PROXY_BUILT_UP_SHARE"
    assert out["A"].year == 2021
    assert "WorldCover" in (out["A"].source or "")


def test_clms_imperviousness_uses_nearest_year_and_ignores_nodata(tmp_path: Path) -> None:
    imd = tmp_path / "clms" / "imperviousness"
    far = np.full((10, 10), 99, dtype="uint8")
    near = np.full((10, 10), 40, dtype="uint8")
    near[0, 0] = 255  # outside-area code: must not pull the mean up
    _write(imd / "IMD_2015_TEST.tif", far)
    _write(imd / "IMD_2018_TEST.tif", near)
    out = ClmsAdapter(tmp_path / "clms").imperviousness_pct(_ctx(reference_year=2021))
    assert out["A"].value == pytest.approx(40.0)
    assert out["A"].year == 2018
    assert "2018" in (out["A"].source or "")


def test_clms_missing_data_raises_source_unavailable(tmp_path: Path) -> None:
    with pytest.raises(SourceUnavailable):
        ClmsAdapter(tmp_path / "nothing").imperviousness_pct(_ctx())


def test_chain_falls_back_per_attribute_and_records_which_adapter(tmp_path: Path) -> None:
    _worldcover(tmp_path / "wc")
    chain = AdapterChain([ClmsAdapter(tmp_path / "clms"), WorldCoverAdapter(tmp_path / "wc")])
    out = chain.compute("imperviousness_pct", _ctx())
    assert out["A"].value == pytest.approx(20.0)
    assert chain.used == {"imperviousness_pct": "worldcover"}


def test_chain_prefers_clms_when_present(tmp_path: Path) -> None:
    _worldcover(tmp_path / "wc")
    _write(tmp_path / "clms" / "imperviousness" / "IMD_2021.tif", np.full((10, 10), 33, "uint8"))
    chain = AdapterChain([ClmsAdapter(tmp_path / "clms"), WorldCoverAdapter(tmp_path / "wc")])
    assert chain.compute("imperviousness_pct", _ctx())["A"].value == pytest.approx(33.0)
    assert chain.used == {"imperviousness_pct": "clms"}


def test_strict_chain_refuses_fallback(tmp_path: Path) -> None:
    _worldcover(tmp_path / "wc")
    chain = AdapterChain(
        [ClmsAdapter(tmp_path / "clms"), WorldCoverAdapter(tmp_path / "wc")], strict=True
    )
    with pytest.raises(SourceUnavailable):
        chain.compute("imperviousness_pct", _ctx())


def test_calling_code_is_adapter_agnostic(tmp_path: Path) -> None:
    """The Pune guarantee: the same call works whichever adapter the config names."""
    _worldcover(tmp_path / "wc")
    _write(tmp_path / "clms" / "imperviousness" / "IMD_2021.tif", np.full((10, 10), 33, "uint8"))
    adapters: list[LandCoverAdapter] = [
        ClmsAdapter(tmp_path / "clms"),
        WorldCoverAdapter(tmp_path / "wc"),
    ]
    for adapter in adapters:
        result = adapter.compute("imperviousness_pct", _ctx())
        assert set(result) == {"A", "B"}
        assert all(isinstance(v, AttributeValue) for v in result.values())


def test_resolve_adapter_from_config() -> None:
    chain = resolve_adapter({"land_cover_adapter": "clms", "land_cover_fallback": "worldcover"})
    assert [a.name for a in chain.adapters] == ["clms", "worldcover"]
    pune = resolve_adapter({"land_cover_adapter": "worldcover", "land_cover_fallback": None})
    assert [a.name for a in pune.adapters] == ["worldcover"]
    with pytest.raises(ValueError):
        resolve_adapter({"land_cover_adapter": "nope"})


def test_population_supersampled_sum(tmp_path: Path) -> None:
    root = tmp_path / "pop"
    _write(root / "GHS_POP_E2020_TEST.tif", np.full((10, 10), 9.0))
    out = population(_ctx(), root)
    assert out["A"].value == pytest.approx(450.0)  # 50 cells x 9
    assert out["B"].value == pytest.approx(225.0)
    assert out["A"].year == 2020


def test_population_small_catchment_gets_pro_rata_share(tmp_path: Path) -> None:
    root = tmp_path / "pop"
    _write(root / "GHS_POP_E2020_TEST.tif", np.full((10, 10), 9.0))
    # One third of a cell in each direction -> 1/9 of it -> 1 person.
    tiny = box(0.0, 0.1 - 0.01 / 3, 0.01 / 3, 0.1)
    out = population(_ctx(catchments={"T": tiny}), root)
    assert out["T"].value == pytest.approx(1.0)


def test_population_off_raster_is_partial_coverage_null(tmp_path: Path) -> None:
    root = tmp_path / "pop"
    _write(root / "GHS_POP_E2020_TEST.tif", np.full((10, 10), 9.0))
    straddling = box(0.05, 0.0, 0.2, 0.1)
    out = population(_ctx(catchments={"S": straddling}), root)
    assert out["S"].value is None and out["S"].flag == "PARTIAL_COVERAGE"


def test_road_density_km_per_km2() -> None:
    ctx = _ctx(metric_crs="EPSG:4326", area_km2={"A": 1.0, "B": 1.0})
    # Pretend degrees are metres (metric_crs=4326 makes to_metric an identity) - a road
    # 0.04 long inside A is 0.04 m -> 4e-5 km over 1 km2.
    roads = [LineString([(0.01, 0.02), (0.05, 0.02)]), LineString([(0.07, 0.0), (0.07, 0.1)])]
    out = road_density(ctx, roads)
    assert out["A"].value == pytest.approx(4e-5)
    assert out["B"].value == pytest.approx(0.0)  # B genuinely has no road - a real zero


def test_riparian_climatology_excludes_post_training_and_non_ok() -> None:
    obs = pd.DataFrame(
        [
            ("R1", date(2023, 1, 10), 0.2, "OK"),
            ("R1", date(2023, 1, 20), 0.4, "OK"),  # January mean 0.3
            ("R1", date(2023, 7, 10), 0.7, "OK"),  # July mean 0.7
            ("R1", date(2023, 7, 11), 0.9, "CLOUD"),  # not OK - ignored
            ("R1", date(2024, 7, 10), -1.0, "OK"),  # after train_end - must not leak in
        ],
        columns=["reach_id", "obs_date", "riparian_ndvi", "riparian_flag"],
    )
    out = reach_riparian_climatology(obs, date(2023, 12, 31))
    assert out["R1"] == pytest.approx(0.5)  # mean of monthly means (0.3, 0.7)


def test_assembled_rows_carry_sources_and_reasons_never_zero_fill() -> None:
    results = {
        "imperviousness_pct": {"A": AttributeValue(20.0, "WC", "PROXY_BUILT_UP_SHARE", 2021)},
        "alan_radiance": {"A": AttributeValue.missing("SOURCE_UNAVAILABLE")},
        "population": {"A": AttributeValue(449.6, "GHS", None, 2020)},
    }
    frame = assemble_rows(
        ["A"],
        results,
        2021,
        {"imperviousness_pct": "worldcover"},
        {"A": ["CATCHMENT_TRUNCATED", "SHORT_LINK"]},
    )
    row = frame.iloc[0]
    assert row["imperviousness_pct"] == 20.0
    assert row["alan_radiance"] is None
    assert row["flags"]["alan_radiance"] == "SOURCE_UNAVAILABLE"
    assert row["flags"]["riparian_width_m"] == "NOT_COMPUTED"
    assert row["flags"]["_catchment"] == ["CATCHMENT_TRUNCATED"]
    assert row["population"] == 449  # int() truncation of a rounded sum, stored INTEGER
    assert row["year"] == 2021
    assert row["adapter"] == "imperviousness_pct=worldcover"
