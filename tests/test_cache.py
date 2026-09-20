"""Cache behaviour. The pipeline is re-run fifty times and should hit the network once,
so an unstable cache key is a real bug, not a performance detail (CLAUDE.md #4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.cache import DiskCache, cache_key, hash_variables


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache("test", root=tmp_path)


def test_cache_key_is_order_insensitive_and_stable() -> None:
    a = cache_key({"lat": 40.2, "lon": -8.4, "start": "2015-01-01"})
    b = cache_key({"start": "2015-01-01", "lon": -8.4, "lat": 40.2})
    assert a == b
    assert a == cache_key({"lat": 40.2, "lon": -8.4, "start": "2015-01-01"})


def test_cache_key_changes_when_any_parameter_changes() -> None:
    base = {"lat": 40.2, "lon": -8.4, "start": "2015-01-01"}
    assert cache_key(base) != cache_key({**base, "start": "2015-01-02"})
    assert cache_key(base) != cache_key({**base, "lat": 40.3})


def test_variables_hash_ignores_order_but_not_membership() -> None:
    assert hash_variables(["precipitation", "temperature_2m"]) == hash_variables(
        ["temperature_2m", "precipitation"]
    )
    assert hash_variables(["precipitation"]) != hash_variables(
        ["precipitation", "temperature_2m"]
    )


def test_second_fetch_is_served_from_disk(cache: DiskCache) -> None:
    calls = {"n": 0}

    def fetch() -> dict[str, int]:
        calls["n"] += 1
        return {"value": calls["n"]}

    params = {"lat": 40.2, "lon": -8.4}
    first = cache.get_or_fetch(params, fetch, slug="coimbra")
    second = cache.get_or_fetch(params, fetch, slug="coimbra")

    assert calls["n"] == 1, "the network was hit twice for identical parameters"
    assert first.hit is False
    assert second.hit is True
    assert second.payload == {"value": 1}


def test_refresh_forces_a_refetch(cache: DiskCache) -> None:
    calls = {"n": 0}

    def fetch() -> dict[str, int]:
        calls["n"] += 1
        return {"value": calls["n"]}

    params = {"lat": 40.2}
    cache.get_or_fetch(params, fetch)
    entry = cache.get_or_fetch(params, fetch, refresh=True)
    assert calls["n"] == 2
    assert entry.payload == {"value": 2}


def test_meta_sidecar_records_provenance(cache: DiskCache) -> None:
    import json

    params = {"lat": 40.2, "lon": -8.4}
    cache.put(params, {"ok": True}, slug="coimbra", url="https://example.invalid/x",
              source="Test source")
    _, meta_path = cache.paths_for(params, "coimbra")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["url"] == "https://example.invalid/x"
    assert meta["source"] == "Test source"
    assert meta["params"] == params
    assert meta["fetched_at"]
