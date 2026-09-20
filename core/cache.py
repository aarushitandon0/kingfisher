"""On-disk response cache.

Every network response hits disk before it hits a DataFrame, keyed by the request
parameters (CLAUDE.md #4). We re-run this pipeline fifty times and should hit the
network once.

Layout:

    data/raw/<namespace>/<slug>__<key8>.json        raw response body, verbatim
    data/raw/<namespace>/<slug>__<key8>.meta.json   params, url, fetched_at, bytes

The key is a SHA-256 over the canonical JSON of the request parameters, so any
change to a parameter (a date, a variable list, an evalscript) is a cache miss.
The slug is cosmetic — it makes the directory browsable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.logging import get_logger
from core.settings import DATA_DIR

log = get_logger(__name__)

_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def cache_key(params: Mapping[str, Any]) -> str:
    """Stable SHA-256 over the request parameters."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_variables(variables: list[str]) -> str:
    """Short, order-insensitive hash of a variable list (used in cache keys and slugs)."""
    joined = ",".join(sorted(variables))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]


def slugify(text: str) -> str:
    return _SLUG_SAFE.sub("-", text).strip("-")[:80]


@dataclass(frozen=True)
class CacheEntry:
    """A cached response plus where it came from."""

    payload: Any
    path: Path
    hit: bool
    fetched_at: str

    @property
    def age_days(self) -> float:
        fetched = datetime.fromisoformat(self.fetched_at)
        return (datetime.now(UTC) - fetched).total_seconds() / 86400.0


class DiskCache:
    """A namespaced JSON response cache under `data/raw/`."""

    def __init__(self, namespace: str, root: Path | None = None) -> None:
        self.namespace = namespace
        self.root = (root or DATA_DIR / "raw") / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def paths_for(self, params: Mapping[str, Any], slug: str = "") -> tuple[Path, Path]:
        key = cache_key(params)
        stem = f"{slugify(slug)}__{key[:8]}" if slug else key[:16]
        return self.root / f"{stem}.json", self.root / f"{stem}.meta.json"

    # -- read / write ------------------------------------------------------
    def get(self, params: Mapping[str, Any], slug: str = "") -> CacheEntry | None:
        body_path, meta_path = self.paths_for(params, slug)
        if not (body_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        payload = json.loads(body_path.read_text(encoding="utf-8"))
        return CacheEntry(
            payload=payload, path=body_path, hit=True, fetched_at=meta["fetched_at"]
        )

    def put(
        self,
        params: Mapping[str, Any],
        payload: Any,
        *,
        slug: str = "",
        url: str = "",
        source: str = "",
    ) -> CacheEntry:
        body_path, meta_path = self.paths_for(params, slug)
        body = json.dumps(payload, separators=(",", ":"), default=str)
        body_path.write_text(body, encoding="utf-8")
        fetched_at = datetime.now(UTC).isoformat()
        meta_path.write_text(
            json.dumps(
                {
                    "namespace": self.namespace,
                    "params": dict(params),
                    "url": url,
                    "source": source,
                    "fetched_at": fetched_at,
                    "bytes": len(body),
                    "key": cache_key(params),
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        return CacheEntry(payload=payload, path=body_path, hit=False, fetched_at=fetched_at)

    def get_or_fetch(
        self,
        params: Mapping[str, Any],
        fetch: Callable[[], Any],
        *,
        slug: str = "",
        url: str = "",
        source: str = "",
        refresh: bool = False,
    ) -> CacheEntry:
        """Return the cached response, or call `fetch()` once and cache the result.

        `fetch` must raise on failure. A cached empty result is still a failure — that
        check belongs to the caller, which knows what "empty" means for its source.
        """
        if not refresh:
            entry = self.get(params, slug)
            if entry is not None:
                log.debug("cache.hit", namespace=self.namespace, path=str(entry.path))
                return entry
        log.info("cache.miss", namespace=self.namespace, url=url, slug=slug)
        payload = fetch()
        entry = self.put(params, payload, slug=slug, url=url, source=source)
        log.info("cache.stored", namespace=self.namespace, path=str(entry.path))
        return entry
