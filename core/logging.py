"""Structured logging. Every pipeline module imports from here — nothing calls
`logging.basicConfig` or `print` on its own.

Usage:

    from core.logging import get_logger, stage

    log = get_logger(__name__)

    with stage(log, "l1_satellite", city="coimbra") as s:
        ...
        s.record(rows_in=4012, rows_out=3788, rows_dropped=224, reason="NO_WATER_PIXELS")

The stage context manager enforces the convention in CLAUDE.md: every pipeline
stage logs rows in, rows out, rows dropped and why.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Idempotently configure structlog + stdlib logging for the whole process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    from core.settings import get_settings

    settings = get_settings()
    level = (level or settings.log_level).upper()
    fmt = (fmt or settings.log_format).lower()

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first call."""
    configure_logging()
    return structlog.get_logger(name)


@dataclass
class StageCounters:
    """Row accounting for one pipeline stage."""

    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        *,
        rows_in: int | None = None,
        rows_out: int | None = None,
        rows_dropped: int | None = None,
        reason: str | None = None,
        **extra: Any,
    ) -> None:
        if rows_in is not None:
            self.rows_in = rows_in
        if rows_out is not None:
            self.rows_out = rows_out
        if rows_dropped is not None:
            self.rows_dropped = rows_dropped
            if reason:
                self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + rows_dropped
        self.extra.update(extra)

    def drop(self, count: int, reason: str) -> None:
        """Record `count` rows dropped for a named reason (a quality_flag value)."""
        self.rows_dropped += count
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + count


@contextmanager
def stage(
    log: structlog.stdlib.BoundLogger,
    name: str,
    **context: Any,
) -> Iterator[StageCounters]:
    """Log stage start/finish with row accounting and duration.

    Re-raises on failure after logging — pipelines fail loudly (CLAUDE.md #3).
    """
    counters = StageCounters()
    log.info("stage.start", stage=name, **context)
    started = time.perf_counter()
    try:
        yield counters
    except Exception as exc:
        log.error(
            "stage.failed",
            stage=name,
            duration_s=round(time.perf_counter() - started, 3),
            error=str(exc),
            error_type=type(exc).__name__,
            **context,
        )
        raise
    log.info(
        "stage.finish",
        stage=name,
        duration_s=round(time.perf_counter() - started, 3),
        rows_in=counters.rows_in,
        rows_out=counters.rows_out,
        rows_dropped=counters.rows_dropped,
        drop_reasons=counters.drop_reasons,
        **context,
        **counters.extra,
    )
