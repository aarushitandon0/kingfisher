"""SQLAlchemy engine/session factory. No schema definitions live here — the
schema is owned by Alembic migrations (see `migrations/versions/`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.settings import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back and re-raises on failure."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# bulk writes
# ---------------------------------------------------------------------------
def to_db_value(value: Any) -> Any:
    """One cell -> a value psycopg writes faithfully.

    The load-bearing case: a pandas/numpy NaN must become SQL NULL. Written as-is it
    lands in a FLOAT column as the IEEE value 'NaN', which is neither NULL nor a
    number - a silent fill that every `IS NULL` check downstream would miss
    (CLAUDE.md #2). numpy scalars become Python scalars; dates stay dates.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        if isinstance(value, np.floating) and np.isnan(value):
            return None
        return value.item()
    if value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, dict | list):
        return json.dumps(value, default=str)
    return value


def frame_records(
    df: pd.DataFrame, columns: Sequence[str], integer_columns: frozenset[str] = frozenset()
) -> Iterator[tuple[Any, ...]]:
    """Rows as DB-ready tuples. An INTEGER column holding NULLs arrives from pandas as
    float64 (178.0), which COPY rejects; those cells are cast back to int here."""
    int_positions = {i for i, c in enumerate(columns) if c in integer_columns}
    for row in df[list(columns)].itertuples(index=False, name=None):
        out = []
        for i, v in enumerate(row):
            value = to_db_value(v)
            if i in int_positions and isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(f"non-integer {value} for INTEGER column {columns[i]}")
                value = int(value)
            out.append(value)
        yield tuple(out)


def integer_columns(table: str) -> frozenset[str]:
    from sqlalchemy import text

    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = :t "
                "AND data_type IN ('integer', 'bigint', 'smallint')"
            ),
            {"t": table},
        )
        return frozenset(r[0] for r in rows)


def bulk_upsert(
    df: pd.DataFrame,
    table: str,
    key_columns: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
) -> int:
    """COPY `df` into a temp table, then INSERT ... ON CONFLICT (key) DO UPDATE.

    COPY is what makes a 1.4M-row drivers_daily write take seconds rather than an
    hour. Only the columns present in `df` (or `columns`) are written or updated;
    anything else on an existing row is left alone. Returns rows written.
    """
    if df.empty:
        return 0
    cols = list(columns or df.columns)
    missing = [k for k in key_columns if k not in cols]
    if missing:
        raise ValueError(f"bulk_upsert into {table}: key columns {missing} not in frame")
    col_sql = ", ".join(f'"{c}"' for c in cols)
    updates = [c for c in cols if c not in key_columns]
    set_sql = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in updates)
    conflict = ", ".join(f'"{k}"' for k in key_columns)
    tmp = f"_tmp_{table}"

    raw = get_engine().raw_connection()
    try:
        # The psycopg connection underneath SQLAlchemy's DBAPI wrapper: COPY is
        # psycopg-specific and not part of the DBAPI cursor interface.
        driver: Any = raw.driver_connection
        with driver.cursor() as cur:
            cur.execute(
                f'CREATE TEMP TABLE "{tmp}" (LIKE "{table}" INCLUDING DEFAULTS) ON COMMIT DROP'
            )
            with cur.copy(f'COPY "{tmp}" ({col_sql}) FROM STDIN') as copy:
                for record in frame_records(df, cols, integer_columns(table)):
                    copy.write_row(record)
            action = f"DO UPDATE SET {set_sql}" if updates else "DO NOTHING"
            cur.execute(
                f'INSERT INTO "{table}" ({col_sql}) SELECT {col_sql} FROM "{tmp}" '
                f"ON CONFLICT ({conflict}) {action}"
            )
            written = cur.rowcount
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    return int(written)
