"""Riparian midsummer windows, seven-quantile variant-aware forecasts, exposure provenance.

- riparian_ndvi_window: one row per (reach, year) - the median riparian NDVI over the
  clear acquisitions of a narrow midsummer window (config/sentinel2.yaml -> plan). The
  per-acquisition riparian columns in `observations` are no longer filled by L1 (they
  are written NOT_REQUESTED); every NULL here carries a flag.
- forecasts: p05/p25/p75/p95 alongside p10/p50/p90; `variant` (A = reach_id feature,
  B = static attributes only), `weather` (ORACLE archive, ASISSUED replayed runs, LIVE)
  and `fit` (walk-forward fold or production). The primary key gains model_version and
  weather, because the same reach/issue/target/variable is now legitimately forecast by
  several models under several weather inputs, and none may overwrite another.
- exposure_features: buffer_m and source, so a count says what radius it was counted in.

Revision ID: 0003_riparian_forecasts_exposure
Revises: 0002_l1_l2_features
Create Date: 2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_riparian_forecasts_exposure"
down_revision: str | None = "0002_l1_l2_features"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RIPARIAN_WINDOW_FLAGS = ("OK", "NO_CLEAR_ACQUISITION", "NO_ACQUISITION")
QUANTILES = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    rendered = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({rendered})"


def upgrade() -> None:
    op.create_table(
        "riparian_ndvi_window",
        sa.Column("reach_id", sa.Text(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("window_start", sa.Date(), nullable=False),
        sa.Column("window_end", sa.Date(), nullable=False),
        sa.Column("riparian_ndvi_median", sa.Float(), nullable=True),
        sa.Column("n_acquisitions", sa.Integer(), nullable=False),
        sa.Column("n_clear", sa.Integer(), nullable=False),
        sa.Column("flag", sa.Text(), nullable=False),
        sa.Column("flag_counts", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("reach_id", "year"),
        sa.CheckConstraint(_in_list("flag", RIPARIAN_WINDOW_FLAGS), name="ck_riparian_window_flag"),
        sa.CheckConstraint(
            "(flag = 'OK') = (riparian_ndvi_median IS NOT NULL)",
            name="ck_riparian_window_ok_iff_value",
        ),
    )

    for q in ("p05", "p25", "p75", "p95"):
        op.add_column("forecasts", sa.Column(q, sa.Float(), nullable=True))
    op.add_column("forecasts", sa.Column("variant", sa.Text(), nullable=False, server_default="A"))
    op.add_column(
        "forecasts", sa.Column("weather", sa.Text(), nullable=False, server_default="ORACLE")
    )
    op.add_column("forecasts", sa.Column("fit", sa.Text(), nullable=True))
    op.add_column("forecasts", sa.Column("horizon", sa.Integer(), nullable=True))
    op.add_column(
        "forecasts",
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()")),
    )
    op.create_check_constraint("ck_forecasts_variant", "forecasts", _in_list("variant", ("A", "B")))
    op.create_check_constraint(
        "ck_forecasts_weather", "forecasts", _in_list("weather", ("ORACLE", "ASISSUED", "LIVE"))
    )
    op.drop_constraint("ck_forecasts_quantile_order", "forecasts")
    ordered = " AND ".join(f"{a} <= {b}" for a, b in zip(QUANTILES, QUANTILES[1:], strict=False))
    nulls = " OR ".join(f"{q} IS NULL" for q in QUANTILES)
    op.create_check_constraint(
        "ck_forecasts_quantile_order", "forecasts", f"({nulls}) OR ({ordered})"
    )
    op.drop_constraint("forecasts_pkey", "forecasts")
    op.create_primary_key(
        "forecasts_pkey",
        "forecasts",
        ["reach_id", "issued_date", "target_date", "variable", "model_version", "weather"],
    )

    op.add_column("exposure_features", sa.Column("buffer_m", sa.Float(), nullable=True))
    op.add_column("exposure_features", sa.Column("source", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("exposure_features", "source")
    op.drop_column("exposure_features", "buffer_m")
    op.drop_constraint("forecasts_pkey", "forecasts")
    op.execute("DELETE FROM forecasts WHERE weather <> 'ORACLE' OR variant <> 'A'")
    op.create_primary_key(
        "forecasts_pkey", "forecasts", ["reach_id", "issued_date", "target_date", "variable"]
    )
    op.drop_constraint("ck_forecasts_quantile_order", "forecasts")
    op.create_check_constraint(
        "ck_forecasts_quantile_order",
        "forecasts",
        "p10 IS NULL OR p50 IS NULL OR p90 IS NULL OR (p10 <= p50 AND p50 <= p90)",
    )
    op.drop_constraint("ck_forecasts_weather", "forecasts")
    op.drop_constraint("ck_forecasts_variant", "forecasts")
    for column in (
        "created_at",
        "horizon",
        "fit",
        "weather",
        "variant",
        "p95",
        "p75",
        "p25",
        "p05",
    ):
        op.drop_column("forecasts", column)
    op.drop_table("riparian_ndvi_window")
