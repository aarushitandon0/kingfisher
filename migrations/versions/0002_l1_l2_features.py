"""L1/L2 feature columns - the MASTERSPEC 6.2 driver set, riparian flag, provenance.

- drivers_daily gains the 6.2 features 0001 did not carry: precip_duration_h,
  temp_low_flow_index, doy_sin/doy_cos and the upstream state lag. The upstream state
  is two variables (turbidity and NDCI), so it is two columns plus a flag that says
  why it is NULL when it is NULL - a headwater reach has no upstream reach, which is a
  different fact from "the upstream reach was not seen yesterday".
- observations gains riparian_flag. The riparian NDVI comes from a different
  geometry (30 m buffer, non-water pixels) than the water indices, so it can be
  missing for a different reason than quality_flag gives.
- catchment_attributes gains per-attribute sources and flags (JSONB). Every value
  says which dataset it came from; every NULL says why.
- driver_query_points records, per reach, the upstream-catchment centroid, the
  weather grid cell it was snapped to and the grid point Open-Meteo actually served.

Revision ID: 0002_l1_l2_features
Revises: 0001_initial
Create Date: 2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_l1_l2_features"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RIPARIAN_FLAGS = ("OK", "CLOUD", "NO_LAND_PIXELS", "OUT_OF_RANGE", "NOT_REQUESTED")
UPSTREAM_FLAGS = ("OK", "NO_UPSTREAM_REACH", "NO_RECENT_UPSTREAM_OBS")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    rendered = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({rendered})"


def upgrade() -> None:
    # ------------------------------------------------------------ drivers --
    op.add_column("drivers_daily", sa.Column("precip_duration_h", sa.Integer(), nullable=True))
    op.add_column("drivers_daily", sa.Column("temp_low_flow_index", sa.Float(), nullable=True))
    op.add_column("drivers_daily", sa.Column("doy_sin", sa.Float(), nullable=True))
    op.add_column("drivers_daily", sa.Column("doy_cos", sa.Float(), nullable=True))
    op.add_column(
        "drivers_daily", sa.Column("upstream_state_lag1_turbidity", sa.Float(), nullable=True)
    )
    op.add_column("drivers_daily", sa.Column("upstream_state_lag1_ndci", sa.Float(), nullable=True))
    op.add_column("drivers_daily", sa.Column("upstream_state_flag", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_drivers_upstream_flag",
        "drivers_daily",
        "upstream_state_flag IS NULL OR " + _in_list("upstream_state_flag", UPSTREAM_FLAGS),
    )

    # -------------------------------------------------------- observations --
    op.add_column("observations", sa.Column("riparian_flag", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_observations_riparian_flag",
        "observations",
        "riparian_flag IS NULL OR " + _in_list("riparian_flag", RIPARIAN_FLAGS),
    )
    op.create_check_constraint(
        "ck_observations_riparian_ok_has_value",
        "observations",
        "riparian_flag IS DISTINCT FROM 'OK' OR riparian_ndvi IS NOT NULL",
    )

    # ------------------------------------------------ catchment attributes --
    op.add_column("catchment_attributes", sa.Column("adapter", sa.Text(), nullable=True))
    op.add_column("catchment_attributes", sa.Column("sources", postgresql.JSONB(), nullable=True))
    op.add_column("catchment_attributes", sa.Column("flags", postgresql.JSONB(), nullable=True))

    # ------------------------------------------------- weather query points --
    op.create_table(
        "driver_query_points",
        sa.Column("reach_id", sa.Text(), primary_key=True),
        sa.Column("centroid_lon", sa.Float(), nullable=False),
        sa.Column("centroid_lat", sa.Float(), nullable=False),
        sa.Column("query_lon", sa.Float(), nullable=False),
        sa.Column("query_lat", sa.Float(), nullable=False),
        sa.Column("served_lon", sa.Float(), nullable=True),
        sa.Column("served_lat", sa.Float(), nullable=True),
        sa.Column("served_elevation_m", sa.Float(), nullable=True),
        sa.Column("grid_deg", sa.Float(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("driver_query_points")
    op.drop_column("catchment_attributes", "flags")
    op.drop_column("catchment_attributes", "sources")
    op.drop_column("catchment_attributes", "adapter")
    op.drop_constraint("ck_observations_riparian_ok_has_value", "observations")
    op.drop_constraint("ck_observations_riparian_flag", "observations")
    op.drop_column("observations", "riparian_flag")
    op.drop_constraint("ck_drivers_upstream_flag", "drivers_daily")
    for column in (
        "upstream_state_flag",
        "upstream_state_lag1_ndci",
        "upstream_state_lag1_turbidity",
        "doy_cos",
        "doy_sin",
        "temp_low_flow_index",
        "precip_duration_h",
    ):
        op.drop_column("drivers_daily", column)
