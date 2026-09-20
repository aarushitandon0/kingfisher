"""Initial schema - all tables from MASTERSPEC.md section 5.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-20
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

QUALITY_FLAGS = ("OK", "CLOUD", "NO_WATER_PIXELS", "OUT_OF_RANGE")
OBS_SOURCES = ("S2", "SNIRH", "EEA", "CITIZEN")
SEVERITIES = ("WATCH", "ALERT", "INSUFFICIENT_EVIDENCE")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    rendered = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({rendered})"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ------------------------------------------------------------------ L0 --
    op.create_table(
        "reaches",
        sa.Column("reach_id", sa.Text(), primary_key=True),
        sa.Column("city", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("geom", Geometry("LINESTRING", srid=4326), nullable=False),
        sa.Column("length_m", sa.Float(), nullable=False),
        sa.Column("strahler_order", sa.Integer(), nullable=True),
        sa.Column("catchment_geom", Geometry("POLYGON", srid=4326), nullable=True),
        sa.Column("catchment_area_km2", sa.Float(), nullable=True),
        # observable / median_water_pixels stay NULL until the Day-1 observability
        # gate has run. NULL means "not yet assessed", not "unobservable".
        sa.Column("observable", sa.Boolean(), nullable=True),
        sa.Column("median_water_pixels", sa.Float(), nullable=True),
    )
    op.create_index("ix_reaches_city", "reaches", ["city"])
    op.create_index("ix_reaches_observable", "reaches", ["observable"])

    op.create_table(
        "reach_topology",
        sa.Column("upstream_id", sa.Text(), nullable=False),
        sa.Column("downstream_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["upstream_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["downstream_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("upstream_id", "downstream_id"),
        sa.CheckConstraint("upstream_id <> downstream_id", name="ck_topology_no_self_edge"),
    )
    op.create_index("ix_topology_downstream", "reach_topology", ["downstream_id"])

    # ------------------------------------------------------------------ L1 --
    op.create_table(
        "observations",
        sa.Column("reach_id", sa.Text(), nullable=False),
        sa.Column("obs_date", sa.Date(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        # Value columns are nullable on purpose: a missing reach-date is NULL plus a
        # quality_flag, never zero and never an unlabelled fill (CLAUDE.md #2).
        sa.Column("turbidity_proxy", sa.Float(), nullable=True),
        sa.Column("ndci", sa.Float(), nullable=True),
        sa.Column("mndwi", sa.Float(), nullable=True),
        sa.Column("riparian_ndvi", sa.Float(), nullable=True),
        sa.Column("water_pixel_count", sa.Integer(), nullable=True),
        sa.Column("cloud_fraction", sa.Float(), nullable=True),
        sa.Column("quality_flag", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("reach_id", "obs_date", "source"),
        sa.CheckConstraint(_in_list("source", OBS_SOURCES), name="ck_observations_source"),
        sa.CheckConstraint(
            _in_list("quality_flag", QUALITY_FLAGS), name="ck_observations_quality_flag"
        ),
        # A row flagged OK must carry at least one value; a flagged row must not
        # masquerade as a clean observation.
        sa.CheckConstraint(
            "quality_flag <> 'OK' OR turbidity_proxy IS NOT NULL OR ndci IS NOT NULL",
            name="ck_observations_ok_has_value",
        ),
    )
    op.create_index("ix_observations_date", "observations", ["obs_date"])
    op.create_index("ix_observations_reach_date", "observations", ["reach_id", "obs_date"])

    # ------------------------------------------------------------------ L2 --
    op.create_table(
        "drivers_daily",
        sa.Column("reach_id", sa.Text(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("precip_mm", sa.Float(), nullable=True),
        sa.Column("precip_max_hourly", sa.Float(), nullable=True),
        sa.Column("temp_mean_c", sa.Float(), nullable=True),
        sa.Column("temp_max_c", sa.Float(), nullable=True),
        sa.Column("soil_moisture", sa.Float(), nullable=True),
        sa.Column("et0", sa.Float(), nullable=True),
        sa.Column("api_7", sa.Float(), nullable=True),
        sa.Column("api_14", sa.Float(), nullable=True),
        sa.Column("api_30", sa.Float(), nullable=True),
        sa.Column("antecedent_dry_days", sa.Integer(), nullable=True),
        sa.Column("first_flush_index", sa.Float(), nullable=True),
        # ARCHIVE (Open-Meteo archive, training) vs FORECAST (live). Kept distinct so
        # leakage tests can assert no forecast-sourced row enters a training fold.
        sa.Column("source", sa.Text(), nullable=False, server_default="ARCHIVE"),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("reach_id", "date"),
        sa.CheckConstraint("source IN ('ARCHIVE', 'FORECAST')", name="ck_drivers_source"),
    )
    op.create_index("ix_drivers_daily_date", "drivers_daily", ["date"])

    op.create_table(
        "catchment_attributes",
        sa.Column("reach_id", sa.Text(), primary_key=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("imperviousness_pct", sa.Float(), nullable=True),
        sa.Column("riparian_ndvi_mean", sa.Float(), nullable=True),
        sa.Column("riparian_width_m", sa.Float(), nullable=True),
        sa.Column("road_density_km_km2", sa.Float(), nullable=True),
        sa.Column("alan_radiance", sa.Float(), nullable=True),
        sa.Column("population", sa.Integer(), nullable=True),
        sa.Column("urban_fraction", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
    )

    # --------------------------------------------------------------- L3/L4 --
    op.create_table(
        "forecasts",
        sa.Column("reach_id", sa.Text(), nullable=False),
        sa.Column("issued_date", sa.Date(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("variable", sa.Text(), nullable=False),
        sa.Column("p10", sa.Float(), nullable=True),
        sa.Column("p50", sa.Float(), nullable=True),
        sa.Column("p90", sa.Float(), nullable=True),
        sa.Column("burstiness", sa.Float(), nullable=True),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("reach_id", "issued_date", "target_date", "variable"),
        sa.CheckConstraint("target_date >= issued_date", name="ck_forecasts_no_hindcast"),
        sa.CheckConstraint(
            "p10 IS NULL OR p50 IS NULL OR p90 IS NULL OR (p10 <= p50 AND p50 <= p90)",
            name="ck_forecasts_quantile_order",
        ),
    )
    op.create_index("ix_forecasts_issued", "forecasts", ["issued_date"])
    op.create_index("ix_forecasts_target", "forecasts", ["target_date"])

    op.create_table(
        "alerts",
        sa.Column(
            "alert_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("reach_id", sa.Text(), nullable=False),
        sa.Column("issued_date", sa.Date(), nullable=False),
        sa.Column("target_window", postgresql.DATERANGE(), nullable=False),
        sa.Column("variable", sa.Text(), nullable=False),
        # A NULL exceedance_prob is legitimate for INSUFFICIENT_EVIDENCE.
        sa.Column("exceedance_prob", sa.Float(), nullable=True),
        sa.Column("threshold_value", sa.Float(), nullable=True),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("attribution", postgresql.JSONB(), nullable=True),
        sa.Column("exposure", postgresql.JSONB(), nullable=True),
        sa.Column("suppressed_reason", sa.Text(), nullable=True),
        sa.Column("model_version", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.CheckConstraint(_in_list("severity", SEVERITIES), name="ck_alerts_severity"),
        sa.CheckConstraint(
            "exceedance_prob IS NULL OR (exceedance_prob >= 0 AND exceedance_prob <= 1)",
            name="ck_alerts_prob_range",
        ),
        # INSUFFICIENT_EVIDENCE always says why. It is an output, not a silent gap.
        sa.CheckConstraint(
            "severity <> 'INSUFFICIENT_EVIDENCE' OR suppressed_reason IS NOT NULL",
            name="ck_alerts_insufficient_has_reason",
        ),
    )
    op.create_index("ix_alerts_reach_issued", "alerts", ["reach_id", "issued_date"])
    op.create_index("ix_alerts_severity", "alerts", ["severity"])

    # ------------------------------------------------------------------ L5 --
    op.create_table(
        "scenarios",
        sa.Column(
            "scenario_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("city", sa.Text(), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_scenarios_city", "scenarios", ["city"])

    op.create_table(
        "scenario_results",
        sa.Column("scenario_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reach_id", sa.Text(), nullable=False),
        sa.Column("baseline_exceedance_days", sa.Float(), nullable=True),
        sa.Column("scenario_exceedance_days", sa.Float(), nullable=True),
        sa.Column("delta", sa.Float(), nullable=True),
        sa.Column("ci_low", sa.Float(), nullable=True),
        sa.Column("ci_high", sa.Float(), nullable=True),
        # Citations for every coefficient applied to this reach. A scenario result
        # without citations must not be servable (MASTERSPEC section 9.2).
        sa.Column("citations", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["scenario_id"], ["scenarios.scenario_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("scenario_id", "reach_id"),
        sa.CheckConstraint(
            "ci_low IS NULL OR ci_high IS NULL OR ci_low <= ci_high",
            name="ck_scenario_results_ci_order",
        ),
    )

    # ------------------------------------------------------------ exposure --
    op.create_table(
        "exposure_features",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column("reach_id", sa.Text(), nullable=False),
        sa.Column("feature_type", sa.Text(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=True),
        sa.Column("nearest_distance_m", sa.Float(), nullable=True),
        sa.Column("geom", Geometry("GEOMETRY", srid=4326), nullable=True),
        sa.ForeignKeyConstraint(["reach_id"], ["reaches.reach_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("reach_id", "feature_type", name="uq_exposure_reach_type"),
    )
    op.create_index("ix_exposure_reach", "exposure_features", ["reach_id"])


def downgrade() -> None:
    op.drop_table("exposure_features")
    op.drop_table("scenario_results")
    op.drop_table("scenarios")
    op.drop_table("alerts")
    op.drop_table("forecasts")
    op.drop_table("catchment_attributes")
    op.drop_table("drivers_daily")
    op.drop_table("observations")
    op.drop_table("reach_topology")
    op.drop_table("reaches")
