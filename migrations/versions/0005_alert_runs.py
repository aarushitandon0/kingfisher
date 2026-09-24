"""alert_runs: one row per live alert run (models/alert_run.py).

A run that issues nothing is still a run. Without this table "no rows in alerts" cannot
tell "the run found nothing to alert on" from "no run has happened" - and /api/alerts
must never pass the second off as the first. The row also keeps what the alerts table
cannot: how many candidates each guardrail suppressed (a Suppressed outcome has no
severity, so it is not an alert row), and the provenance of the forecast it judged.

Also a partial index for "the latest production forecast": without it the lookup every
map load makes scans every walk-forward row (~300k) to find ~7k production ones.

Revision ID: 0005_alert_runs
Revises: 0004_api_alerts_scenarios
Create Date: 2026-09-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_alert_runs"
down_revision: str | None = "0004_api_alerts_scenarios"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "alert_runs",
        sa.Column("city", sa.Text(), nullable=False),
        sa.Column("issued_date", sa.Date(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("weather", sa.Text(), nullable=False),
        # severity -> rows written to `alerts`
        sa.Column("counts", postgresql.JSONB(), nullable=False),
        # guardrail -> candidates it suppressed (not written to `alerts`)
        sa.Column("suppressed", postgresql.JSONB(), nullable=False),
        sa.Column("basis", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("city", "issued_date", name="alert_runs_pkey"),
    )
    op.create_index(
        "ix_forecasts_production_latest",
        "forecasts",
        ["variant", sa.text("issued_date DESC"), sa.text("created_at DESC")],
        postgresql_where=sa.text("fit = 'production'"),
    )


def downgrade() -> None:
    op.drop_index("ix_forecasts_production_latest", "forecasts")
    op.drop_table("alert_runs")
