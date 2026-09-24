"""API-facing columns: alert basis/guardrails, production-forecast provenance, scenarios.

- alerts: basis, guardrails, withheld_exceedance_prob, threshold_derivation - what
  engine.alerts.to_record produces and /api/alerts/{id} serves ("attribution + exposure +
  basis"). withheld_exceedance_prob is only ever set on INSUFFICIENT_EVIDENCE rows, and
  such a row still never carries exceedance_prob.
- forecasts: future_drivers_missing - on the production forecast issued on the last
  archive day, the target-day weather does not exist yet; the count says how many of the
  t+h driver features were NULL, so a weaker forecast is labelled, not hidden.
- scenarios: the full engine.scenarios.ScenarioResult (result) and the model it ran on.
- scenario_results: one row per (scenario, reach, VARIABLE) - results are per target
  variable - with the status, the reason when not OK, and the baseline interval next to
  the scenario interval (ci_low/ci_high). An OK row must carry citations and both
  intervals; a row that is not OK must say why and carries no scenario numbers.

Revision ID: 0004_api_alerts_scenarios
Revises: 0003_riparian_forecasts_exposure
Create Date: 2026-09-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_api_alerts_scenarios"
down_revision: str | None = "0003_riparian_forecasts_exposure"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCENARIO_STATUSES = ("OK", "INSUFFICIENT_EVIDENCE", "NOT_ESTIMABLE")


def upgrade() -> None:
    op.add_column("alerts", sa.Column("basis", postgresql.JSONB(), nullable=True))
    op.add_column("alerts", sa.Column("guardrails", postgresql.JSONB(), nullable=True))
    op.add_column("alerts", sa.Column("withheld_exceedance_prob", sa.Float(), nullable=True))
    op.add_column("alerts", sa.Column("threshold_derivation", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_alerts_withheld_only_when_insufficient",
        "alerts",
        "withheld_exceedance_prob IS NULL OR severity = 'INSUFFICIENT_EVIDENCE'",
    )
    op.create_check_constraint(
        "ck_alerts_insufficient_has_no_prob",
        "alerts",
        "severity <> 'INSUFFICIENT_EVIDENCE' OR exceedance_prob IS NULL",
    )

    op.add_column("forecasts", sa.Column("future_drivers_missing", sa.Integer(), nullable=True))

    op.add_column("scenarios", sa.Column("result", postgresql.JSONB(), nullable=True))
    op.add_column("scenarios", sa.Column("model_version", sa.Text(), nullable=True))

    # scenario_results was never written before this revision; adding NOT NULL columns
    # fails loudly if it somehow holds rows, rather than inventing values for them.
    op.add_column("scenario_results", sa.Column("variable", sa.Text(), nullable=False))
    op.add_column("scenario_results", sa.Column("status", sa.Text(), nullable=False))
    op.add_column("scenario_results", sa.Column("reason", sa.Text(), nullable=True))
    op.add_column("scenario_results", sa.Column("baseline_ci_low", sa.Float(), nullable=True))
    op.add_column("scenario_results", sa.Column("baseline_ci_high", sa.Float(), nullable=True))
    op.drop_constraint("scenario_results_pkey", "scenario_results")
    op.create_primary_key(
        "scenario_results_pkey", "scenario_results", ["scenario_id", "reach_id", "variable"]
    )
    rendered = ", ".join(f"'{s}'" for s in SCENARIO_STATUSES)
    op.create_check_constraint(
        "ck_scenario_results_status", "scenario_results", f"status IN ({rendered})"
    )
    op.create_check_constraint(
        "ck_scenario_results_ok_is_complete",
        "scenario_results",
        "status <> 'OK' OR (citations IS NOT NULL AND scenario_exceedance_days IS NOT NULL "
        "AND baseline_exceedance_days IS NOT NULL AND ci_low IS NOT NULL "
        "AND baseline_ci_low IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_scenario_results_not_ok_says_why",
        "scenario_results",
        "status = 'OK' OR (reason IS NOT NULL AND scenario_exceedance_days IS NULL "
        "AND delta IS NULL)",
    )
    op.create_check_constraint(
        "ck_scenario_results_strictly_wider",
        "scenario_results",
        "status <> 'OK' OR (ci_high - ci_low) > (baseline_ci_high - baseline_ci_low)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_scenario_results_strictly_wider", "scenario_results")
    op.drop_constraint("ck_scenario_results_not_ok_says_why", "scenario_results")
    op.drop_constraint("ck_scenario_results_ok_is_complete", "scenario_results")
    op.drop_constraint("ck_scenario_results_status", "scenario_results")
    op.drop_constraint("scenario_results_pkey", "scenario_results")
    op.execute("DELETE FROM scenario_results")
    op.create_primary_key("scenario_results_pkey", "scenario_results", ["scenario_id", "reach_id"])
    for column in ("baseline_ci_high", "baseline_ci_low", "reason", "status", "variable"):
        op.drop_column("scenario_results", column)
    op.drop_column("scenarios", "model_version")
    op.drop_column("scenarios", "result")
    op.drop_column("forecasts", "future_drivers_missing")
    op.drop_constraint("ck_alerts_insufficient_has_no_prob", "alerts")
    op.drop_constraint("ck_alerts_withheld_only_when_insufficient", "alerts")
    for column in ("threshold_derivation", "withheld_exceedance_prob", "guardrails", "basis"):
        op.drop_column("alerts", column)
