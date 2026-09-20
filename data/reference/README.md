# data/reference/

The only part of `data/` that is committed. Everything here is hand-curated and small.

- `incidents.csv` — documented fish kills, sewage discharges and algal blooms used to
  score anomaly detection (MASTERSPEC §7.3). Columns: `date`, `city`, `reach_id`,
  `incident_type`, `description`, `source_url`, `confidence`.
  One row per verified event, each with a source URL. 5–15 events over 10 years is
  enough to report precision and recall with honest confidence intervals.

If ground truth stays thin, the fallback is a statistical anomaly definition (exceedance
of the reach's own seasonal 95th percentile), labelled in the README as a proxy rather
than a verified incident.

Provenance for every acquired file is logged in `DATA_INVENTORY.md` at the repo root.
