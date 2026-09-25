---
title: Kingfisher
emoji: 🐦
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Early warning and resilience planning for urban streams
---

# Kingfisher - live demo

Early-warning and resilience-planning system for urban streams (OneAquaHealth IEEE Global
Hackathon 2026, Track 6: Resilience Informatics). Coimbra and Pune.

This Space runs a frozen snapshot of the production database: the latest production
forecast and alert run, observations, catchments and exposure. Scenario runs are computed
live by the production model. Kingfisher maps exposure pathways; it does not predict health
outcomes and does not declare water safe or unsafe. Scenario outputs are planning
estimates, not predictions.

API docs: `/docs`. Health: `/health`.
