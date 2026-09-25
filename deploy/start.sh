#!/usr/bin/env bash
# Start Postgres, then the API (which also serves the frontend) in the foreground.
set -euo pipefail
pg_ctl -D "$PGDATA" -o "-k /tmp -c listen_addresses=127.0.0.1" -l /tmp/postgres.log -w start
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-7860}" --proxy-headers
