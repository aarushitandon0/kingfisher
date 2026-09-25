# Kingfisher demo image: PostGIS + FastAPI + the built frontend in one container, for a free
# Hugging Face Docker Space (runs as uid 1000, serves on 7860). Built from the bundle that
# scripts/build_space.py assembles - never from the repo root.

# Python 3.11 + the API's deps, built on the same Debian (bullseye) as the PostGIS image so
# the interpreter can be copied across as-is.
FROM python:3.11-slim-bullseye AS py
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

FROM postgis/postgis:16-3.4
COPY --from=py /usr/local /usr/local
COPY --from=py /opt/venv /opt/venv
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && ldconfig

RUN useradd -m -u 1000 user
ENV PATH=/usr/lib/postgresql/16/bin:/opt/venv/bin:$PATH \
    PGDATA=/home/user/pgdata \
    DATABASE_URL=postgresql+psycopg://kingfisher:kingfisher@127.0.0.1:5432/kingfisher \
    STATIC_DIR=/home/user/app/web \
    KINGFISHER_ENV=demo \
    SCENARIO_WARMUP=true \
    PYTHONUNBUFFERED=1

USER user
WORKDIR /home/user/app

# The database is restored at build time, so a cold start only has to start Postgres.
COPY --chown=user db/ /home/user/db/
RUN initdb -U kingfisher --auth=trust -D "$PGDATA" >/dev/null \
 && pg_ctl -D "$PGDATA" -o "-k /tmp -c listen_addresses=127.0.0.1" -w start \
 && createdb -h 127.0.0.1 -U kingfisher kingfisher \
 && pg_restore -h 127.0.0.1 -U kingfisher -d kingfisher --no-owner /home/user/db/kingfisher.dump \
 && gunzip -c /home/user/db/forecasts_production.csv.gz \
    | psql -h 127.0.0.1 -U kingfisher -d kingfisher -v ON_ERROR_STOP=1 \
        -c "\copy forecasts FROM STDIN WITH (FORMAT csv, HEADER true)" \
 && psql -h 127.0.0.1 -U kingfisher -d kingfisher -c "ANALYZE" >/dev/null \
 && pg_ctl -D "$PGDATA" -m fast -w stop \
 && rm -rf /home/user/db

COPY --chown=user app/ /home/user/app/
COPY --chown=user start.sh /home/user/start.sh

EXPOSE 7860
CMD ["bash", "/home/user/start.sh"]
