"""Alembic environment. Reads DATABASE_URL via core.settings; no URL in alembic.ini."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from core.settings import get_settings

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Schema is defined in migration scripts, not in ORM models, so there is no
# target_metadata to autogenerate against.
target_metadata = None

# PostGIS internals must never be diffed or dropped by a migration.
EXCLUDED_TABLES = {"spatial_ref_sys", "geography_columns", "geometry_columns", "raster_columns",
                   "raster_overviews"}


def include_object(obj, name, type_, reflected, compare_to):  # noqa: ANN001, ANN201
    return not (type_ == "table" and name in EXCLUDED_TABLES)


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
