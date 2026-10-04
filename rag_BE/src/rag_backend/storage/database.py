"""Connection lifecycle and infrastructure readiness."""

from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from rag_backend.config import Settings

SCHEMA_REVISION = '0005_conversations'


@dataclass(frozen=True)
class DatabaseChecks:
    database: bool = False
    pgvector: bool = False
    schema: bool = False

    @property
    def ready(self) -> bool:
        return self.database and self.pgvector and self.schema


def create_database_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_timeout=5,
        hide_parameters=True,
        connect_args={
            'connect_timeout': 3,
            'options': '-c statement_timeout=5000',
        },
    )


def check_database(engine: Engine) -> DatabaseChecks:
    try:
        with engine.connect() as connection:
            connection.execute(text('SELECT 1'))
            vector_exists = bool(
                connection.scalar(
                    text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                )
            )
            version_table = connection.scalar(text("SELECT to_regclass('public.alembic_version')"))
            revision = None
            if version_table:
                revision = connection.scalar(text('SELECT version_num FROM public.alembic_version'))
            return DatabaseChecks(
                database=True, pgvector=vector_exists, schema=revision == SCHEMA_REVISION
            )
    except SQLAlchemyError:
        # The client gets check results, never driver errors or credentials.
        return DatabaseChecks()
