"""Migration credentials come from the same settings as the API."""

from alembic import context

from rag_backend.config import Settings
from rag_backend.storage.database import create_database_engine

settings = Settings()


def run_online(connection):
    context.configure(
        connection=connection,
        target_metadata=None,
        version_table_schema=context.config.attributes.get('version_table_schema'),
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(
        url=settings.database_url(),
        target_metadata=None,
        literal_binds=True,
        dialect_opts={'paramstyle': 'named'},
    )
    with context.begin_transaction():
        context.run_migrations()
elif context.config.attributes.get('connection') is not None:
    # Tests supply a connection scoped to a freshly created disposable schema.
    run_online(context.config.attributes['connection'])
else:
    # Migrations own their engine and run explicitly, never on API startup.
    engine = create_database_engine(settings)
    try:
        with engine.connect() as connection:
            run_online(connection)
    finally:
        engine.dispose()
