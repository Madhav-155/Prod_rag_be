import asyncio
import os
import re
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from rag_backend.config import BACKEND_ROOT, Settings
from rag_backend.ingestion.service import SourceService
from rag_backend.jobs.runner import JobRunner
from rag_backend.jobs.service import JobQueue
from rag_backend.storage.database import create_database_engine
from rag_backend.storage.files import LocalFileStore


@pytest.fixture
def system(tmp_path, request):
    if os.getenv('RAG_RUN_DB_TESTS') != '1':
        pytest.skip('Set RAG_RUN_DB_TESTS=1 to run isolated PostgreSQL integration tests')
    settings = Settings()
    admin = create_database_engine(settings)
    schema = f'test_s2_{uuid4().hex}'
    assert re.fullmatch(r'test_s2_[0-9a-f]{32}', schema)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA {schema}'))
    engine = create_engine(
        settings.database_url(),
        connect_args={
            'options': f'-c search_path={schema},public -c statement_timeout=5000',
            'connect_timeout': 3,
        },
    )
    try:
        with engine.begin() as connection:
            config = Config(str(BACKEND_ROOT / 'alembic.ini'))
            config.attributes['connection'] = connection
            config.attributes['version_table_schema'] = schema
            command.upgrade(config, getattr(request, 'param', 'head'))
        files = LocalFileStore(tmp_path, 1024)
        service = SourceService(engine, files)
        queue = JobQueue(engine, lease_seconds=60)
        yield SimpleNamespace(
            schema=schema,
            engine=engine,
            files=files,
            service=service,
            queue=queue,
            runner=JobRunner(queue, files),
            settings=settings,
        )
    finally:
        engine.dispose()
        # Only the generated, validated per-test schema is removed. Public/user data is untouched.
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        admin.dispose()


def stage(system, data=b'hello world'):
    async def chunks():
        yield data

    return asyncio.run(system.files.receive(chunks()))


@pytest.fixture
def upload(system):
    def accept(data=b'hello world', *, filename='notes.txt', key=None, source_id=None):
        staged = stage(system, data)
        try:
            return system.service.accept(staged, filename, key or uuid4(), source_id)
        finally:
            system.files.discard(staged)

    return accept
