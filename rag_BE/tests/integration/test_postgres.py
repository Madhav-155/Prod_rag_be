import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from rag_backend.config import Settings
from rag_backend.main import create_app
from rag_backend.storage.database import check_database, create_database_engine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv('RAG_RUN_DB_TESTS') != '1',
        reason='Set RAG_RUN_DB_TESTS=1 after starting PostgreSQL and running migrations',
    ),
]


def test_database_has_migration_and_vector_extension():
    engine = create_database_engine(Settings())
    try:
        assert check_database(engine).ready
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT '[1,2]'::vector <-> '[1,2]'::vector")) == 0
    finally:
        engine.dispose()


def test_api_readiness_against_real_postgres():
    with TestClient(create_app()) as client:
        response = client.get('/health/ready')
    assert response.status_code == 200
    assert response.json()['checks'] == {
        'database': True,
        'pgvector': True,
        'schema': True,
    }
