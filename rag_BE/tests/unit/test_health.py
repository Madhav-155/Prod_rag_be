import pytest
from fastapi.testclient import TestClient

from rag_backend.api.dependencies import get_database_checks
from rag_backend.config import Settings
from rag_backend.main import create_app
from rag_backend.storage.database import DatabaseChecks


def app_for(checks):
    app = create_app(Settings(_env_file=None, database_password='test'))
    app.dependency_overrides[get_database_checks] = lambda: checks
    return app


def test_liveness_works_without_database():
    with TestClient(app_for(DatabaseChecks())) as client:
        response = client.get('/health/live')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


@pytest.mark.parametrize(
    'checks',
    [
        DatabaseChecks(),
        DatabaseChecks(database=True),
        DatabaseChecks(database=True, pgvector=True),
        DatabaseChecks(database=True, schema=True),
    ],
)
def test_readiness_fails_closed_for_missing_requirements(checks):
    with TestClient(app_for(checks)) as client:
        response = client.get('/health/ready')
    assert response.status_code == 503
    assert response.json()['status'] == 'not_ready'
    assert 'password' not in response.text


def test_ready_when_all_requirements_pass():
    with TestClient(app_for(DatabaseChecks(True, True, True))) as client:
        response = client.get('/health/ready')
        spec = client.get('/openapi.json').json()
    assert response.status_code == 200
    assert response.json()['status'] == 'ready'
    assert '503' in spec['paths']['/health/ready']['get']['responses']
