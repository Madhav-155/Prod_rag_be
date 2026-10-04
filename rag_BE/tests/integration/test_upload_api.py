from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from rag_backend.api.dependencies import get_job_queue, get_source_service
from rag_backend.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def client(system):
    app = create_app(system.settings)
    app.dependency_overrides[get_source_service] = lambda: system.service
    app.dependency_overrides[get_job_queue] = lambda: system.queue
    with TestClient(app) as client:
        yield client


def headers(key=None):
    return {'Content-Type': 'application/octet-stream', 'Idempotency-Key': str(key or uuid4())}


def test_upload_poll_version_and_delete_contract(client, system):
    response = client.post('/sources?filename=notes.txt', content=b'notes', headers=headers())
    assert response.status_code == 202
    receipt = response.json()
    job_url = '/jobs/' + receipt['job_id']
    job = client.get(job_url).json()
    assert job['status'] == 'queued'
    assert 'lease_token' not in job
    system.runner.run_once()
    assert client.get(job_url).json()['status'] == 'succeeded'
    source_url = '/sources/' + receipt['source_id']
    versions = client.get(source_url + '/versions').json()
    assert versions[0]['state'] == 'verified'
    assert 'storage_key' not in versions[0]
    assert 'upload_key' not in versions[0]
    assert client.delete(source_url).json()['original_files_retained']
    assert client.get(source_url).status_code == 404
    assert client.get(source_url + '/versions').status_code == 404


def test_reupload_route_and_replay(client):
    first = client.post('/sources?filename=a.txt', content=b'a', headers=headers()).json()
    url = '/sources/' + first['source_id'] + '/versions?filename=a.txt'
    key = uuid4()
    second = client.post(url, content=b'b', headers=headers(key))
    assert second.status_code == 202
    assert second.json()['version_number'] == 2
    replay = client.post(url, content=b'b', headers=headers(key))
    assert replay.json()['replayed']


@pytest.mark.parametrize(
    ('name', 'data', 'status'),
    [
        ('notes.exe', b'file', 415),
        ('../notes.txt', b'file', 422),
        ('notes.txt', b'', 422),
        ('notes.txt', b'x' * 1025, 413),
    ],
)
def test_invalid_uploads(client, system, name, data, status):
    response = client.post('/sources', params={'filename': name}, content=data, headers=headers())
    assert response.status_code == status
    assert system.service.list_sources(50, 0) == []


def test_missing_idempotency_key_rejected(client):
    response = client.post(
        '/sources?filename=a.txt',
        content=b'a',
        headers={'Content-Type': 'application/octet-stream'},
    )
    assert response.status_code == 422


def test_multipart_not_accepted_as_binary(client):
    response = client.post(
        '/sources?filename=a.txt',
        content=b'a',
        headers={'Content-Type': 'multipart/form-data', 'Idempotency-Key': str(uuid4())},
    )
    assert response.status_code == 415


def test_streamed_body_limit_is_enforced_without_content_length(client):
    response = client.post(
        '/sources?filename=a.txt', content=iter([b'a' * 600, b'b' * 600]), headers=headers()
    )
    assert response.status_code == 413


def test_missing_job_and_invalid_retry(client):
    assert client.get('/jobs/' + str(uuid4())).status_code == 404
    receipt = client.post('/sources?filename=a.txt', content=b'a', headers=headers()).json()
    assert client.post('/jobs/' + receipt['job_id'] + '/retry').status_code == 409
