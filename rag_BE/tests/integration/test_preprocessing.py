from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from rag_backend.api.dependencies import get_job_queue, get_source_service
from rag_backend.config import BACKEND_ROOT
from rag_backend.domain.errors import DomainError, LostLease
from rag_backend.ingestion.contracts import Options
from rag_backend.ingestion.pipeline import run
from rag_backend.main import create_app

pytestmark = pytest.mark.integration


def process(system):
    assert system.runner.run_once()['status'] == 'succeeded'
    return system.runner.run_once()


def parse_claim(system, upload):
    receipt = upload(b'# Heading\n\nThe number is 42.', filename='notes.md')
    system.runner.run_once()
    job = system.queue.claim()
    version = system.queue.payload(receipt['version_id'])
    result = run(
        system.files.resolve(version['storage_key']),
        version['original_name'],
        version['id'],
        Options(),
    )
    return receipt, job, result


def test_full_text_pipeline_persists_original_evidence(system, upload):
    receipt = upload(b'# Heading\n\nThe number is 42.', filename='notes.md')
    assert process(system)['status'] == 'succeeded'
    version = system.queue.payload(receipt['version_id'])
    assert version['state'] == 'verified'
    assert version['preprocessing_status'] == 'approved'
    assert version['quality_report']['searchable'] is False
    blocks = system.service.evidence(receipt['source_id'], receipt['version_id'], 'blocks', 100, 0)
    chunks = system.service.evidence(receipt['source_id'], receipt['version_id'], 'chunks', 100, 0)
    assert blocks and chunks
    block_ids = {str(block['id']) for block in blocks}
    assert all(span['block_id'] in block_ids for chunk in chunks for span in chunk['spans'])
    assert len(system.queue.list_for_version(receipt['version_id'])) == 3


def test_pdf_pipeline_survives_subprocess_and_storage(system, upload, pdf_bytes):
    system.files.max_bytes = 100_000
    receipt = upload(pdf_bytes(), filename='report.pdf')
    assert process(system)['status'] == 'succeeded'
    blocks = system.service.evidence(receipt['source_id'], receipt['version_id'], 'blocks', 100, 0)
    assert all(block['locator']['page'] == 1 for block in blocks)


def test_quarantine_keeps_canonical_structure_but_no_chunks(system, upload):
    receipt = upload(b'| Name | Value |\n| --- | --- |\n| A | 42 |\n', filename='table.md')
    assert process(system)['status'] == 'failed'
    version = system.queue.payload(receipt['version_id'])
    assert version['preprocessing_status'] == 'quarantined'
    assert version['quality_report']['reasons'] == ['mixed_content_processing_deferred']
    assert system.service.evidence(receipt['source_id'], receipt['version_id'], 'blocks', 100, 0)
    assert (
        system.service.evidence(receipt['source_id'], receipt['version_id'], 'chunks', 100, 0) == []
    )
    job = next(
        j for j in system.queue.list_for_version(receipt['version_id']) if j['kind'] == 'preprocess'
    )
    with pytest.raises(DomainError, match='corrected new version'):
        system.queue.retry(job['id'])


def test_failed_new_version_does_not_change_old_evidence(system, upload):
    old = upload()
    process(system)
    before = system.service.evidence(old['source_id'], old['version_id'], 'blocks', 100, 0)
    new = upload(b'not a PDF', filename='bad.pdf', source_id=old['source_id'])
    assert process(system)['status'] == 'failed'
    assert system.queue.payload(new['version_id'])['preprocessing_status'] == 'quarantined'
    assert system.queue.payload(old['version_id'])['preprocessing_status'] == 'approved'
    assert system.service.evidence(old['source_id'], old['version_id'], 'blocks', 100, 0) == before


def test_invalid_span_rolls_back_job_completion_and_all_evidence(system, upload):
    receipt, job, result = parse_claim(system, upload)
    result.chunks[0].spans[0].block_id = uuid4()
    with pytest.raises(ValueError, match='evidence reference'):
        system.queue.complete_preprocessing(job['id'], job['lease_token'], result)
    assert system.queue.get(job['id'])['status'] == 'running'
    assert (
        system.service.evidence(receipt['source_id'], receipt['version_id'], 'blocks', 100, 0) == []
    )


def test_stale_preprocessor_cannot_write_after_takeover(system, upload):
    receipt, old, result = parse_claim(system, upload)
    with system.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE jobs SET lease_expires_at=clock_timestamp()-interval '1 second' "
                'WHERE id=:id'
            ),
            {'id': old['id']},
        )
    new = system.queue.claim()
    with pytest.raises(LostLease):
        system.queue.complete_preprocessing(old['id'], old['lease_token'], result)
    assert (
        system.service.evidence(receipt['source_id'], receipt['version_id'], 'blocks', 100, 0) == []
    )
    assert system.queue.complete_preprocessing(new['id'], new['lease_token'], result) == 'succeeded'
    count = len(
        system.service.evidence(receipt['source_id'], receipt['version_id'], 'blocks', 100, 0)
    )
    with pytest.raises(LostLease):
        system.queue.complete_preprocessing(new['id'], new['lease_token'], result)
    assert (
        len(system.service.evidence(receipt['source_id'], receipt['version_id'], 'blocks', 100, 0))
        == count
    )


def test_delete_during_parse_prevents_persistence(system, upload):
    receipt, job, result = parse_claim(system, upload)
    system.service.delete(receipt['source_id'])
    with pytest.raises(LostLease):
        system.queue.complete_preprocessing(job['id'], job['lease_token'], result)
    with system.engine.connect() as connection:
        assert connection.scalar(text('SELECT count(*) FROM canonical_blocks')) == 0


def test_api_exposes_quality_and_version_scoped_evidence(system, upload):
    receipt = upload()
    process(system)
    app = create_app(system.settings)
    app.dependency_overrides[get_source_service] = lambda: system.service
    app.dependency_overrides[get_job_queue] = lambda: system.queue
    prefix = f'/sources/{receipt["source_id"]}/versions/{receipt["version_id"]}'
    with TestClient(app) as client:
        assert client.get(prefix + '/blocks').status_code == 200
        assert client.get(prefix + '/chunks').json()
        assert (
            len(client.get('/jobs', params={'version_id': str(receipt['version_id'])}).json()) == 3
        )
        assert (
            client.get(f'/sources/{uuid4()}/versions/{receipt["version_id"]}/blocks').status_code
            == 404
        )
        client.delete('/sources/' + str(receipt['source_id']))
        assert client.get(prefix + '/blocks').status_code == 404
        assert client.get(prefix + '/chunks').status_code == 404


@pytest.mark.parametrize('system', ['0002_sources_jobs'], indirect=True)
def test_migration_backfills_preprocess_jobs_for_existing_verified_originals(system, upload):
    receipt = upload()
    with system.engine.begin() as connection:
        connection.execute(
            text("UPDATE source_versions SET state='verified' WHERE id=:id"),
            {'id': receipt['version_id']},
        )
        connection.execute(
            text("UPDATE jobs SET status='succeeded' WHERE id=:id"), {'id': receipt['job_id']}
        )
        config = Config(str(BACKEND_ROOT / 'alembic.ini'))
        config.attributes['connection'] = connection
        config.attributes['version_table_schema'] = system.schema
        command.upgrade(config, 'head')
    jobs = system.queue.list_for_version(receipt['version_id'])
    assert [j['kind'] for j in jobs] == ['verify_upload', 'preprocess']
    assert system.runner.run_once()['status'] == 'succeeded'
