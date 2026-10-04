from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import text

from rag_backend.domain.errors import DomainError, LostLease

pytestmark = pytest.mark.integration


def row_count(system, table):
    assert table in {'sources', 'source_versions', 'jobs'}
    with system.engine.connect() as connection:
        return connection.scalar(text(f'SELECT count(*) FROM {table}'))


def expire(system, job_id):
    with system.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE jobs SET lease_expires_at = clock_timestamp() - interval '1 second' "
                'WHERE id=:id'
            ),
            {'id': job_id},
        )


def make_available(system, job_id):
    with system.engine.begin() as connection:
        connection.execute(
            text('UPDATE jobs SET available_at=clock_timestamp() WHERE id=:id'), {'id': job_id}
        )


def test_upload_creates_version_original_and_job_atomically(system, upload):
    receipt = upload()
    assert (
        row_count(system, 'sources')
        == row_count(system, 'source_versions')
        == row_count(system, 'jobs')
        == 1
    )
    version = system.queue.payload(receipt['version_id'])
    assert version['state'] == 'stored'
    assert system.files.resolve(version['storage_key']).read_bytes() == b'hello world'
    result = system.runner.run_once()
    assert result['status'] == 'succeeded'
    assert system.queue.payload(receipt['version_id'])['state'] == 'verified'
    assert system.runner.run_once()['status'] == 'succeeded'
    assert system.queue.payload(receipt['version_id'])['preprocessing_status'] == 'approved'
    assert system.runner.run_once() is None


def test_repeated_request_key_reuses_version_and_job(system, upload):
    key = uuid4()
    original = upload(key=key)
    replay = upload(key=key)
    assert replay['replayed']
    assert replay['version_id'] == original['version_id']
    assert replay['job_id'] == original['job_id']
    assert row_count(system, 'source_versions') == row_count(system, 'jobs') == 1
    assert len(list(system.files.root.rglob('*.bin'))) == 1
    assert list(system.files.root.rglob('*.part')) == []


def test_same_key_changed_content_or_filename_conflicts(system, upload):
    key = uuid4()
    upload(key=key)
    with pytest.raises(DomainError) as error:
        upload(b'changed', key=key)
    assert error.value.status == 409
    with pytest.raises(DomainError):
        upload(key=key, filename='other.txt')
    assert row_count(system, 'source_versions') == 1


def test_concurrent_duplicate_uploads_create_only_one_version(system, upload):
    key = uuid4()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: upload(key=key), range(2)))
    assert results[0]['version_id'] == results[1]['version_id']
    assert sorted(r['replayed'] for r in results) == [False, True]
    assert row_count(system, 'jobs') == 1


def test_updates_have_distinct_ordered_immutable_versions(system, upload):
    first = upload(b'first')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(lambda data: upload(data, source_id=first['source_id']), [b'second', b'third'])
        )
    versions = system.service.versions(first['source_id'], 50, 0)
    assert [v['number'] for v in versions] == [3, 2, 1]
    old = system.queue.payload(first['version_id'])
    assert system.files.resolve(old['storage_key']).read_bytes() == b'first'


def test_failed_file_promotion_rolls_back_database(system, upload, monkeypatch):
    def fail(*args):
        raise OSError('simulated full disk')

    monkeypatch.setattr(system.files, 'promote', fail)
    with pytest.raises(OSError):
        upload()
    assert row_count(system, 'sources') == 0
    assert row_count(system, 'source_versions') == 0
    assert row_count(system, 'jobs') == 0
    assert list(system.files.root.rglob('*.part')) == []


def test_workers_cannot_claim_same_job(system, upload):
    receipt = upload()
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: system.queue.claim(), range(2)))
    actual = [c for c in claims if c is not None]
    assert len(actual) == 1
    assert actual[0]['id'] == receipt['job_id']
    assert actual[0]['attempts'] == 1


def test_expired_worker_is_fenced_after_reclaim(system, upload):
    upload()
    old = system.queue.claim()
    expire(system, old['id'])
    new = system.queue.claim()
    assert new['id'] == old['id']
    assert new['lease_token'] != old['lease_token']
    assert new['attempts'] == 2
    with pytest.raises(LostLease):
        system.queue.complete(old['id'], old['lease_token'])
    with pytest.raises(LostLease):
        system.queue.heartbeat(old['id'], old['lease_token'])
    system.queue.complete(new['id'], new['lease_token'])


def test_heartbeat_renews_lease_but_wrong_token_cannot(system, upload):
    upload()
    job = system.queue.claim()
    system.queue.heartbeat(job['id'], job['lease_token'])
    assert system.queue.get(job['id'])['lease_expires_at'] >= job['lease_expires_at']
    with pytest.raises(LostLease):
        system.queue.heartbeat(job['id'], uuid4())


def test_transient_failures_back_off_and_stop_at_limit(system, upload):
    receipt = upload()
    for attempt in range(1, 4):
        job = system.queue.claim()
        assert job['attempts'] == attempt
        status = system.queue.fail(job['id'], job['lease_token'], 'temporary_read', retryable=True)
        assert status == ('queued' if attempt < 3 else 'failed')
        if attempt < 3:
            assert system.queue.claim() is None
            make_available(system, job['id'])
    assert system.queue.payload(receipt['version_id'])['state'] == 'failed'
    with pytest.raises(DomainError):
        system.queue.retry(receipt['job_id'])


def test_crashes_at_attempt_limit_become_terminal(system, upload):
    receipt = upload()
    for _ in range(3):
        job = system.queue.claim()
        assert job
        expire(system, job['id'])
    assert system.queue.claim() is None
    assert system.queue.get(receipt['job_id'])['status'] == 'failed'
    assert system.queue.payload(receipt['version_id'])['state'] == 'failed'


def test_corrupted_original_fails_integrity_and_can_retry_after_restoration(system, upload):
    receipt = upload()
    version = system.queue.payload(receipt['version_id'])
    path = system.files.resolve(version['storage_key'])
    path.write_bytes(b'changed')
    assert system.runner.run_once()['status'] == 'failed'
    assert system.queue.get(receipt['job_id'])['last_error_code'] == 'original_integrity_failed'
    path.write_bytes(b'hello world')
    retry = system.queue.retry(receipt['job_id'])
    assert retry['attempts'] == 1
    assert system.runner.run_once()['status'] == 'succeeded'


def test_deleting_source_cancels_running_job_and_fences_worker(system, upload):
    receipt = upload()
    job = system.queue.claim()
    result = system.service.delete(receipt['source_id'])
    assert result == {'deleted': True, 'original_files_retained': True}
    assert system.queue.get(job['id'])['status'] == 'cancelled'
    assert system.service.list_sources(50, 0) == []
    with pytest.raises(LostLease):
        system.queue.complete(job['id'], job['lease_token'])
    with pytest.raises(DomainError):
        system.queue.retry(job['id'])
    with pytest.raises(DomainError):
        upload(source_id=receipt['source_id'])


def test_unknown_source_upload_does_not_leave_rows(system, upload):
    with pytest.raises(DomainError) as error:
        upload(source_id=uuid4())
    assert error.value.status == 404
    assert row_count(system, 'source_versions') == 0
