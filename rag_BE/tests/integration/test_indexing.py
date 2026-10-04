import json

import pytest
from sqlalchemy import text

from rag_backend.domain.errors import LostLease
from rag_backend.indexing.embeddings import EmbeddingError, Profile
from rag_backend.indexing.service import build_index
from rag_backend.jobs.runner import JobRunner

pytestmark = pytest.mark.integration


class FakeEmbedder:
    # Test-only synthetic vectors. The application never substitutes these for a real model.
    def profile(self):
        return Profile('test', 'synthetic', 'test-digest')

    def embed(self, texts, profile):
        return [[1.0] + [0.0] * 767 for _ in texts]


def prepare(system, upload, content=b'Oranges are orange.', source_id=None):
    receipt = upload(content, source_id=source_id)
    assert system.runner.run_once()['status'] == 'succeeded'
    assert system.runner.run_once()['status'] == 'succeeded'
    return receipt


def run_index(system, embedder=None):
    return JobRunner(system.queue, system.files, embedder=embedder or FakeEmbedder()).run_once()


def visible(system):
    with system.engine.connect() as connection:
        return list(
            connection.execute(text('SELECT * FROM published_retrieval_records')).mappings()
        )


def test_index_publication_fts_vector_and_original_links(system, upload):
    receipt = prepare(system, upload)
    assert not visible(system)
    assert run_index(system)['status'] == 'succeeded'
    rows = visible(system)
    assert len(rows) == 1 and rows[0]['version_id'] == receipt['version_id']
    with system.engine.connect() as connection:
        assert (
            connection.scalar(
                text(
                    'SELECT count(*) FROM published_retrieval_records '
                    "WHERE search_terms @@ plainto_tsquery('simple', 'oranges')"
                )
            )
            == 1
        )
        assert connection.scalar(
            text('SELECT embedding <=> CAST(:v AS vector) FROM published_retrieval_records'),
            {'v': json.dumps([1.0] + [0.0] * 767)},
        ) == pytest.approx(0)
        assert (
            connection.scalar(
                text(
                    'SELECT count(*) FROM retrieval_records r '
                    'JOIN evidence_chunks c ON c.id=r.chunk_id'
                )
            )
            == 1
        )
    assert (
        system.service.get_source(receipt['source_id'])['published_version_id']
        == receipt['version_id']
    )
    assert system.queue.payload(receipt['version_id'])['indexing_status'] == 'indexed'
    assert run_index(system) is None


def test_failed_update_keeps_old_publication_and_retry_can_replace_it(system, upload):
    old = prepare(system, upload)
    run_index(system)
    new = prepare(system, upload, b'New orange facts.', old['source_id'])

    class Broken(FakeEmbedder):
        def embed(self, *args):
            raise EmbeddingError('embedding_model_missing')

    assert run_index(system, Broken())['status'] == 'failed'
    assert visible(system)[0]['version_id'] == old['version_id']
    version = system.queue.payload(new['version_id'])
    assert version['indexing_status'] == 'failed' and version['preprocessing_status'] == 'approved'
    job = next(j for j in system.queue.list_for_version(new['version_id']) if j['kind'] == 'index')
    system.queue.retry(job['id'])
    assert run_index(system)['status'] == 'succeeded'
    assert visible(system)[0]['version_id'] == new['version_id']


def test_partial_vectors_roll_back_job_records_and_publication(system, upload):
    receipt = prepare(system, upload)
    job = system.queue.claim(('index',))
    result = build_index(system.engine, receipt['version_id'], FakeEmbedder(), lambda: None)
    result.vectors = []
    with pytest.raises(EmbeddingError):
        system.queue.complete_indexing(job['id'], job['lease_token'], result)
    assert not visible(system)
    assert system.queue.get(job['id'])['status'] == 'running'
    with system.engine.connect() as connection:
        assert connection.scalar(text('SELECT count(*) FROM retrieval_records')) == 0


def test_late_old_job_never_replaces_newer_published_version(system, upload):
    old = prepare(system, upload)
    old_job = system.queue.claim(('index',))
    old_result = build_index(system.engine, old['version_id'], FakeEmbedder(), lambda: None)
    new = prepare(system, upload, b'New information.', old['source_id'])
    assert run_index(system)['status'] == 'succeeded'
    system.queue.complete_indexing(old_job['id'], old_job['lease_token'], old_result)
    assert visible(system)[0]['version_id'] == new['version_id']


def test_deletion_hides_published_vectors_and_fences_pending_publish(system, upload):
    receipt = prepare(system, upload)
    run_index(system)
    new = prepare(system, upload, b'New text.', receipt['source_id'])
    job = system.queue.claim(('index',))
    result = build_index(system.engine, new['version_id'], FakeEmbedder(), lambda: None)
    system.service.delete(receipt['source_id'])
    assert not visible(system)
    with pytest.raises(LostLease):
        system.queue.complete_indexing(job['id'], job['lease_token'], result)


def test_expired_index_lease_cannot_publish_or_duplicate_records(system, upload):
    receipt = prepare(system, upload)
    job = system.queue.claim(('index',))
    result = build_index(system.engine, receipt['version_id'], FakeEmbedder(), lambda: None)
    with system.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE jobs SET lease_expires_at=clock_timestamp()-interval '1 second' "
                'WHERE id=:id'
            ),
            {'id': job['id']},
        )
    replacement = system.queue.claim(('index',))
    with pytest.raises(LostLease):
        system.queue.complete_indexing(job['id'], job['lease_token'], result)
    assert not visible(system)
    system.queue.complete_indexing(replacement['id'], replacement['lease_token'], result)
    with pytest.raises(LostLease):
        system.queue.complete_indexing(replacement['id'], replacement['lease_token'], result)
    assert len(visible(system)) == 1


def test_quarantined_source_has_no_index_job(system, upload):
    receipt = upload(b'not a pdf', filename='bad.pdf')
    system.runner.run_once()
    system.runner.run_once()
    assert all(j['kind'] != 'index' for j in system.queue.list_for_version(receipt['version_id']))
    assert not visible(system)
