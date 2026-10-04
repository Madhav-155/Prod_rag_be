import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from rag_backend.api.dependencies import get_retrieval_service
from rag_backend.indexing.embeddings import EmbeddingError, Profile
from rag_backend.jobs.runner import JobRunner
from rag_backend.main import create_app
from rag_backend.retrieval.contracts import SearchRequest
from rag_backend.retrieval.repository import RetrievalRepository
from rag_backend.retrieval.reranking import RerankError
from rag_backend.retrieval.service import RetrievalService

pytestmark = pytest.mark.integration


class Embeddings:
    def profile(self):
        return Profile('test', 'retrieval', 'v1')

    def embed(self, texts, profile):
        return [
            [1.0, 0.0] + [0.0] * 766 if 'Saturn' in value else [0.0, 1.0] + [0.0] * 766
            for value in texts
        ]

    def embed_query(self, query, profile):
        return self.embed([query], profile)[0]


class Reranker:
    profile = 'test-reranker'

    def score(self, query, documents):
        return [10.0 if query in doc else -10.0 for doc in documents]


def indexed(system, upload, content=b'Saturn has icy rings.', source_id=None):
    receipt = upload(content, source_id=source_id)
    runner = JobRunner(system.queue, system.files, embedder=Embeddings())
    for _ in range(3):
        assert runner.run_once()['status'] == 'succeeded'
    return receipt


def service(system, reranker=None, embedder=None):
    return RetrievalService(
        RetrievalRepository(system.engine), embedder or Embeddings(), reranker or Reranker()
    )


def test_hybrid_search_returns_original_citations_and_trace(system, upload):
    receipt = indexed(system, upload)
    indexed(system, upload, b'Bread needs flour and yeast.')
    result = service(system).search(SearchRequest(query='Saturn', limit=2))
    assert result.status == 'ok' and result.hits[0].source_id == receipt['source_id']
    assert result.hits[0].citations[0].excerpt == 'Saturn has icy rings.'
    assert result.hits[0].citations[0].locator
    assert result.trace['ranking'] == 'cross_encoder'
    assert result.hits[0].rrf_score == pytest.approx(2 / 61)
    assert result.trace['returned_count'] == 2 and not result.warnings


def test_source_filter_applies_to_both_channels(system, upload):
    first = indexed(system, upload)
    second = indexed(system, upload, b'Bread ingredients.')
    result = service(system).search(SearchRequest(query='Saturn', source_ids=[second['source_id']]))
    assert [h.source_id for h in result.hits] == [second['source_id']]
    assert all(h.source_id != first['source_id'] for h in result.hits)


def test_deleted_and_unpublished_versions_never_return(system, upload):
    original = indexed(system, upload)
    pending = upload(b'Saturn replacement not published.', source_id=original['source_id'])
    result = service(system).search(SearchRequest(query='Saturn'))
    assert [h.version_id for h in result.hits] == [original['version_id']]
    assert all(h.version_id != pending['version_id'] for h in result.hits)
    system.service.delete(original['source_id'])
    assert not service(system).search(SearchRequest(query='Saturn')).hits


def test_different_embedding_profile_cannot_mix_vectors(system, upload):
    indexed(system, upload)

    class Changed(Embeddings):
        def profile(self):
            return Profile('test', 'retrieval', 'v2')

    assert (
        service(system, embedder=Changed()).search(SearchRequest(query='Saturn')).status
        == 'no_results'
    )


def test_deletion_during_reranking_is_rechecked_before_hydration(system, upload):
    receipt = indexed(system, upload)

    class Deletes(Reranker):
        def score(self, query, documents):
            system.service.delete(receipt['source_id'])
            return super().score(query, documents)

    assert not service(system, Deletes()).search(SearchRequest(query='Saturn')).hits


def test_new_publication_during_reranking_invalidates_old_candidates(system, upload):
    old = indexed(system, upload)

    class Updates(Reranker):
        def score(self, query, documents):
            indexed(system, upload, b'Saturn updated evidence.', old['source_id'])
            return super().score(query, documents)

    result = service(system, Updates()).search(SearchRequest(query='Saturn'))
    assert not result.hits


def test_reranker_outage_returns_marked_rrf_not_fake_scores(system, upload):
    indexed(system, upload)

    class Broken(Reranker):
        def score(self, *args):
            raise RerankError('reranker_timeout')

    result = service(system, Broken()).search(SearchRequest(query='Saturn'))
    assert result.hits[0].rerank_score is None
    assert result.warnings == ['reranker_timeout', 'using_rrf_order']
    assert result.trace['ranking'] == 'rrf'
    assert 'reranking' in result.trace['stage_ms']


def test_invalid_reranker_output_does_not_escape_into_results(system, upload):
    indexed(system, upload)

    class Invalid(Reranker):
        def score(self, *args):
            return [float('nan')]

    result = service(system, Invalid()).search(SearchRequest(query='Saturn'))
    assert 'reranker_invalid_scores' in result.warnings
    assert result.hits[0].rerank_score is None


def test_sql_like_input_is_data_and_api_contract_is_bounded(system, upload):
    indexed(system, upload)
    app = create_app(system.settings)
    app.dependency_overrides[get_retrieval_service] = lambda: service(system)
    with TestClient(app) as client:
        response = client.post('/search', json={'query': "Saturn'; DROP TABLE sources; --"})
        assert response.status_code == 200
        assert 'storage_key' not in response.text and 'lease_token' not in response.text
        assert client.post('/search', json={'query': 'Saturn', 'limit': 100}).status_code == 422
        assert (
            client.post('/search', json={'query': 'Saturn', 'source_ids': []}).json()['hits'] == []
        )
    with system.engine.connect() as connection:
        assert connection.scalar(text('SELECT count(*) FROM sources')) == 1


def test_model_outage_is_safe_503(system):
    class Broken(Embeddings):
        def profile(self):
            raise EmbeddingError('embedding_unavailable', True)

    app = create_app(system.settings)
    app.dependency_overrides[get_retrieval_service] = lambda: service(system, embedder=Broken())
    with TestClient(app) as client:
        response = client.post('/search', json={'query': 'Saturn'})
        assert response.status_code == 503
        assert response.json()['error']['code'] == 'embedding_unavailable'
