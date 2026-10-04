"""Opt-in smoke test: real local embeddings, isolated database, synthetic documents."""

import json
import os

import pytest
from sqlalchemy import text

from rag_backend.config import Settings
from rag_backend.indexing.embeddings import OllamaEmbedder
from rag_backend.jobs.runner import JobRunner
from rag_backend.retrieval.contracts import SearchRequest
from rag_backend.retrieval.repository import RetrievalRepository
from rag_backend.retrieval.reranking import LocalReranker
from rag_backend.retrieval.service import RetrievalService

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv('RAG_RUN_MODEL_TESTS') != '1',
        reason='Set RAG_RUN_MODEL_TESTS=1 for local model smoke',
    ),
]


def test_real_local_embedding_publication_and_semantic_match(system, upload):
    adapter = OllamaEmbedder(timeout=60)
    runner = JobRunner(system.queue, system.files, embedder=adapter)
    astronomy = upload(
        b'The planet Saturn has rings composed of ice particles and rock. '
        b'Saturn orbits the Sun and is a gas giant.'
    )
    assert runner.run_once()['status'] == 'succeeded'
    assert runner.run_once()['status'] == 'succeeded'
    assert runner.run_once()['status'] == 'succeeded'
    cooking = upload(
        b'To bake bread, mix flour, water, salt and yeast. Knead the dough, '
        b'let it rise, then bake it in a hot oven.'
    )
    assert runner.run_once()['status'] == 'succeeded'
    assert runner.run_once()['status'] == 'succeeded'
    assert runner.run_once()['status'] == 'succeeded'
    profile = adapter.profile()
    query = adapter.embed_query('What are the rings around Saturn made of?', profile)
    with system.engine.connect() as connection:
        rows = list(
            connection.execute(
                text(
                    'SELECT version_id, vector_dims(embedding) AS dims '
                    'FROM published_retrieval_records WHERE profile_id=:profile '
                    'ORDER BY embedding <=> CAST(:query AS vector)'
                ),
                {'profile': profile.id, 'query': json.dumps(query)},
            ).mappings()
        )
    assert [row['version_id'] for row in rows] == [astronomy['version_id'], cooking['version_id']]
    assert all(row['dims'] == 768 for row in rows)
    search = RetrievalService(
        RetrievalRepository(system.engine),
        adapter,
        LocalReranker(system.settings.data_dir / 'models' / 'minilm-l6-v2'),
    )
    result = search.search(SearchRequest(query='What are Saturn rings made of?', limit=2))
    assert not result.warnings
    assert result.trace['ranking'] == 'cross_encoder'
    assert result.hits[0].version_id == astronomy['version_id']
    assert result.hits[0].rerank_score > result.hits[1].rerank_score
    assert result.hits[0].citations and result.hits[0].citations[0].locator


def test_real_reranker_scores_evidence_after_first_token_window():
    reranker = LocalReranker(Settings().data_dir / 'models' / 'minilm-l6-v2')
    filler = 'The baker prepares flour yeast dough and bread. ' * 70
    scores = reranker.score(
        'What are Saturn rings made of?',
        [filler, filler + ' Saturn rings consist of ice particles and rocky debris.'],
    )
    assert len(scores) == 2 and scores[1] > scores[0]
