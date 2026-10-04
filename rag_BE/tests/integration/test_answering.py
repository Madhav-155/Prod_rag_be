import json
import os

import pytest
from fastapi.testclient import TestClient

from rag_backend.answering.contracts import AnswerRequest
from rag_backend.answering.nli import LocalNLI
from rag_backend.answering.providers import Generated, OllamaGenerator
from rag_backend.answering.service import AnswerService
from rag_backend.api.dependencies import get_answer_service
from rag_backend.config import Settings
from rag_backend.indexing.embeddings import OllamaEmbedder, Profile
from rag_backend.jobs.runner import JobRunner
from rag_backend.main import create_app
from rag_backend.retrieval.repository import RetrievalRepository
from rag_backend.retrieval.reranking import LocalReranker
from rag_backend.retrieval.service import RetrievalService

pytestmark = pytest.mark.integration


class Embeddings:
    def profile(self):
        return Profile('test', 'answering', 'v1')

    def embed(self, texts, profile):
        return [[1.0] + [0.0] * 767 for _ in texts]

    def embed_query(self, query, profile):
        return self.embed([query], profile)[0]


class Reranker:
    profile = 'synthetic'

    def score(self, query, documents):
        return [5.0] * len(documents)


class NLI:
    profile = 'synthetic'

    def predict(self, pairs):
        return [{'entailment': 0.99, 'neutral': 0.005, 'contradiction': 0.005} for _ in pairs]


class Generator:
    def generate(self, messages, schema):
        return Generated(
            json.dumps(
                {
                    'status': 'answer',
                    'claims': [{'text': 'Saturn rings contain ice.', 'evidence_ids': ['E1']}],
                }
            ),
            'synthetic',
        )


def build(system, upload, embedder=None, reranker=None, generator=None, nli=None):
    embedder = embedder or Embeddings()
    receipt = upload(b'Saturn rings contain ice.')
    runner = JobRunner(system.queue, system.files, embedder=embedder)
    for _ in range(3):
        assert runner.run_once()['status'] == 'succeeded'
    retrieval = RetrievalService(
        RetrievalRepository(system.engine), embedder, reranker or Reranker()
    )
    return receipt, AnswerService(retrieval, generator or Generator(), nli or NLI())


def test_answer_endpoint_upload_to_publication_to_original_citation(system, upload):
    receipt, answering = build(system, upload)
    app = create_app(system.settings)
    app.dependency_overrides[get_answer_service] = lambda: answering
    with TestClient(app) as client:
        response = client.post('/answers', json={'query': 'What do Saturn rings contain?'})
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'answer'
        assert data['evidence'][0]['hit']['version_id'] == str(receipt['version_id'])
        assert data['evidence'][0]['hit']['citations'][0]['excerpt'] == 'Saturn rings contain ice.'
        assert 'storage_key' not in response.text
        assert client.post('/answers', json={'query': 'Saturn', 'task': 'shell'}).status_code == 422
        assert (
            client.post('/answers', json={'query': 'Saturn', 'source_ids': []}).json()['status']
            == 'abstain'
        )


def test_real_database_deletion_during_generation_discards_answer(system, upload):
    receipt, answering = build(system, upload)

    class Deleting(Generator):
        def generate(self, messages, schema):
            system.service.delete(receipt['source_id'])
            return super().generate(messages, schema)

    answering.generator = Deleting()
    result = answering.answer(AnswerRequest(query='What do Saturn rings contain?'))
    assert result.status == 'abstain' and result.trace['reason'] == 'source_changed'
    assert not result.evidence


@pytest.mark.skipif(os.getenv('RAG_RUN_MODEL_TESTS') != '1', reason='Local model opt-in')
def test_real_nli_label_mapping_and_token_limit():
    from rag_backend.answering.nli import NLIError

    nli = LocalNLI(Settings().data_dir / 'models' / 'nli-minilm2')
    result = nli.predict(
        [
            {'premise': 'The door is open.', 'hypothesis': 'The door is open.'},
            {'premise': 'The door is open.', 'hypothesis': 'The door is closed.'},
            {'premise': 'The door is open.', 'hypothesis': 'The house is red.'},
        ]
    )
    assert max(result[0], key=result[0].get) == 'entailment'
    assert max(result[1], key=result[1].get) == 'contradiction'
    assert max(result[2], key=result[2].get) == 'neutral'
    with pytest.raises(NLIError):
        nli.predict([{'premise': 'The door is open. ' * 1000, 'hypothesis': 'It is open.'}])


@pytest.mark.skipif(os.getenv('RAG_RUN_MODEL_TESTS') != '1', reason='Local model opt-in')
def test_real_upload_retrieve_generate_validate_cited_answer(system, upload):
    settings = system.settings
    receipt, answering = build(
        system,
        upload,
        OllamaEmbedder(timeout=60),
        LocalReranker(settings.data_dir / 'models' / 'minilm-l6-v2'),
        OllamaGenerator(settings.generation_url, settings.generation_model),
        LocalNLI(settings.data_dir / 'models' / 'nli-minilm2'),
    )
    result = answering.answer(AnswerRequest(query='What do Saturn rings contain?'))
    assert result.status == 'answer', result.trace
    assert 'ice' in result.text.lower() and '[E1]' in result.text
    assert result.evidence[0].hit.version_id == receipt['version_id']
    assert result.trace['validation'][-1]['reason'] == 'validated'
    assert result.trace['generations'][0]['usage']['api_cost_usd'] == 0

    # Persist and replay the real validated response without a second model call.
    from types import SimpleNamespace
    from uuid import uuid4

    from rag_backend.conversations.repository import ConversationRepository
    from rag_backend.conversations.service import ConversationService, TurnRequest

    history = ConversationService(
        ConversationRepository(system.engine), SimpleNamespace(answer=lambda request: result)
    )
    cid = history.repository.create()['id']
    key = uuid4()
    request = TurnRequest(query='What do Saturn rings contain?')
    saved = history.turn(cid, key, request)
    assert saved['response']['status'] == 'answer'
    assert saved['response']['evidence'][0]['available']
    assert history.turn(cid, key, request)['replayed']
    assert history.history(cid)['turns'][0]['response']['text'] == result.text
