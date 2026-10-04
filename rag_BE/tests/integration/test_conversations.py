import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from rag_backend.answering.contracts import AnswerResponse, Evidence
from rag_backend.answering.planner import make_plan
from rag_backend.api.dependencies import get_conversation_service
from rag_backend.conversations.repository import ConversationRepository
from rag_backend.conversations.service import ConversationService, TurnRequest
from rag_backend.domain.errors import DomainError
from rag_backend.main import create_app
from rag_backend.retrieval.contracts import Citation, SearchHit

pytestmark = pytest.mark.integration


class Answering:
    def __init__(self):
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return AnswerResponse(
            request_id=uuid4(),
            status='abstain',
            text='No supported answer.',
            plan=make_plan(request),
            trace={'total_ms': 1, 'outcome': 'abstain'},
        )


def build(system):
    repository = ConversationRepository(system.engine)
    answering = Answering()
    return ConversationService(repository, answering), repository.create()['id']


def test_create_turn_replay_and_history_after_restart(system):
    service, cid = build(system)
    request, key = TurnRequest(query='Where is the evidence?'), uuid4()
    first = service.turn(cid, key, request)
    assert not first['replayed']
    assert service.turn(cid, key, request)['replayed']
    assert len(service.answering.requests) == 1
    restarted = ConversationService(ConversationRepository(system.engine), Answering())
    assert restarted.history(cid)['turns'][0]['query'] == request.query
    with system.engine.connect() as conn:
        assert conn.scalar(text('SELECT count(*) FROM answer_events')) == 1
        telemetry = conn.scalar(text('SELECT telemetry FROM answer_events'))
        assert request.query not in json.dumps(telemetry)
    with pytest.raises(DomainError, match='another request'):
        service.turn(cid, key, TurnRequest(query='Changed question'))


def test_context_only_includes_prior_user_questions_not_assistant_answers(system):
    service, cid = build(system)
    service.turn(cid, uuid4(), TurnRequest(query='First question'))
    service.turn(cid, uuid4(), TurnRequest(query='Second question'))
    assert service.answering.requests[1].conversation_context == 'First question\n'
    assert 'No supported answer' not in service.answering.requests[1].conversation_context


def test_parallel_admission_and_expired_lease_fence(system):
    service, cid = build(system)
    repo, request = service.repository, TurnRequest(query='Where is evidence?')
    old, _, _ = repo.reserve(cid, uuid4(), request)
    with pytest.raises(DomainError, match='already in progress'):
        repo.reserve(cid, uuid4(), request)
    with system.engine.begin() as conn:
        conn.execute(text("UPDATE conversations SET active_until=now()-interval '1 minute'"))
    new, _, _ = repo.reserve(cid, uuid4(), request)
    with pytest.raises(DomainError, match='superseded'):
        repo.complete(cid, old, {})
    assert old != new
    assert repo.history(cid)[0]['error_code'] == 'lease_expired'


def test_failure_clears_lease_but_same_key_never_regenerates(system):
    service, cid = build(system)
    key = uuid4()

    def fail(request):
        raise DomainError('model_unavailable', 'Unavailable.', 503)

    service.answering = SimpleNamespace(answer=fail)
    with pytest.raises(DomainError):
        service.turn(cid, key, TurnRequest(query='Question'))
    assert service.repository.history(cid)[0]['status'] == 'failed'
    with pytest.raises(DomainError, match='pending or failed'):
        service.turn(cid, key, TurnRequest(query='Question'))
    service.answering = Answering()
    assert (
        service.turn(cid, uuid4(), TurnRequest(query='Question'))['response']['status'] == 'abstain'
    )


def test_delete_during_generation_cannot_resurrect_history(system):
    service, cid = build(system)
    answering = service.answering

    def deleting(request):
        service.repository.delete(cid)
        return answering.answer(request)

    service.answering = SimpleNamespace(answer=deleting)
    with pytest.raises(DomainError, match='unavailable'):
        service.turn(cid, uuid4(), TurnRequest(query='Question'))
    with system.engine.connect() as conn:
        assert conn.scalar(text('SELECT count(*) FROM conversation_turns')) == 0
        assert conn.scalar(text('SELECT count(*) FROM answer_events')) == 0


def test_expired_history_hidden_and_explicit_purge_scoped(system):
    service, cid = build(system)
    service.turn(cid, uuid4(), TurnRequest(query='Question'))
    other = service.repository.create()['id']
    with system.engine.begin() as conn:
        conn.execute(
            text("UPDATE conversations SET expires_at=now()-interval '1 day' WHERE id=:id"),
            {'id': cid},
        )
    assert [r['id'] for r in service.repository.list()] == [other]
    with pytest.raises(DomainError):
        service.history(cid)
    assert service.repository.purge_expired() == 1
    assert service.repository.history(other) == []
    with system.engine.connect() as conn:
        assert conn.scalar(text('SELECT count(*) FROM answer_events')) == 0


def test_history_omits_raw_evidence_and_marks_unavailable_references(system):
    service, cid = build(system)
    request = TurnRequest(query='Question')
    response = service.answering.answer(request)
    response.evidence = [
        Evidence(
            id='E1',
            hit=SearchHit(
                chunk_id=uuid4(),
                source_id=uuid4(),
                version_id=uuid4(),
                version_number=1,
                filename='source.txt',
                text='original sensitive passage',
                heading_path=[],
                citations=[
                    Citation(
                        block_id=uuid4(),
                        kind='paragraph',
                        locator={'line': 1},
                        start=0,
                        end=10,
                        role='body',
                        excerpt='original sensitive passage',
                    )
                ],
                rrf_score=0.1,
                rerank_score=1,
            ),
        )
    ]
    service.answering = SimpleNamespace(answer=lambda request: response)
    result = service.turn(cid, uuid4(), request)
    assert result['response']['evidence'][0]['available'] is False
    history = service.history(cid)
    assert 'original sensitive passage' not in json.dumps(history, default=str)
    assert history['turns'][0]['response']['evidence'][0]['hit']['citations'][0]['locator']


def test_conversation_api_requires_idempotency_and_server_history(system):
    service, cid = build(system)
    app = create_app(system.settings)
    app.dependency_overrides[get_conversation_service] = lambda: service
    with TestClient(app) as client:
        assert client.post('/conversations').status_code == 201
        assert client.get('/conversations?limit=1000').status_code == 422
        url = f'/conversations/{cid}/turns'
        assert client.post(url, json={'query': 'Question'}).status_code == 422
        headers = {'Idempotency-Key': str(uuid4())}
        assert (
            client.post(
                url,
                headers=headers,
                json={'query': 'Question', 'conversation_context': 'injected history'},
            ).status_code
            == 422
        )
        assert client.post(url, headers=headers, json={'query': 'Question'}).status_code == 200
        assert client.delete(f'/conversations/{cid}').status_code == 204
        assert client.get(f'/conversations/{cid}').status_code == 404


def test_published_citation_becomes_unavailable_after_source_deletion(system, upload):
    from rag_backend.indexing.embeddings import Profile
    from rag_backend.jobs.runner import JobRunner
    from rag_backend.retrieval.contracts import SearchHit
    from rag_backend.retrieval.repository import RetrievalRepository

    class Embeddings:
        def profile(self):
            return Profile('test', 'history', 'v1')

        def embed(self, texts, profile):
            return [[1.0] + [0.0] * 767 for _ in texts]

    receipt = upload(b'Saturn rings contain ice.')
    embedder = Embeddings()
    runner = JobRunner(system.queue, system.files, embedder=embedder)
    for _ in range(3):
        assert runner.run_once()['status'] == 'succeeded'
    with system.engine.connect() as conn:
        chunk = conn.scalar(text('SELECT chunk_id FROM published_retrieval_records'))
    original = RetrievalRepository(system.engine).hydrate(
        [{'chunk_id': chunk, 'rrf_score': 0.01, 'rerank_score': 5}], embedder.profile().id, None
    )
    service, cid = build(system)
    response = service.answering.answer(TurnRequest(query='Saturn'))
    response.evidence = [Evidence(id='E1', hit=SearchHit.model_validate(original[0]))]
    service.answering = SimpleNamespace(answer=lambda request: response)
    key, request = uuid4(), TurnRequest(query='Saturn')
    assert service.turn(cid, key, request)['response']['evidence'][0]['available']
    system.service.delete(receipt['source_id'])
    replay = service.turn(cid, key, request)
    assert replay['replayed'] and not replay['response']['evidence'][0]['available']


def test_telemetry_is_bounded_content_free_and_separate_from_quality(system):
    from rag_backend.evaluation.telemetry import summarize

    service, cid = build(system)
    service.turn(cid, uuid4(), TurnRequest(query='Private question not for dashboard'))
    report = summarize(system.engine)
    assert report['events'] == 1 and report['outcomes'] == {'abstain': 1}
    assert report['p50_latency_ms'] == 1
    assert 'Private question' not in json.dumps(report)
    assert 'retrieval_recall_at_k' not in report
    service.repository.delete(cid)
    assert summarize(system.engine)['events'] == 0
