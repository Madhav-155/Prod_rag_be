import subprocess
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rag_backend.domain.errors import DomainError
from rag_backend.retrieval.contracts import SearchRequest
from rag_backend.retrieval.ranking import deduplicate, diverse, fuse
from rag_backend.retrieval.reranking import LocalReranker, RerankError
from rag_backend.retrieval.service import RetrievalService


def candidate(content='Evidence', **overrides):
    return {
        'chunk_id': uuid4(),
        'source_id': uuid4(),
        'version_id': uuid4(),
        'text': content,
        'rank': 1,
        'channel': 'lexical',
        **overrides,
    }


def test_query_normalization_preserves_meaning():
    assert SearchRequest(query='  cafe\u0301\n  42 ').query == 'café 42'
    assert SearchRequest(query='NOT approved?').query == 'NOT approved?'


@pytest.mark.parametrize(
    'kwargs',
    [
        {'query': ' '},
        {'query': '???'},
        {'query': 'x\0y'},
        {'query': 'x' * 1001},
        {'query': 'hi', 'limit': 11},
        {'query': 'hi', 'limit': 0},
        {'query': 'hi', 'source_ids': ['invalid']},
        {'query': 'hi', 'raw_sql': 'anything'},
    ],
)
def test_request_bounds(kwargs):
    with pytest.raises(ValidationError):
        SearchRequest(**kwargs)


def test_rrf_uses_ranks_not_incompatible_raw_scores():
    first, second = candidate('First'), candidate('Second', rank=2)
    combined = fuse([first, second, {**second, 'channel': 'semantic', 'rank': 1}])
    assert combined[0]['chunk_id'] == second['chunk_id']
    assert combined[0]['rrf_score'] == pytest.approx(1 / 62 + 1 / 61)


def test_deduplication_keeps_different_numbers_and_distinct_evidence():
    version = uuid4()
    items = [
        candidate('Value 42 is approved.', version_id=version),
        candidate('Value 43 is approved.', version_id=version),
        candidate('Value 42 is not approved.', version_id=version),
        candidate('Value 42  is approved.'),
    ]
    assert len(deduplicate(items)) == 3


def test_diversity_prefers_other_sources_but_fills_available_slots():
    source = uuid4()
    items = [candidate(str(i), source_id=source) for i in range(4)] + [candidate('other')]
    assert diverse(items, 3)[2]['text'] == 'other'
    assert len(diverse(items[:4], 4)) == 4


def test_explicit_empty_scope_never_calls_a_model():
    service = RetrievalService(None, None, None)
    response = service.search(SearchRequest(query='hello', source_ids=[]))
    assert response.status == 'no_results' and not response.hits


def test_busy_service_rejects_without_unbounded_queue():
    service = RetrievalService(None, None, None)
    service.capacity.acquire()
    service.capacity.acquire()
    with pytest.raises(DomainError) as error:
        service.search(SearchRequest(query='hello'))
    assert error.value.code == 'retrieval_busy'


def test_missing_local_reranker_is_explicit_and_does_not_download(tmp_path):
    with pytest.raises(RerankError):
        LocalReranker(tmp_path).score('question', ['evidence'])
    assert not list(tmp_path.iterdir())


def test_reranker_process_timeout_is_reported(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('test', 30)

    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(RerankError, match='reranker_timeout'):
        LocalReranker(tmp_path).score('question', ['evidence'])


def test_tampered_model_is_rejected_before_inference(tmp_path):
    from rag_backend.retrieval.rerank_worker import verify_files

    (tmp_path / 'model.onnx').write_bytes(b'not the pinned model')
    (tmp_path / 'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError, match='reranker_integrity_failed'):
        verify_files(tmp_path)


def test_reranker_windows_cover_all_tokens_including_the_tail():
    from rag_backend.retrieval.rerank_worker import build_windows

    document = list(range(1000, 2200))
    windows, owners = build_windows([7, 8, 9], [document])
    passages = [window['input_ids'][5:-1] for window in windows]
    assert set(token for part in passages for token in part) == set(document)
    assert passages[-1][-1] == document[-1]
    assert all(len(window['input_ids']) <= 512 for window in windows)
    assert len(owners) == len(windows) and set(owners) == {0}
    assert passages[0][-32:] == passages[1][:32]
    with pytest.raises(ValueError, match='reranker_window_limit'):
        build_windows([7], [list(range(5000))])
    with pytest.raises(ValueError, match='reranker_query_limit'):
        build_windows([7] * 129, [document])
