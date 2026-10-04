import json

import httpx
import pytest

from rag_backend.indexing.embeddings import (
    EmbeddingError,
    OllamaEmbedder,
    Profile,
    validate_vectors,
)


@pytest.mark.parametrize(
    'value',
    [[], [[1.0]], [[0.0] * 768], [[float('nan')] * 768], [[float('inf')] * 768], [[True] * 768]],
)
def test_rejects_invalid_vectors(value):
    with pytest.raises(EmbeddingError):
        validate_vectors(value, 1)


def test_normalizes_and_profiles_are_versioned():
    assert sum(v * v for v in validate_vectors([[2.0] * 768], 1)[0]) == pytest.approx(1)
    assert Profile('ollama', 'model', 'old').id != Profile('ollama', 'model', 'new').id


@pytest.mark.parametrize(
    'url',
    [
        'https://example.com',
        'http://localhost:11434',
        'http://127.0.0.1:11434/api',
        'http://user:pass@127.0.0.1',
    ],
)
def test_never_accepts_remote_or_credentialed_endpoints(url):
    with pytest.raises(ValueError):
        OllamaEmbedder(url=url)


def test_local_adapter_sends_nontruncating_batch_and_checks_digest():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [{'name': 'test', 'digest': 'digest1'}]})
        assert json.loads(request.content) == {
            'model': 'test',
            'input': ['hello'],
            'truncate': False,
        }
        return httpx.Response(200, json={'embeddings': [[1.0] * 768]})

    adapter = OllamaEmbedder(model='test', transport=httpx.MockTransport(handler))
    assert len(adapter.embed(['hello'], adapter.profile())) == 1
    assert calls == ['/api/tags', '/api/tags', '/api/embed', '/api/tags']


@pytest.mark.parametrize(
    'status,retryable', [(400, False), (404, False), (302, False), (429, True), (500, True)]
)
def test_error_bodies_are_not_exposed(status, retryable):
    adapter = OllamaEmbedder(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(status, text='sensitive provider error')
        )
    )
    with pytest.raises(EmbeddingError) as error:
        adapter.profile()
    assert error.value.retryable is retryable
    assert 'sensitive' not in str(error.value)


def test_model_missing_never_auto_downloads():
    adapter = OllamaEmbedder(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={'models': []}))
    )
    with pytest.raises(EmbeddingError, match='embedding_model_missing'):
        adapter.profile()


def test_model_changed_is_rejected():
    adapter = OllamaEmbedder(
        model='test',
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={'models': [{'name': 'test', 'digest': 'new'}]})
        ),
    )
    with pytest.raises(EmbeddingError, match='embedding_model_changed'):
        adapter.embed(['hello'], Profile('ollama', 'test', 'old'))


def test_embeddinggemma_document_query_format_and_profile():
    inputs = []

    def handler(request):
        if request.url.path == '/api/tags':
            return httpx.Response(
                200, json={'models': [{'name': 'embeddinggemma:latest', 'digest': 'test-digest'}]}
            )
        body = json.loads(request.content)
        assert body['truncate'] is False
        inputs.extend(body['input'])
        return httpx.Response(200, json={'embeddings': [[1.0] * 768]})

    adapter = OllamaEmbedder(transport=httpx.MockTransport(handler))
    profile = adapter.profile()
    assert 'embeddinggemma-retrieval-v2' in profile.representation
    adapter.embed(['Original text'], profile)
    assert len(adapter.embed_query('A question?', profile)) == 768
    assert inputs == [
        'title: none | text: Original text',
        'task: search result | query: A question?',
    ]
    assert profile.id != Profile('ollama', 'embeddinggemma:latest', 'test-digest').id
