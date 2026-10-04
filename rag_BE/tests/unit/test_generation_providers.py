import json

import httpx
import pytest

from rag_backend.answering.context import schema
from rag_backend.answering.providers import GenerationError, OllamaGenerator, OpenAIGenerator
from rag_backend.config import Settings

MESSAGES = [{'role': 'user', 'content': 'Synthetic public test'}]


def test_local_generation_contract_and_no_thinking_or_tool_execution():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [{'name': 'test', 'digest': 'd1'}]})
        if request.url.path == '/api/show':
            return httpx.Response(200, json={'capabilities': ['completion', 'thinking']})
        body = json.loads(request.content)
        assert body['think'] is False and body['stream'] is False
        assert body['format']['additionalProperties'] is False
        assert body['options']['num_predict'] == 512 and 'tools' not in body
        return httpx.Response(
            200,
            json={
                'done': True,
                'done_reason': 'stop',
                'message': {'content': '{"status":"abstain","claims":[]}', 'thinking': 'private'},
                'prompt_eval_count': 10,
                'eval_count': 5,
            },
        )

    generator = OllamaGenerator(
        'http://127.0.0.1:11434', 'test', transport=httpx.MockTransport(handler)
    )
    result = generator.generate(MESSAGES, schema())
    assert 'private' not in result.text and result.usage['api_cost_usd'] == 0
    assert paths == ['/api/tags', '/api/show', '/api/chat', '/api/tags']


@pytest.mark.parametrize(
    'info',
    [
        {'remote_host': 'remote.example', 'capabilities': ['completion']},
        {'capabilities': ['embedding']},
    ],
)
def test_local_capability_gate_prevents_generation(info):
    def handler(request):
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [{'name': 'test', 'digest': 'd'}]})
        assert request.url.path == '/api/show'
        return httpx.Response(200, json=info)

    generator = OllamaGenerator(
        'http://127.0.0.1:11434', 'test', transport=httpx.MockTransport(handler)
    )
    with pytest.raises(GenerationError):
        generator.generate(MESSAGES, schema())


def remote_settings(**overrides):
    values = dict(
        _env_file=None,
        database_password='synthetic-test-password',
        openai_model='configured-test-model',
        openai_api_key='test-secret-not-real',
        remote_generation_enabled=True,
        remote_data_policy_reviewed=True,
        remote_kill_switch=False,
        remote_call_limit=1,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize(
    'override',
    [
        {'remote_generation_enabled': False},
        {'remote_data_policy_reviewed': False},
        {'remote_kill_switch': True},
        {'remote_call_limit': 0},
        {'openai_api_key': None},
        {'openai_model': ''},
    ],
)
def test_every_remote_gate_prevents_network_io(override):
    def forbidden(request):
        pytest.fail('Remote request must not happen')

    generator = OpenAIGenerator(remote_settings(**override), httpx.MockTransport(forbidden))
    with pytest.raises(GenerationError, match='disabled_or_limit'):
        generator.generate(MESSAGES, schema())


def test_remote_mock_contract_and_call_cap():
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == 'https://api.openai.com/v1/responses'
        body = json.loads(request.content)
        assert body['store'] is False and body['tools'] == []
        assert body['max_output_tokens'] == 512
        assert body['text']['format']['strict'] is True
        assert body['text']['format']['schema'] == schema()
        return httpx.Response(
            200,
            json={
                'status': 'completed',
                'model': 'configured-test-model',
                'output': [
                    {'type': 'reasoning', 'summary': ['do not expose']},
                    {
                        'type': 'message',
                        'content': [
                            {'type': 'output_text', 'text': '{"status":"abstain","claims":[]}'}
                        ],
                    },
                ],
                'usage': {'input_tokens': 10, 'output_tokens': 6},
            },
        )

    generator = OpenAIGenerator(remote_settings(), httpx.MockTransport(handler))
    result = generator.generate(MESSAGES, schema())
    assert 'do not expose' not in result.text
    assert result.usage['api_cost_usd'] is None
    with pytest.raises(GenerationError, match='disabled_or_limit'):
        generator.generate(MESSAGES, schema())
    assert len(requests) == 1
    assert 'test-secret-not-real' not in repr(generator.settings)


def test_failed_remote_request_consumes_cap_and_hides_body():
    generator = OpenAIGenerator(
        remote_settings(),
        httpx.MockTransport(lambda request: httpx.Response(500, text='sensitive server details')),
    )
    with pytest.raises(GenerationError, match='remote_generation_failed'):
        generator.generate(MESSAGES, schema())
    assert generator.calls == 1
    with pytest.raises(GenerationError, match='disabled_or_limit'):
        generator.generate(MESSAGES, schema())


def test_context_limit_precedes_network():
    generator = OpenAIGenerator(remote_settings())
    with pytest.raises(GenerationError, match='context_limit'):
        generator.generate([{'role': 'user', 'content': 'x' * 7000}], schema())
