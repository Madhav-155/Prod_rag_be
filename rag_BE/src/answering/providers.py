import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol

import httpx

from rag_backend.indexing.embeddings import EmbeddingError, OllamaEmbedder


class GenerationError(Exception):
    pass


@dataclass
class Generated:
    text: str
    profile: str
    usage: dict = field(default_factory=dict)


class Generator(Protocol):
    def generate(self, messages: list[dict], schema: dict) -> Generated: ...


def check_input(messages):
    if len(json.dumps(messages).encode()) > 6000:
        raise GenerationError('generation_context_limit')


class OllamaGenerator:
    def __init__(self, url, model, timeout=180, transport=None):
        self.client = OllamaEmbedder(url, model, timeout, transport)

    def generate(self, messages, schema):
        check_input(messages)
        try:
            profile = self.client.profile()
            if ':cloud' in profile.model:
                raise GenerationError('remote_model_not_allowed')
            info = self.client._request('POST', '/api/show', {'model': profile.model})
            if info.get('remote_host') or info.get('remote_model'):
                raise GenerationError('remote_model_not_allowed')
            if 'completion' not in info.get('capabilities', []):
                raise GenerationError('generation_capability_missing')
            payload = {
                'model': profile.model,
                'messages': messages,
                'format': schema,
                'stream': False,
                'options': {'temperature': 0, 'num_predict': 512, 'num_ctx': 8192},
            }
            if 'thinking' in info.get('capabilities', []):
                payload['think'] = False
            response = self.client._request('POST', '/api/chat', payload)
            if not response.get('done') or response.get('done_reason') == 'length':
                raise GenerationError('generation_incomplete')
            message = response.get('message', {})
            if message.get('tool_calls') or not isinstance(message.get('content'), str):
                raise GenerationError('generation_invalid_response')
            if self.client.profile() != profile:
                raise GenerationError('generation_model_changed')
            return Generated(
                message['content'],
                f'ollama:{profile.model}@{profile.digest}',
                {
                    'input_tokens': response.get('prompt_eval_count'),
                    'output_tokens': response.get('eval_count'),
                    'api_cost_usd': 0,
                },
            )
        except (EmbeddingError, AttributeError, TypeError, ValueError) as error:
            raise GenerationError('local_generation_unavailable') from error


class OpenAIGenerator:
    """Explicit opt-in only. No remote fallback and no automatic retries."""

    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport
        self.calls, self.lock = 0, Lock()

    def generate(self, messages, schema):
        check_input(messages)
        s = self.settings
        with self.lock:
            if (
                not s.remote_generation_enabled
                or not s.remote_data_policy_reviewed
                or s.remote_kill_switch
                or not s.openai_api_key
                or not s.openai_model
                or self.calls >= s.remote_call_limit
            ):
                raise GenerationError('remote_generation_disabled_or_limit')
            self.calls += 1
        try:
            with httpx.Client(
                timeout=90, trust_env=False, follow_redirects=False, transport=self.transport
            ) as client:
                response = client.post(
                    'https://api.openai.com/v1/responses',
                    headers={'Authorization': 'Bearer ' + s.openai_api_key.get_secret_value()},
                    json={
                        'model': s.openai_model,
                        'input': messages,
                        'store': False,
                        'max_output_tokens': 512,
                        'tools': [],
                        'text': {
                            'format': {
                                'type': 'json_schema',
                                'name': 'grounded_answer',
                                'strict': True,
                                'schema': schema,
                            }
                        },
                    },
                )
            if response.status_code != 200:
                raise GenerationError('remote_generation_failed')
            data = response.json()
            if data.get('status') != 'completed':
                raise GenerationError('generation_incomplete')
            parts = [
                part['text']
                for item in data.get('output', [])
                if item.get('type') == 'message'
                for part in item.get('content', [])
                if part.get('type') == 'output_text'
            ]
            if not parts:
                raise GenerationError('generation_invalid_response')
            usage = data.get('usage', {})
            return Generated(
                ''.join(parts),
                'openai:' + str(data.get('model', s.openai_model)),
                {
                    'input_tokens': usage.get('input_tokens'),
                    'output_tokens': usage.get('output_tokens'),
                    'api_cost_usd': None,
                    'remote_calls_this_process': self.calls,
                },
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise GenerationError('remote_generation_failed') from error
