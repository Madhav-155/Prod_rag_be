import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import urlsplit

import httpx

DIMENSIONS = 768


class EmbeddingError(Exception):
    def __init__(self, code: str, retryable: bool = False):
        super().__init__(code)
        self.code, self.retryable = code, retryable


@dataclass(frozen=True)
class Profile:
    provider: str
    model: str
    digest: str
    dimensions: int = DIMENSIONS
    representation: str = 'original-text-v1/simple-fts'

    @property
    def id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class Embedder(Protocol):
    def profile(self) -> Profile: ...
    def embed(self, texts: list[str], profile: Profile) -> list[list[float]]: ...
    def embed_query(self, query: str, profile: Profile) -> list[float]: ...


def validate_vectors(vectors, count: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise EmbeddingError('embedding_count_mismatch')
    normalized = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != DIMENSIONS:
            raise EmbeddingError('embedding_dimension_mismatch')
        if any(type(x) not in (float, int) or not math.isfinite(x) for x in vector):
            raise EmbeddingError('embedding_invalid_values')
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm <= 0:
            raise EmbeddingError('embedding_zero_or_invalid_norm')
        normalized.append([x / norm for x in vector])
    return normalized


class OllamaEmbedder:
    """Loopback only, no proxies, redirects, auto-downloads or cloud fallback."""

    def __init__(
        self,
        url='http://127.0.0.1:11434',
        model='embeddinggemma:latest',
        timeout=60,
        transport=None,
    ):
        parsed = urlsplit(url)
        if (
            parsed.scheme != 'http'
            or parsed.hostname not in {'127.0.0.1', '::1'}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ('', '/')
        ):
            raise ValueError('Embedding endpoint must be an HTTP loopback origin')
        self.url, self.model, self.timeout, self.transport = (
            url.rstrip('/'),
            model,
            timeout,
            transport,
        )

    def _request(self, method, path, payload=None):
        try:
            with httpx.Client(
                base_url=self.url,
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, json=payload)
            if response.status_code != 200:
                raise EmbeddingError(
                    'embedding_http_error',
                    response.status_code == 429 or response.status_code >= 500,
                )
            value = response.json()
            if not isinstance(value, dict):
                raise EmbeddingError('embedding_invalid_response')
            return value
        except httpx.HTTPError as error:
            raise EmbeddingError('embedding_unavailable', True) from error
        except ValueError as error:
            raise EmbeddingError('embedding_invalid_response') from error

    def profile(self):
        models = self._request('GET', '/api/tags').get('models')
        if not isinstance(models, list):
            raise EmbeddingError('embedding_invalid_response')
        for item in models:
            if isinstance(item, dict) and item.get('name') == self.model:
                digest = item.get('digest')
                if not isinstance(digest, str) or not digest.strip():
                    raise EmbeddingError('embedding_digest_missing')
                representation = (
                    'embeddinggemma-retrieval-v2/title-none/search-result/simple-fts'
                    if self.model.split(':')[0] == 'embeddinggemma'
                    else 'original-text-v1/simple-fts'
                )
                return Profile('ollama', self.model, digest, representation=representation)
        raise EmbeddingError('embedding_model_missing')

    def embed(self, texts, profile):
        if self.model.split(':')[0] == 'embeddinggemma':
            texts = ['title: none | text: ' + text for text in texts]
        return self._embed(texts, profile)

    def embed_query(self, query, profile):
        if self.model.split(':')[0] == 'embeddinggemma':
            query = 'task: search result | query: ' + query
        return self._embed([query], profile)[0]

    def _embed(self, texts, profile):
        if self.profile() != profile:
            raise EmbeddingError('embedding_model_changed')
        data = self._request(
            'POST', '/api/embed', {'model': self.model, 'input': texts, 'truncate': False}
        )
        vectors = validate_vectors(data.get('embeddings'), len(texts))
        if self.profile() != profile:
            raise EmbeddingError('embedding_model_changed')
        return vectors
