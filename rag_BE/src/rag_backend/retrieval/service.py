import logging
import math
import time
from threading import BoundedSemaphore
from uuid import uuid4

from rag_backend.domain.errors import DomainError
from rag_backend.indexing.embeddings import EmbeddingError, validate_vectors
from rag_backend.retrieval.contracts import SearchRequest, SearchResponse
from rag_backend.retrieval.ranking import deduplicate, diverse, fuse
from rag_backend.retrieval.reranking import RerankError

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, repository, embedder, reranker):
        self.repository, self.embedder, self.reranker = repository, embedder, reranker
        self.capacity = BoundedSemaphore(2)

    def search(self, request: SearchRequest):
        if not self.capacity.acquire(blocking=False):
            raise DomainError('retrieval_busy', 'Retrieval is busy; retry shortly.', 503)
        try:
            return self._search(request)
        except EmbeddingError as error:
            raise DomainError(error.code, 'Local query embedding is unavailable.', 503) from error
        finally:
            self.capacity.release()

    def _search(self, request):
        started, timings = time.monotonic(), {}
        request_id, warnings = uuid4(), []

        def timed(name, call):
            before = time.monotonic()
            try:
                result = call()
            finally:
                timings[name] = round((time.monotonic() - before) * 1000, 2)
            if time.monotonic() - started > 120:
                raise DomainError('retrieval_timeout', 'Retrieval time budget exceeded.', 503)
            return result

        if request.source_ids == []:
            return SearchResponse(
                request_id=request_id,
                status='no_results',
                hits=[],
                warnings=[],
                trace={'reason': 'empty_source_scope'},
            )
        profile = timed('profile', self.embedder.profile)
        vector = timed(
            'query_embedding',
            lambda: validate_vectors([self.embedder.embed_query(request.query, profile)], 1)[0],
        )
        rows = timed(
            'hybrid_search',
            lambda: self.repository.candidates(
                request.query, vector, profile.id, request.source_ids
            ),
        )
        candidates = timed('fusion_deduplication', lambda: deduplicate(fuse(rows)))[:20]
        if candidates:
            try:
                scores = timed(
                    'reranking',
                    lambda: self.reranker.score(
                        request.query, [item['text'] for item in candidates]
                    ),
                )
                if len(scores) != len(candidates) or any(
                    type(score) not in (int, float) or not math.isfinite(score) for score in scores
                ):
                    raise RerankError('reranker_invalid_scores')
                for item, score in zip(candidates, scores, strict=True):
                    item['rerank_score'] = score
                candidates.sort(
                    key=lambda x: (-x['rerank_score'], -x['rrf_score'], str(x['chunk_id']))
                )
            except RerankError as error:
                # Explicit quality degradation, never fabricated reranker scores.
                allowed = {
                    'reranker_timeout',
                    'reranker_not_provisioned',
                    'reranker_integrity_failed',
                    'reranker_input_limit',
                    'reranker_query_limit',
                    'reranker_window_limit',
                    'reranker_invalid_scores',
                }
                warnings.append(str(error) if str(error) in allowed else 'reranker_unavailable')
                warnings.append('using_rrf_order')
        evidence = timed(
            'evidence_hydration',
            lambda: self.repository.hydrate(candidates, profile.id, request.source_ids),
        )
        hits = diverse(evidence, request.limit)
        trace = {
            'embedding_profile': profile.id,
            'reranker_profile': self.reranker.profile,
            'candidate_count': len(candidates),
            'returned_count': len(hits),
            'stage_ms': timings,
            'total_ms': round((time.monotonic() - started) * 1000, 2),
            'ranking': ('rrf' if warnings else 'cross_encoder') if candidates else 'none',
            'rrf_constant': 60,
            'api_cost_usd': 0,
            'cache_hit': False,
        }
        # No query, passage text, filenames or model prompts in telemetry.
        logger.info(
            'retrieval_complete',
            extra={'request_id': str(request_id), 'retrieval_trace': trace, 'warnings': warnings},
        )
        return SearchResponse(
            request_id=request_id,
            status='ok' if hits else 'no_results',
            hits=hits,
            warnings=warnings,
            trace=trace,
        )
