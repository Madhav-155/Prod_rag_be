import logging
import time
from threading import BoundedSemaphore
from uuid import uuid4

from rag_backend.answering.context import assemble, messages, schema
from rag_backend.answering.contracts import AnswerResponse
from rag_backend.answering.planner import TOOLS, make_plan
from rag_backend.answering.providers import GenerationError
from rag_backend.answering.validation import validate
from rag_backend.domain.errors import DomainError
from rag_backend.retrieval.contracts import SearchHit, SearchRequest

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self):
        self.calls = 0
        self.started = time.monotonic()
        self.events = []

    def run(self, name, action):
        if name not in TOOLS or self.calls >= 6 or time.monotonic() - self.started > 300:
            raise DomainError('answer_budget_exceeded', 'Answering budget exceeded.', 503)
        self.calls += 1
        started = time.monotonic()
        try:
            return action()
        finally:
            self.events.append(
                {'tool': name, 'latency_ms': round((time.monotonic() - started) * 1000, 2)}
            )


class AnswerService:
    def __init__(self, retrieval, generator, nli):
        self.retrieval, self.generator, self.nli = retrieval, generator, nli
        self.capacity = BoundedSemaphore(1)

    def answer(self, request):
        if not self.capacity.acquire(blocking=False):
            raise DomainError('answering_busy', 'Answering is busy; retry shortly.', 503)
        try:
            return self._answer(request)
        finally:
            self.capacity.release()

    def _answer(self, request):
        request_started = time.monotonic()
        request_id, plan, tools = uuid4(), make_plan(request), ToolExecutor()
        planning_ms = round((time.monotonic() - request_started) * 1000, 2)
        tools.started = request_started
        evidence, generations, checks = [], [], []
        profile = None
        retrieval_trace = {}

        def rehydrate():
            return [
                SearchHit.model_validate(item)
                for item in self.retrieval.repository.hydrate(
                    [item.hit.model_dump() for item in evidence], profile, request.source_ids
                )
            ]

        def current():
            fresh = tools.run('original_evidence_hydration', rehydrate)
            return [item.model_dump() for item in fresh] == [e.hit.model_dump() for e in evidence]

        def finish(status, reason, draft=None):
            nonlocal evidence
            if evidence and not current():
                status, reason, evidence, draft = 'abstain', 'source_changed', [], None
            if status == 'answer' and draft:
                text = '\n'.join(
                    c.text + ' [' + ', '.join(c.evidence_ids) + ']' for c in draft.claims
                )
                if plan.task in {'summarize', 'generate_section'}:
                    text = 'Based on the retrieved excerpts only:\n' + text
                used = {key for claim in draft.claims for key in claim.evidence_ids}
                shown = [item for item in evidence if item.id in used]
            elif status in {'extracts', 'search'}:
                text = 'Source extracts (not a generated answer):\n' + '\n\n'.join(
                    f'[{item.id}] {item.hit.text}' for item in evidence
                )
                shown = evidence
            elif status == 'clarify':
                text = (
                    'Please make the question self-contained and select the source(s). '
                    'Comparisons require exactly two sources; summaries and draft sections '
                    'need an explicit source selection.'
                )
                shown = []
            else:
                text, shown = 'I cannot provide a supported answer from the available evidence.', []
            trace = {
                'task': plan.task,
                'planning_ms': planning_ms,
                'outcome': status,
                'reason': reason,
                'tool_calls': tools.calls,
                'tools': tools.events,
                'generation_calls': len(generations),
                'generations': generations,
                'validation': checks,
                'retrieval': retrieval_trace,
                'nli_profile': self.nli.profile,
                'total_ms': round((time.monotonic() - tools.started) * 1000, 2),
            }
            logger.info(
                'answer_complete', extra={'request_id': str(request_id), 'answer_trace': trace}
            )
            return AnswerResponse(
                request_id=request_id,
                status=status,
                text=text,
                claims=draft.claims if draft and status == 'answer' else [],
                evidence=shown,
                plan=plan,
                trace=trace,
            )

        if plan.task in {'clarify', 'abstain'}:
            return finish(plan.task, 'planner_boundary')
        results = tools.run(
            'hybrid_retrieval',
            lambda: self.retrieval.search(
                SearchRequest(
                    query=request.query, source_ids=request.source_ids, limit=request.limit
                )
            ),
        )
        profile = results.trace.get('embedding_profile')
        retrieval_trace = results.trace
        hits = results.hits
        if plan.task == 'answer':
            hits = [hit for hit in hits if hit.rerank_score is not None and hit.rerank_score >= 0]
        evidence = assemble(request, plan.task, hits)
        if not evidence:
            return finish('abstain', 'insufficient_safe_evidence')
        if plan.task == 'compare' and {e.hit.source_id for e in evidence} != set(
            request.source_ids
        ):
            return finish('abstain', 'comparison_evidence_missing')
        if not current():
            evidence = []
            return finish('abstain', 'source_changed')
        if plan.task == 'search':
            return finish('search', 'evidence_only')
        for attempt in range(2):
            if attempt and not current():
                evidence = []
                return finish('abstain', 'source_changed')
            if time.monotonic() - tools.started > 240:
                return finish('abstain', 'answer_budget_exceeded')
            started = time.monotonic()
            try:
                generated = self.generator.generate(
                    messages(request, plan.task, evidence, repair=bool(attempt)), schema()
                )
                generations.append(
                    {
                        'profile': generated.profile,
                        'usage': generated.usage,
                        'latency_ms': round((time.monotonic() - started) * 1000, 2),
                    }
                )
            except GenerationError as error:
                generations.append(
                    {
                        'error': str(error),
                        'latency_ms': round((time.monotonic() - started) * 1000, 2),
                    }
                )
                return finish('extracts', 'generation_unavailable')
            outcome, reason, draft = tools.run(
                'citation_validation',
                lambda generated=generated, evidence=evidence: validate(
                    generated.text, evidence, request, plan.task, self.nli, self.retrieval.reranker
                ),
            )
            checks.append({'outcome': outcome, 'reason': reason})
            if outcome != 'repair':
                return finish(outcome, reason, draft)
        return finish('abstain', 'repair_exhausted')
