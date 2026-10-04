import math
import re

from pydantic import ValidationError

from rag_backend.answering.contracts import Draft
from rag_backend.answering.nli import NLIError
from rag_backend.answering.planner import unsafe_text
from rag_backend.retrieval.reranking import RerankError


def validate(raw, evidence, request, task, nli, relevance):
    try:
        draft = Draft.model_validate_json(raw)
    except (ValidationError, ValueError):
        return 'repair', 'invalid_schema', None
    if draft.status != 'answer':
        return draft.status, 'model_declined', None
    if not draft.claims:
        return 'repair', 'empty_answer', None
    by_id = {item.id: item for item in evidence}
    cited_sources = set()
    for claim in draft.claims:
        if not claim.text.strip():
            return 'repair', 'empty_claim', None
        if any(key not in by_id for key in claim.evidence_ids):
            return 'repair', 'unknown_citation', None
        if unsafe_text(claim.text):
            return 'abstain', 'unsafe_claim', None
        cited = ' '.join(by_id[key].hit.text for key in claim.evidence_ids)
        numbers = r'(?<!\w)[+-]?\d+(?:[.,:/-]\d+)*(?:%|\b)'
        if not set(re.findall(numbers, claim.text)) <= set(re.findall(numbers, cited)):
            return 'abstain', 'unsupported_value', None
        if any(symbol not in cited for symbol in re.findall(r'`([^`]+)`', claim.text)):
            return 'abstain', 'unsupported_symbol', None
        cited_sources.update(by_id[key].hit.source_id for key in claim.evidence_ids)
    if task == 'compare' and cited_sources != set(request.source_ids or []):
        return 'abstain', 'comparison_incomplete', None
    try:
        if task == 'answer':
            scores = relevance.score(request.query, [c.text for c in draft.claims])
            if len(scores) != len(draft.claims) or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in scores
            ):
                return 'abstain', 'query_coverage_uncertain', None
        pairs = [
            {'premise': item.hit.text, 'hypothesis': claim.text}
            for claim in draft.claims
            for item in evidence
        ]
        scores = nli.predict(pairs)
        if not isinstance(scores, list) or len(scores) != len(pairs):
            raise NLIError('invalid_scores')
        for row in scores:
            if (
                not isinstance(row, dict)
                or set(row) != {'entailment', 'neutral', 'contradiction'}
                or any(
                    not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                    for v in row.values()
                )
                or abs(sum(row.values()) - 1) > 0.01
            ):
                raise NLIError('invalid_scores')
        for i, claim in enumerate(draft.claims):
            support = scores[i * len(evidence) : (i + 1) * len(evidence)]
            if any(row['contradiction'] >= 0.9 for row in support):
                return 'abstain', 'conflicting_evidence', None
            if any(
                row['entailment'] < 0.9
                for item, row in zip(evidence, support, strict=True)
                if item.id in claim.evidence_ids
            ):
                return 'abstain', 'unsupported_or_uncertain_claim', None
    except (NLIError, RerankError):
        return 'extracts', 'validator_unavailable_or_limit', None
    return 'answer', 'validated', draft
