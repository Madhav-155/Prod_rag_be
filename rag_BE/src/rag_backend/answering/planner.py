import re

from rag_backend.answering.contracts import Plan

TOOLS = ['hybrid_retrieval', 'original_evidence_hydration', 'citation_validation']
SECRET = re.compile(
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
    r'\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16})\b|'
    r'\b(?:password|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S+',
    re.I,
)
INJECTION = re.compile(
    r'ignore (?:all |the |any )?(?:previous|prior|system) instructions|'
    r'reveal (?:the |your )?system prompt|<\|(?:im_start|system)\|>|'
    r'you are now (?:the |an? )?(?:system|assistant)',
    re.I,
)


def unsafe_text(value):
    return bool(SECRET.search(value) or INJECTION.search(value))


def make_plan(request):
    query = request.query.lower()
    task = request.task
    if task == 'auto':
        task = (
            'summarize'
            if re.search(r'\bsummari[sz]e\b|\bsummary\b', query)
            else 'compare'
            if re.search(r'\bcompare\b|\bversus\b', query)
            else 'generate_section'
            if 'draft a section' in query
            else 'answer'
        )
    if unsafe_text(request.query) or re.search(
        r'\b(?:diagnose me|prescribe|dosage|legal advice|investment advice|'
        r'reveal.*password|show.*api key)\b',
        query,
    ):
        task = 'abstain'
    elif query.strip(' ?.').lower() in {
        'what about it',
        'explain that',
        'continue',
        'tell me more',
    }:
        task = 'clarify'
    elif task in {'summarize', 'generate_section'} and not request.source_ids:
        task = 'clarify'
    elif task == 'compare' and (not request.source_ids or len(set(request.source_ids)) != 2):
        task = 'clarify'
    return Plan(
        task=task,
        information_need=request.query,
        allowed_tools=[] if task in {'clarify', 'abstain'} else TOOLS,
        sequence=[]
        if task in {'clarify', 'abstain'}
        else ['retrieve', 'hydrate', 'recheck_citations', 'return_extracts']
        if task == 'search'
        else [
            'retrieve',
            'hydrate',
            'assemble_context',
            'generate',
            'validate',
            'recheck_citations',
        ],
        evidence_requirements=[
            'published_originals_only',
            'claim_level_citations',
            'no_unsupported_values',
            'abstain_on_conflict',
        ],
        stop_criteria=[
            'missing_evidence',
            'unsafe_content',
            'budget_exhausted',
            'unsupported_claim',
            'source_changed',
        ],
    )
