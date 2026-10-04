import hashlib
import importlib.metadata
import json
from pathlib import Path
from uuid import UUID

from rag_backend.ingestion.chunking import make_chunks
from rag_backend.ingestion.contracts import Options, PreprocessFailure, PreprocessingResult
from rag_backend.ingestion.detection import detect
from rag_backend.ingestion.normalization import normalize
from rag_backend.ingestion.parsers import pdf, text
from rag_backend.ingestion.quality import validate
from rag_backend.ingestion.structure import analyze


def profile_for(options: Options) -> str:
    identity = {
        'pipeline': 'restructure-v1',
        'pdfplumber': importlib.metadata.version('pdfplumber'),
        'markdown-it-py': importlib.metadata.version('markdown-it-py'),
        'options': options.model_dump(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return f'restructure-v1-{digest}'


def run(path: Path, filename: str, version_id: UUID, options: Options) -> PreprocessingResult:
    profile = profile_for(options)
    result = PreprocessingResult(version_id=version_id, profile=profile, status='quarantined')
    try:
        format = detect(path, filename)
        result.completed_stages.append('file_type_detector')
        document = (
            pdf.parse(path, options) if format == 'pdf' else text.parse(path, format, options)
        )
        result.completed_stages.append('document_parser')
        document = analyze(document)
        result.completed_stages.append('structure_analyzer')
        result.blocks = normalize(document, version_id, profile)
        result.completed_stages.append('canonicalization_normalization')
        result.metrics = {
            **document.metrics,
            'format': format,
            'blocks': len(result.blocks),
            'quality_method': 'deterministic_rules',
            'options': options.model_dump(),
        }
        result.reasons = validate(document, result.blocks, options)
        result.completed_stages.append('extraction_quality_validation')
        if result.reasons:
            return result
        result.chunks = make_chunks(result.blocks, version_id, profile, options)
        result.completed_stages.append('structure_aware_chunking')
        if not result.chunks:
            raise PreprocessFailure('no_candidate_chunks')
        result.status = 'approved'
        return result
    except PreprocessFailure as error:
        result.reasons = [error.code]
        result.chunks = []
        return result
