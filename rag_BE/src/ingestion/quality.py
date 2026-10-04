import re

from rag_backend.ingestion.contracts import CanonicalBlock, Options, ParsedDocument


def validate(document: ParsedDocument, blocks: list[CanonicalBlock], options: Options) -> list[str]:
    reasons = []
    text = ''.join(block.text for block in blocks)
    if not text.strip():
        reasons.append('no_readable_content')
    if document.metrics.get('ocr_pages'):
        reasons.append('ocr_required')
    if document.metrics.get('complex_layout_pages'):
        reasons.append('complex_pdf_layout')
    if any(block.kind in {'image', 'table', 'code', 'html'} for block in blocks):
        reasons.append('mixed_content_processing_deferred')
    if re.search(r'\(cid:\d+\)', text):
        reasons.append('unmapped_pdf_glyphs')
    if text and text.count('\ufffd') / len(text) > 0.01:
        reasons.append('low_text_quality')
    if any(ord(char) < 32 and char not in '\t\n\r' for char in text):
        reasons.append('unsafe_control_characters')
    if len(text) > options.max_chars or len(blocks) > options.max_blocks:
        reasons.append('extraction_limit_exceeded')
    for block in blocks:
        location = block.locator
        if document.format == 'pdf':
            if not location.get('page') or len(location.get('bbox', [])) != 4:
                reasons.append('missing_source_location')
                break
        elif not location.get('line_start') or location.get('line_end', 0) < location['line_start']:
            reasons.append('missing_source_location')
            break
    return list(dict.fromkeys(reasons))
