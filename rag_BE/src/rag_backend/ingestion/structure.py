"""Identify hierarchy separately from extraction and normalization."""

import statistics

from rag_backend.ingestion.contracts import ParsedDocument


def analyze(document: ParsedDocument) -> ParsedDocument:
    sizes = [
        block.attributes['font_size']
        for block in document.blocks
        if block.kind == 'paragraph' and block.attributes.get('font_size')
    ]
    body_size = statistics.median(sizes) if sizes else 0
    for block in document.blocks:
        if (
            document.format == 'pdf'
            and block.kind == 'paragraph'
            and 0 < len(block.raw_text) <= 160
            and body_size
            and block.attributes.get('font_size', 0) >= body_size * 1.2
        ):
            block.kind = 'heading'
            block.attributes.update(
                {
                    'level': 1 if block.attributes['font_size'] >= body_size * 1.5 else 2,
                    'title': block.raw_text.strip(),
                    'heading_method': 'font_size_heuristic',
                }
            )
    if document.format == 'pdf':
        merged = []
        for block in document.blocks:
            previous = merged[-1] if merged else None
            size = block.attributes.get('font_size', 0)
            if (
                previous
                and previous.kind == block.kind == 'paragraph'
                and size
                and previous.locator['page'] == block.locator['page']
                and abs(previous.attributes.get('font_size', 0) - size) < 0.5
                and 0 <= block.locator['bbox'][1] - previous.locator['bbox'][3] <= size * 0.5
                and abs(block.locator['bbox'][0] - previous.locator['bbox'][0]) <= size
            ):
                previous.attributes.setdefault(
                    'line_bboxes', [list(previous.locator['bbox'])]
                ).append(list(block.locator['bbox']))
                previous.raw_text += '\n' + block.raw_text
                a, b = previous.locator['bbox'], block.locator['bbox']
                previous.locator['bbox'] = [
                    min(a[0], b[0]),
                    min(a[1], b[1]),
                    max(a[2], b[2]),
                    max(a[3], b[3]),
                ]
                previous.attributes['paragraph_method'] = 'line_spacing_heuristic'
            else:
                merged.append(block)
        document.blocks = merged
    return document
