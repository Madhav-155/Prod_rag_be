"""Heading Detector, Table Preserver and Boundary Detector for canonical evidence."""

import re
from uuid import UUID, uuid5

from rag_backend.ingestion.contracts import (
    CanonicalBlock,
    Chunk,
    EvidenceSpan,
    Options,
    PreprocessFailure,
)


def boundary(text: str, start: int, budget: int) -> int:
    end = min(len(text), start + budget)
    if end == len(text):
        return end
    segment = text[start:end]
    stops = list(re.finditer(r'[.!?](?:\s+|$)|\n', segment))
    if stops:
        return start + stops[-1].end()
    spaces = list(re.finditer(r'\s+', segment))
    return start + spaces[-1].end() if spaces else end


def make_chunks(blocks: list[CanonicalBlock], version_id: UUID, profile: str, options: Options):
    chunks = []
    spans = []
    parts = []
    heading_path = []
    prefix = ''
    previous = None
    heading_blocks = {}

    def emit(kind='paragraph'):
        nonlocal spans, parts, previous
        if not parts:
            return
        text = prefix + '\n\n'.join(parts)
        ordinal = len(chunks)
        referenced = {span.block_id for span in spans}
        heading_refs = []
        for length in range(1, len(heading_path) + 1):
            heading = heading_blocks.get(tuple(heading_path[:length]))
            if heading and heading.id not in referenced:
                heading_refs.append(
                    EvidenceSpan(
                        block_id=heading.id, start=0, end=len(heading.text), role='context'
                    )
                )
        full_spans = heading_refs + list(spans)
        identity = f'{profile}:chunk:{ordinal}:' + ':'.join(
            f'{s.block_id}:{s.start}:{s.end}:{s.role}' for s in full_spans
        )
        chunks.append(
            Chunk(
                id=uuid5(version_id, identity),
                ordinal=ordinal,
                kind=kind,
                text=text,
                heading_path=list(heading_path),
                spans=full_spans,
            )
        )
        previous = (list(spans), list(parts)) if kind == 'paragraph' else None
        spans, parts = [], []

    def add_overlap():
        if not previous or not options.overlap_chars:
            return
        old_spans, old_parts = previous
        last, text = old_spans[-1], old_parts[-1]
        if last.role != 'content':
            return
        sentences = list(re.finditer(r'[^.!?]+[.!?]?\s*', text))
        candidate = sentences[-1].group() if sentences else text
        if 0 < len(candidate) <= options.overlap_chars:
            start = last.end - len(candidate)
            spans.append(
                EvidenceSpan(block_id=last.block_id, start=start, end=last.end, role='context')
            )
            parts.append(candidate)

    for block in blocks:
        if block.heading_path != heading_path:
            emit()
            heading_path = block.heading_path
            prefix = ('Section: ' + ' > '.join(heading_path) + '\n\n') if heading_path else ''
            previous = None
        if block.kind == 'heading':
            heading_blocks[tuple(block.heading_path)] = block
        if len(prefix) >= options.chunk_chars // 2:
            raise PreprocessFailure('heading_context_too_large')
        if block.kind in {'table', 'code', 'image', 'html'}:
            emit()
            previous = None
            if len(prefix) + len(block.text) > options.chunk_chars:
                raise PreprocessFailure('atomic_block_too_large')
            if block.text:
                spans = [EvidenceSpan(block_id=block.id, start=0, end=len(block.text))]
                parts = [block.text]
                emit(block.kind)
            continue
        cursor = 0
        while cursor < len(block.text):
            # Context is only copied between adjacent prose chunks in the same section.
            if not parts and block.kind != 'heading':
                add_overlap()
            used = len(prefix) + sum(len(p) for p in parts) + 2 * len(parts)
            budget = options.chunk_chars - used
            if budget <= 0:
                emit()
                continue
            end = boundary(block.text, cursor, budget)
            spans.append(EvidenceSpan(block_id=block.id, start=cursor, end=end))
            parts.append(block.text[cursor:end])
            cursor = end
            if cursor < len(block.text):
                emit()
    emit()
    return chunks
