from pathlib import Path
from uuid import uuid4

import pytest

from rag_backend.ingestion.chunking import make_chunks
from rag_backend.ingestion.contracts import Options, ParsedDocument, PreprocessFailure, RawBlock
from rag_backend.ingestion.execution import execute
from rag_backend.ingestion.normalization import normalize
from rag_backend.ingestion.pipeline import run


def process(tmp_path, data, name='notes.txt', options=None, version=None):
    path = tmp_path / 'original.bin'
    path.write_bytes(data)
    return run(path, name, version or uuid4(), options or Options())


def test_complete_restructuring_sequence_and_lineage(tmp_path):
    version = uuid4()
    data = b'# Overview\r\n\r\nFirst paragraph is 42.\r\n\r\n## Detail\r\n\r\nDate: 2026-10-01.\r\n'
    result = process(tmp_path, data, 'notes.md', version=version)
    repeat = process(tmp_path, data, 'notes.md', version=version)
    assert result.status == 'approved'
    assert result.completed_stages == [
        'file_type_detector',
        'document_parser',
        'structure_analyzer',
        'canonicalization_normalization',
        'extraction_quality_validation',
        'structure_aware_chunking',
    ]
    assert result.model_dump() == repeat.model_dump()
    assert result.blocks[-1].heading_path == ['Overview', 'Detail']
    assert result.blocks[-1].locator == {'line_start': 7, 'line_end': 7}
    assert '\r\n' in result.blocks[-1].raw_text
    assert '\r' not in result.blocks[-1].text
    assert '2026-10-01' in result.chunks[-1].text


@pytest.mark.parametrize(
    ('data', 'name', 'reason'),
    [
        (b'not a pdf', 'bad.pdf', 'file_type_mismatch'),
        (b'%PDF-1.4 invalid', 'bad.txt', 'file_type_mismatch'),
        (b'\xff\xfeinvalid', 'bad.txt', 'unsupported_text_encoding'),
        (b'text\x00binary', 'bad.txt', 'binary_content_in_text'),
        (b' \n ', 'empty.txt', 'no_readable_content'),
        (b'%PDF-1.4 invalid', 'bad.pdf', 'malformed_pdf'),
    ],
)
def test_unusable_sources_quarantine_without_chunks(tmp_path, data, name, reason):
    result = process(tmp_path, data, name)
    assert result.status == 'quarantined'
    assert reason in result.reasons
    assert result.chunks == []


def test_markdown_mixed_blocks_are_preserved_but_not_approved(tmp_path):
    fence = chr(96) * 3
    data = (
        '# Heading\n\n| Item | Value |\n| --- | --- |\n| A | 42 |\n\n'
        f'{fence}python\nprint(42)\n{fence}\n\n![figure](https://example.invalid/a.png)\n'
    ).encode()
    result = process(tmp_path, data, 'mixed.md')
    assert result.status == 'quarantined'
    assert {b.kind for b in result.blocks} >= {'heading', 'table', 'code', 'image'}
    assert result.reasons == ['mixed_content_processing_deferred']
    assert result.chunks == []
    assert next(b for b in result.blocks if b.kind == 'code').attributes['executed'] is False


def test_character_limit_quarantines(tmp_path):
    result = process(tmp_path, b'a' * 101, options=Options(max_chars=100))
    assert result.reasons == ['character_limit_exceeded']


def test_pdf_text_retains_page_bbox_and_values(tmp_path, pdf_bytes):
    result = process(tmp_path, pdf_bytes(), 'report.pdf')
    assert result.status == 'approved'
    assert any(b.kind == 'heading' for b in result.blocks)
    assert all(b.locator['page'] == 1 and len(b.locator['bbox']) == 4 for b in result.blocks)
    assert '42' in ''.join(c.text for c in result.chunks)
    assert result.metrics['pages'] == 1


@pytest.mark.parametrize(
    ('arguments', 'reason'),
    [
        ({'scan': True}, 'ocr_required'),
        ({'table': True}, 'mixed_content_processing_deferred'),
        ({'columns': True}, 'complex_pdf_layout'),
        ({'encrypted': True}, 'encrypted_pdf'),
    ],
)
def test_pdf_quality_routes(tmp_path, pdf_bytes, arguments, reason):
    result = process(tmp_path, pdf_bytes(**arguments), 'report.pdf')
    assert result.status == 'quarantined'
    assert reason in result.reasons
    assert result.chunks == []
    if arguments.get('table'):
        table = next(b for b in result.blocks if b.kind == 'table')
        assert ['A', '42'] in table.attributes['rows']


def test_pdf_page_limit(tmp_path, pdf_bytes):
    result = process(tmp_path, pdf_bytes(pages=2), 'report.pdf', Options(max_pages=1))
    assert result.reasons == ['page_limit_exceeded']


def test_prose_chunks_cover_all_original_text_with_small_overlap(tmp_path):
    data = ('# Section\n\n' + 'A short sentence has value 42. ' * 60).encode()
    options = Options(chunk_chars=300, overlap_chars=70)
    result = process(tmp_path, data, 'notes.md', options)
    assert result.status == 'approved'
    assert len(result.chunks) > 2
    assert all(len(chunk.text) <= 300 for chunk in result.chunks)
    assert any(span.role == 'context' for chunk in result.chunks for span in chunk.spans)
    heading_id = result.blocks[0].id
    assert all(any(span.block_id == heading_id for span in chunk.spans) for chunk in result.chunks)
    for block in result.blocks:
        covered = set()
        for chunk in result.chunks:
            for span in chunk.spans:
                if span.block_id == block.id and span.role == 'content':
                    covered.update(range(span.start, span.end))
        assert covered == set(range(len(block.text)))


def test_atomic_table_and_code_never_split_or_overlap():
    version = uuid4()
    document = ParsedDocument(
        format='md',
        blocks=[
            RawBlock(kind='table', raw_text='|A|42|', locator={'line_start': 1, 'line_end': 1}),
            RawBlock(kind='code', raw_text='return 42', locator={'line_start': 3, 'line_end': 3}),
        ],
    )
    blocks = normalize(document, version, 'test')
    chunks = make_chunks(blocks, version, 'test', Options())
    assert [c.kind for c in chunks] == ['table', 'code']
    assert all(len(c.spans) == 1 and c.spans[0].role == 'content' for c in chunks)
    blocks[0].text = 'a' * 1900
    with pytest.raises(PreprocessFailure, match='atomic_block_too_large'):
        make_chunks(blocks, version, 'test', Options())


def test_subprocess_returns_validated_evidence(tmp_path):
    path = tmp_path / 'original.bin'
    path.write_bytes(b'Original number is 42.')
    result = execute(path, 'notes.txt', uuid4(), Options(), lambda: None, timeout=15)
    assert result.status == 'approved'
    assert '42' in result.chunks[0].text


def test_timeout_terminates_only_the_parser_child(monkeypatch):
    from unittest.mock import MagicMock

    from rag_backend.ingestion import execution

    child = MagicMock()
    child.poll.return_value = None
    monkeypatch.setattr(execution.subprocess, 'Popen', lambda *a, **k: child)
    times = iter([0, 2])
    monkeypatch.setattr(execution.time, 'monotonic', lambda: next(times))
    result = execute(Path('unused'), 'notes.txt', uuid4(), Options(), lambda: None, timeout=1)
    assert result.reasons == ['parser_timeout']
    child.kill.assert_called_once()


def test_structure_analyzer_groups_pdf_lines_and_preserves_their_boxes():
    from rag_backend.ingestion.structure import analyze

    document = ParsedDocument(
        format='pdf',
        blocks=[
            RawBlock(
                raw_text='First line',
                locator={'page': 1, 'bbox': [10, 10, 100, 20]},
                attributes={'font_size': 12},
            ),
            RawBlock(
                raw_text='Second line',
                locator={'page': 1, 'bbox': [10, 22, 90, 32]},
                attributes={'font_size': 12},
            ),
        ],
    )
    result = analyze(document)
    assert len(result.blocks) == 1
    assert result.blocks[0].raw_text == 'First line\nSecond line'
    assert len(result.blocks[0].attributes['line_bboxes']) == 2


def test_inline_html_is_preserved_and_quarantined(tmp_path):
    result = process(tmp_path, b'Text <img src=x> continues.', 'notes.md')
    assert result.status == 'quarantined'
    assert 'mixed_content_processing_deferred' in result.reasons
