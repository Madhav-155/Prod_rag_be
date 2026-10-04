from pathlib import Path

from markdown_it import MarkdownIt

from rag_backend.ingestion.contracts import Options, ParsedDocument, PreprocessFailure, RawBlock


def parse(path: Path, format: str, options: Options) -> ParsedDocument:
    try:
        text = path.read_bytes().decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise PreprocessFailure('unsupported_text_encoding') from error
    if len(text) > options.max_chars:
        raise PreprocessFailure('character_limit_exceeded')
    if '\x00' in text:
        raise PreprocessFailure('binary_content_in_text')
    lines = text.splitlines(keepends=True)
    blocks = []
    covered = set()

    def append(start, end, kind='paragraph', attributes=None):
        raw = ''.join(lines[start:end])
        if raw.strip():
            blocks.append(
                RawBlock(
                    kind=kind,
                    raw_text=raw,
                    locator={'line_start': start + 1, 'line_end': end},
                    attributes=attributes or {},
                )
            )
        covered.update(range(start, end))

    if format == 'md':
        tokens = MarkdownIt('commonmark').enable('table').parse(text)
        for index, token in enumerate(tokens):
            if token.map is None:
                continue
            start, end = token.map
            if any(i in covered for i in range(start, end)):
                continue
            if token.type == 'heading_open':
                title = tokens[index + 1].content
                append(start, end, 'heading', {'level': int(token.tag[1:]), 'title': title})
            elif token.type in {'fence', 'code_block'}:
                append(start, end, 'code', {'language': token.info.strip(), 'executed': False})
            elif token.type == 'table_open':
                append(start, end, 'table', {'representation': 'markdown'})
            elif token.type == 'html_block':
                append(start, end, 'html')
            elif token.type == 'paragraph_open':
                children = tokens[index + 1].children or []
                kind = 'image' if any(c.type == 'image' for c in children) else 'paragraph'
                if any(c.type == 'html_inline' for c in children):
                    kind = 'html'
                append(start, end, kind, {'external_resources_fetched': False})
    # Plain text and any unclaimed Markdown lines remain traceable; never drop syntax silently.
    start = None
    for index in range(len(lines) + 1):
        usable = index < len(lines) and index not in covered and bool(lines[index].strip())
        if usable and start is None:
            start = index
        if not usable and start is not None:
            append(start, index)
            start = None
    blocks.sort(key=lambda block: block.locator['line_start'])
    if len(blocks) > options.max_blocks:
        raise PreprocessFailure('block_limit_exceeded')
    return ParsedDocument(format=format, blocks=blocks, metrics={'lines': len(lines)})
