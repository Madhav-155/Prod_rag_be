import csv
import io
import statistics
from pathlib import Path

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException

from rag_backend.ingestion.contracts import Options, ParsedDocument, PreprocessFailure, RawBlock


def parse(path: Path, options: Options) -> ParsedDocument:
    blocks = []
    metrics = {'pages': 0, 'ocr_pages': [], 'blank_pages': [], 'complex_layout_pages': []}
    total_chars = 0
    try:
        with pdfplumber.open(path) as pdf:
            if pdf.doc.encryption is not None:
                raise PreprocessFailure('encrypted_pdf')
            if len(pdf.pages) > options.max_pages:
                raise PreprocessFailure('page_limit_exceeded')
            metrics['pages'] = len(pdf.pages)
            for page in pdf.pages:
                total_chars += len(page.chars)
                if total_chars > options.max_chars:
                    raise PreprocessFailure('character_limit_exceeded')
                page_blocks = []
                tables = page.find_tables()
                locations = [table.bbox for table in tables]
                for table in tables:
                    rows = table.extract()
                    output = io.StringIO()
                    csv.writer(output).writerows(rows)
                    page_blocks.append(
                        RawBlock(
                            kind='table',
                            raw_text=output.getvalue(),
                            locator={'page': page.page_number, 'bbox': list(table.bbox)},
                            attributes={
                                'rows': rows,
                                'representation': 'csv',
                                'structure_only': True,
                            },
                        )
                    )
                for image in page.images:
                    bbox = [image['x0'], image['top'], image['x1'], image['bottom']]
                    page_blocks.append(
                        RawBlock(
                            kind='image',
                            raw_text='',
                            locator={'page': page.page_number, 'bbox': bbox},
                            attributes={
                                'object_name': str(image.get('name', '')),
                                'reference_only': True,
                            },
                        )
                    )
                if page.curves:
                    page_blocks.append(
                        RawBlock(
                            kind='image',
                            raw_text='',
                            locator={
                                'page': page.page_number,
                                'bbox': [0, 0, page.width, page.height],
                            },
                            attributes={'vector_graphics': True, 'reference_only': True},
                        )
                    )
                if page.rotation or any(not char.get('upright', True) for char in page.chars):
                    metrics['complex_layout_pages'].append(page.page_number)
                if any(
                    annotation.get('data', {}).get('Subtype')
                    and 'Widget' in str(annotation['data']['Subtype'])
                    for annotation in page.annots
                ):
                    metrics['complex_layout_pages'].append(page.page_number)
                lines = page.extract_text_lines(layout=False, strip=False, return_chars=True)
                for line in lines:
                    center = ((line['x0'] + line['x1']) / 2, (line['top'] + line['bottom']) / 2)
                    if any(
                        x0 <= center[0] <= x1 and top <= center[1] <= bottom
                        for x0, top, x1, bottom in locations
                    ):
                        continue
                    chars = sorted(line.get('chars', []), key=lambda item: item['x0'])
                    if any(
                        b['x0'] - a['x1'] > page.width * 0.12
                        for a, b in zip(chars, chars[1:], strict=False)
                    ):
                        metrics['complex_layout_pages'].append(page.page_number)
                    sizes = [char['size'] for char in chars]
                    page_blocks.append(
                        RawBlock(
                            raw_text=line['text'],
                            locator={
                                'page': page.page_number,
                                'bbox': [line['x0'], line['top'], line['x1'], line['bottom']],
                            },
                            attributes={'font_size': statistics.median(sizes) if sizes else 0},
                        )
                    )
                text_length = sum(len(block.raw_text.strip()) for block in page_blocks)
                if text_length < 20 and (page.images or page.curves):
                    metrics['ocr_pages'].append(page.page_number)
                if not page_blocks:
                    if page.chars or page.lines or page.rects:
                        metrics['ocr_pages'].append(page.page_number)
                    else:
                        metrics['blank_pages'].append(page.page_number)
                page_blocks.sort(
                    key=lambda block: (block.locator['bbox'][1], block.locator['bbox'][0])
                )
                blocks.extend(page_blocks)
                if len(blocks) > options.max_blocks:
                    raise PreprocessFailure('block_limit_exceeded')
                page.close()
    except PdfminerException as error:
        cause = error.args[0] if error.args else None
        code = 'encrypted_pdf' if isinstance(cause, PDFPasswordIncorrect) else 'malformed_pdf'
        raise PreprocessFailure(code) from error
    except PDFPasswordIncorrect as error:
        raise PreprocessFailure('encrypted_pdf') from error
    except PDFSyntaxError as error:
        raise PreprocessFailure('malformed_pdf') from error
    metrics['complex_layout_pages'] = sorted(set(metrics['complex_layout_pages']))
    return ParsedDocument(format='pdf', blocks=blocks, metrics=metrics)
