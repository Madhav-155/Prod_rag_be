import hashlib
import unicodedata
from uuid import UUID, uuid5

from rag_backend.ingestion.contracts import CanonicalBlock, ParsedDocument


def normalize(document: ParsedDocument, version_id: UUID, profile: str) -> list[CanonicalBlock]:
    blocks = []
    headings = []
    for ordinal, raw in enumerate(document.blocks):
        # NFC and newline normalization preserve values; raw extraction remains unchanged.
        text = unicodedata.normalize('NFC', raw.raw_text.replace('\r\n', '\n').replace('\r', '\n'))
        if raw.kind == 'heading':
            level = raw.attributes.get('level', 1)
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, raw.attributes.get('title', text.strip())))
        identity = f'{profile}:block:{ordinal}:{hashlib.sha256(raw.raw_text.encode()).hexdigest()}'
        blocks.append(
            CanonicalBlock(
                **raw.model_dump(),
                id=uuid5(version_id, identity),
                ordinal=ordinal,
                text=text,
                heading_path=[title for _, title in headings],
            )
        )
    return blocks
