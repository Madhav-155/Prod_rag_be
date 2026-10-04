from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

Kind = Literal['paragraph', 'heading', 'table', 'code', 'image', 'html']


class PreprocessFailure(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class Options(BaseModel):
    max_pages: int = Field(200, ge=1)
    max_chars: int = Field(2_000_000, ge=1)
    max_blocks: int = Field(20_000, ge=1)
    chunk_chars: int = Field(1800, ge=100)
    overlap_chars: int = Field(160, ge=0)

    @model_validator(mode='after')
    def validate_overlap(self):
        if self.overlap_chars >= self.chunk_chars // 2:
            raise ValueError('Overlap must use less than half of the chunk budget')
        return self


class RawBlock(BaseModel):
    kind: Kind = 'paragraph'
    raw_text: str
    locator: dict
    attributes: dict = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    format: Literal['pdf', 'txt', 'md']
    blocks: list[RawBlock]
    metrics: dict = Field(default_factory=dict)


class CanonicalBlock(RawBlock):
    id: UUID
    ordinal: int
    text: str
    heading_path: list[str] = Field(default_factory=list)


class EvidenceSpan(BaseModel):
    block_id: UUID
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    role: Literal['content', 'context'] = 'content'


class Chunk(BaseModel):
    id: UUID
    ordinal: int
    kind: Kind
    text: str
    heading_path: list[str]
    spans: list[EvidenceSpan]


class PreprocessingResult(BaseModel):
    version_id: UUID
    profile: str
    status: Literal['approved', 'quarantined']
    reasons: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    completed_stages: list[str] = Field(default_factory=list)
    blocks: list[CanonicalBlock] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
