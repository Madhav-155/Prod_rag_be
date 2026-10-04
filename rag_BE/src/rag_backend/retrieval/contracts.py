import unicodedata
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=1000)
    source_ids: list[UUID] | None = Field(default=None, max_length=50)
    limit: int = Field(default=5, ge=1, le=10)

    @field_validator('query')
    @classmethod
    def normalize(cls, value):
        value = unicodedata.normalize('NFC', value)
        if any(unicodedata.category(c) == 'Cc' and c not in '\n\r\t' for c in value):
            raise ValueError('Query contains unsupported control characters')
        value = ' '.join(value.split())
        if not value or not any(c.isalnum() for c in value):
            raise ValueError('Query must contain searchable text')
        return value


class Citation(BaseModel):
    block_id: UUID
    kind: str
    locator: dict
    start: int
    end: int
    role: str
    excerpt: str


class SearchHit(BaseModel):
    chunk_id: UUID
    source_id: UUID
    version_id: UUID
    version_number: int
    filename: str
    text: str
    heading_path: list[str]
    citations: list[Citation]
    rrf_score: float
    rerank_score: float | None


class SearchResponse(BaseModel):
    request_id: UUID
    status: Literal['ok', 'no_results']
    hits: list[SearchHit]
    warnings: list[str]
    trace: dict
