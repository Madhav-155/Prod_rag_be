from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class UploadReceipt(BaseModel):
    source_id: UUID
    version_id: UUID
    version_number: int
    job_id: UUID
    replayed: bool


class SourceResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    published_version_id: UUID | None


class VersionResponse(BaseModel):
    id: UUID
    source_id: UUID
    number: int
    original_name: str
    sha256: str
    size_bytes: int
    state: Literal['stored', 'verified', 'failed']
    created_at: datetime
    verified_at: datetime | None
    preprocessing_status: Literal['pending', 'approved', 'quarantined', 'failed']
    preprocessing_profile: str | None
    quality_report: dict
    preprocessed_at: datetime | None
    indexing_status: Literal['pending', 'indexed', 'failed']
    embedding_profile: str | None
    indexed_at: datetime | None


class DeletionResponse(BaseModel):
    deleted: bool
    original_files_retained: bool
