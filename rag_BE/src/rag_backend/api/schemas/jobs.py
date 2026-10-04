from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class JobResponse(BaseModel):
    id: UUID
    version_id: UUID
    kind: Literal['verify_upload', 'preprocess', 'index']
    status: Literal['queued', 'running', 'succeeded', 'failed', 'cancelled']
    attempts: int
    max_attempts: int
    available_at: datetime
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
