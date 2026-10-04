from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from rag_backend.api.dependencies import get_job_queue
from rag_backend.api.schemas.jobs import JobResponse
from rag_backend.jobs.service import JobQueue

router = APIRouter(prefix='/jobs', tags=['jobs'])
Queue = Annotated[JobQueue, Depends(get_job_queue)]


@router.get('', response_model=list[JobResponse])
def list_jobs(version_id: UUID, queue: Queue):
    return queue.list_for_version(version_id)


@router.get('/{job_id}', response_model=JobResponse)
def get_job(job_id: UUID, queue: Queue):
    return queue.get(job_id)


@router.post('/{job_id}/retry', response_model=JobResponse, status_code=202)
def retry_job(job_id: UUID, queue: Queue):
    return queue.retry(job_id)
