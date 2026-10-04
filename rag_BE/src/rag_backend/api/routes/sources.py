from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request
from starlette.concurrency import run_in_threadpool

from rag_backend.api.dependencies import get_source_service
from rag_backend.api.schemas.sources import (
    DeletionResponse,
    SourceResponse,
    UploadReceipt,
    VersionResponse,
)
from rag_backend.domain.errors import DomainError
from rag_backend.ingestion.contracts import CanonicalBlock, Chunk
from rag_backend.ingestion.service import SourceService
from rag_backend.storage.files import validate_filename

router = APIRouter(prefix='/sources', tags=['sources'])
Service = Annotated[SourceService, Depends(get_source_service)]
BINARY_BODY = {
    'requestBody': {
        'required': True,
        'content': {'application/octet-stream': {'schema': {'type': 'string', 'format': 'binary'}}},
    }
}


async def receive_upload(
    request: Request,
    service: SourceService,
    filename: str,
    key: UUID,
    source_id: UUID | None = None,
):
    validate_filename(filename)
    if (
        request.headers.get('content-type', '').split(';')[0].strip().lower()
        != 'application/octet-stream'
    ):
        raise DomainError(
            'invalid_content_type', 'Send the raw file as application/octet-stream.', 415
        )
    content_length = request.headers.get('content-length')
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError as error:
            raise DomainError('invalid_length', 'Invalid Content-Length.', 400) from error
        if length < 0:
            raise DomainError('invalid_length', 'Invalid Content-Length.', 400)
        if length > service.files.max_bytes:
            raise DomainError('upload_too_large', 'Upload exceeds the configured limit.', 413)
    staged = None
    try:
        staged = await service.files.receive(request.stream())
        return await run_in_threadpool(service.accept, staged, filename, key, source_id)
    except OSError as error:
        raise DomainError(
            'storage_unavailable', 'Local file storage is unavailable.', 503
        ) from error
    finally:
        if staged is not None:
            await run_in_threadpool(service.files.discard, staged)


@router.post('', response_model=UploadReceipt, status_code=202, openapi_extra=BINARY_BODY)
async def upload_source(
    request: Request,
    service: Service,
    filename: Annotated[str, Query(min_length=1, max_length=255)],
    idempotency_key: Annotated[UUID, Header()],
):
    return await receive_upload(request, service, filename, idempotency_key)


@router.post(
    '/{source_id}/versions',
    response_model=UploadReceipt,
    status_code=202,
    openapi_extra=BINARY_BODY,
)
async def upload_version(
    source_id: UUID,
    request: Request,
    service: Service,
    filename: Annotated[str, Query(min_length=1, max_length=255)],
    idempotency_key: Annotated[UUID, Header()],
):
    return await receive_upload(request, service, filename, idempotency_key, source_id)


@router.get('', response_model=list[SourceResponse])
def list_sources(
    service: Service, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
):
    return service.list_sources(limit, offset)


@router.get('/{source_id}', response_model=SourceResponse)
def get_source(source_id: UUID, service: Service):
    return service.get_source(source_id)


@router.get('/{source_id}/versions', response_model=list[VersionResponse])
def list_versions(
    source_id: UUID,
    service: Service,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return service.versions(source_id, limit, offset)


@router.delete('/{source_id}', response_model=DeletionResponse)
def delete_source(source_id: UUID, service: Service):
    return service.delete(source_id)


@router.get('/{source_id}/versions/{version_id}/blocks', response_model=list[CanonicalBlock])
def original_blocks(
    source_id: UUID,
    version_id: UUID,
    service: Service,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return service.evidence(source_id, version_id, 'blocks', limit, offset)


@router.get('/{source_id}/versions/{version_id}/chunks', response_model=list[Chunk])
def candidate_chunks(
    source_id: UUID,
    version_id: UUID,
    service: Service,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return service.evidence(source_id, version_id, 'chunks', limit, offset)
