from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Response

from rag_backend.api.dependencies import get_conversation_service
from rag_backend.conversations.service import ConversationService, TurnRequest

router = APIRouter(prefix='/conversations', tags=['conversations'])
Service = Annotated[ConversationService, Depends(get_conversation_service)]


@router.post('', status_code=201)
def create(service: Service):
    return service.repository.create()


@router.get('')
def list_conversations(
    service: Service, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0, le=10000)
):
    return service.repository.list(limit, offset)


@router.get('/{conversation_id}')
def history(conversation_id: UUID, service: Service):
    return service.history(conversation_id)


@router.post('/{conversation_id}/turns')
def turn(
    conversation_id: UUID,
    request: TurnRequest,
    service: Service,
    idempotency_key: Annotated[UUID, Header(alias='Idempotency-Key')],
):
    return service.turn(conversation_id, idempotency_key, request)


@router.delete('/{conversation_id}', status_code=204)
def delete(conversation_id: UUID, service: Service):
    service.repository.delete(conversation_id)
    return Response(status_code=204)
