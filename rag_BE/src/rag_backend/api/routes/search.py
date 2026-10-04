from typing import Annotated

from fastapi import APIRouter, Depends

from rag_backend.api.dependencies import get_retrieval_service
from rag_backend.retrieval.contracts import SearchRequest, SearchResponse
from rag_backend.retrieval.service import RetrievalService

router = APIRouter(prefix='/search', tags=['search'])


@router.post('', response_model=SearchResponse)
def search(
    request: SearchRequest, service: Annotated[RetrievalService, Depends(get_retrieval_service)]
):
    return service.search(request)
