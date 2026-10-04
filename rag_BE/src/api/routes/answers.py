from typing import Annotated

from fastapi import APIRouter, Depends

from rag_backend.answering.contracts import AnswerRequest, AnswerResponse
from rag_backend.answering.service import AnswerService
from rag_backend.api.dependencies import get_answer_service

router = APIRouter(prefix='/answers', tags=['answers'])


@router.post('', response_model=AnswerResponse)
def answer(request: AnswerRequest, service: Annotated[AnswerService, Depends(get_answer_service)]):
    return service.answer(request)
