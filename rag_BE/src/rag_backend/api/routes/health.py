from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from rag_backend.api.dependencies import get_database_checks
from rag_backend.storage.database import DatabaseChecks

router = APIRouter(prefix='/health', tags=['health'])


class LivenessResponse(BaseModel):
    status: Literal['ok'] = 'ok'


class ReadinessResponse(BaseModel):
    status: Literal['ready', 'not_ready']
    checks: DatabaseChecks


@router.get('/live', response_model=LivenessResponse)
def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    '/ready',
    response_model=ReadinessResponse,
    responses={503: {'model': ReadinessResponse, 'description': 'Infrastructure not ready'}},
)
def readiness(
    response: Response,
    checks: Annotated[DatabaseChecks, Depends(get_database_checks)],
) -> ReadinessResponse:
    if not checks.ready:
        response.status_code = 503
    return ReadinessResponse(status='ready' if checks.ready else 'not_ready', checks=checks)
