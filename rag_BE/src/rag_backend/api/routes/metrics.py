from fastapi import APIRouter, Query, Request

from rag_backend.evaluation.telemetry import summarize

router = APIRouter(prefix='/metrics', tags=['metrics'])


@router.get('/answers')
def answers(request: Request, limit: int = Query(1000, ge=1, le=1000)):
    return summarize(request.app.state.database_engine, limit)
