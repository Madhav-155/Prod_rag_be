from fastapi import Request

from rag_backend.conversations.repository import ConversationRepository
from rag_backend.conversations.service import ConversationService
from rag_backend.ingestion.service import SourceService
from rag_backend.jobs.service import JobQueue
from rag_backend.storage.database import DatabaseChecks, check_database


def get_retrieval_service(request: Request):
    return request.app.state.retrieval_service


def get_answer_service(request: Request):
    return request.app.state.answer_service


def get_conversation_service(request: Request):
    return ConversationService(
        ConversationRepository(request.app.state.database_engine),
        request.app.state.answer_service,
    )


def get_database_checks(request: Request) -> DatabaseChecks:
    return check_database(request.app.state.database_engine)


def get_source_service(request: Request) -> SourceService:
    return SourceService(
        request.app.state.database_engine,
        request.app.state.file_store,
        request.app.state.settings.job_max_attempts,
    )


def get_job_queue(request: Request) -> JobQueue:
    return JobQueue(request.app.state.database_engine, request.app.state.settings.job_lease_seconds)
