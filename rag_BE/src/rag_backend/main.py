"""Run with: python -m uvicorn rag_backend.main:create_app --factory --host 127.0.0.1"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_backend import __version__
from rag_backend.answering.nli import LocalNLI
from rag_backend.answering.providers import OllamaGenerator, OpenAIGenerator
from rag_backend.answering.service import AnswerService
from rag_backend.api.errors import install_error_handlers
from rag_backend.api.routes.answers import router as answers_router
from rag_backend.api.routes.conversations import router as conversations_router
from rag_backend.api.routes.health import router as health_router
from rag_backend.api.routes.jobs import router as jobs_router
from rag_backend.api.routes.metrics import router as metrics_router
from rag_backend.api.routes.search import router as search_router
from rag_backend.api.routes.sources import router as sources_router
from rag_backend.config import Settings
from rag_backend.indexing.embeddings import OllamaEmbedder
from rag_backend.retrieval.repository import RetrievalRepository
from rag_backend.retrieval.reranking import LocalReranker
from rag_backend.retrieval.service import RetrievalService
from rag_backend.storage.database import create_database_engine
from rag_backend.storage.files import LocalFileStore


def create_app(settings: Settings | None = None) -> FastAPI:
    configuration = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_database_engine(configuration)
        app.state.database_engine = engine
        app.state.settings = configuration
        app.state.retrieval_service = RetrievalService(
            RetrievalRepository(engine),
            OllamaEmbedder(
                configuration.embedding_url,
                configuration.embedding_model,
                configuration.embedding_http_timeout,
            ),
            LocalReranker(configuration.data_dir / 'models' / 'minilm-l6-v2'),
        )
        app.state.file_store = LocalFileStore(
            configuration.data_dir, configuration.max_upload_bytes
        )
        generator = (
            OllamaGenerator(
                configuration.generation_url,
                configuration.generation_model,
                configuration.generation_http_timeout,
            )
            if configuration.generation_provider == 'ollama'
            else OpenAIGenerator(configuration)
        )
        app.state.answer_service = AnswerService(
            app.state.retrieval_service,
            generator,
            LocalNLI(configuration.data_dir / 'models' / 'nli-minilm2'),
        )
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title='RAG Backend',
        version=__version__,
        description='Stage 7: local conversations, grounded answering and offline evaluation.',
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(sources_router)
    app.include_router(jobs_router)
    app.include_router(search_router)
    app.include_router(answers_router)
    app.include_router(conversations_router)
    app.include_router(metrics_router)
    install_error_handlers(app)
    return app
