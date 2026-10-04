"""Run with python -m rag_backend.worker [--once]."""

import argparse
import json
import time

from sqlalchemy.exc import SQLAlchemyError

from rag_backend.config import Settings
from rag_backend.indexing.embeddings import OllamaEmbedder
from rag_backend.ingestion.contracts import Options
from rag_backend.jobs.runner import JobRunner
from rag_backend.jobs.service import JobQueue
from rag_backend.storage.database import check_database, create_database_engine
from rag_backend.storage.files import LocalFileStore


def main() -> int:
    parser = argparse.ArgumentParser(description='Local verification, parsing and indexing worker')
    parser.add_argument('--once', action='store_true', help='Process at most one eligible job')
    args = parser.parse_args()
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        if not check_database(engine).ready:
            print(json.dumps({'status': 'not_ready', 'code': 'database_or_schema_unavailable'}))
            return 1
        queue = JobQueue(engine, settings.job_lease_seconds)
        runner = JobRunner(
            queue,
            LocalFileStore(settings.data_dir, settings.max_upload_bytes),
            options=Options(
                max_pages=settings.preprocess_max_pages,
                max_chars=settings.preprocess_max_chars,
                max_blocks=settings.preprocess_max_blocks,
                chunk_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            ),
            timeout=settings.preprocess_timeout_seconds,
            embedder=OllamaEmbedder(
                settings.embedding_url, settings.embedding_model, settings.embedding_http_timeout
            ),
            index_timeout=settings.index_timeout_seconds,
        )
        while True:
            try:
                result = runner.run_once()
                if result:
                    print(json.dumps(result), flush=True)
                if args.once:
                    return 0
                if result is None:
                    time.sleep(settings.worker_poll_seconds)
            except SQLAlchemyError:
                # Preserve leases for recovery; never print driver exceptions or credentials.
                print(json.dumps({'status': 'error', 'code': 'database_unavailable'}), flush=True)
                if args.once:
                    return 1
                time.sleep(settings.worker_poll_seconds)
    except KeyboardInterrupt:
        # An in-flight lease will expire and be recovered by the next worker.
        return 0
    finally:
        engine.dispose()


if __name__ == '__main__':
    raise SystemExit(main())
