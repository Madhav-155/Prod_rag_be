import time

from rag_backend.domain.errors import DomainError, LostLease
from rag_backend.indexing.embeddings import EmbeddingError
from rag_backend.indexing.service import build_index
from rag_backend.ingestion.contracts import Options
from rag_backend.ingestion.execution import execute
from rag_backend.jobs.service import JobQueue
from rag_backend.storage.files import LocalFileStore


class JobRunner:
    def __init__(
        self,
        queue: JobQueue,
        files: LocalFileStore,
        options: Options | None = None,
        timeout: int = 120,
        embedder=None,
        index_timeout: int = 300,
    ):
        self.queue = queue
        self.files = files
        self.options = options or Options()
        self.timeout = timeout
        self.embedder = embedder
        self.index_timeout = index_timeout

    def run_once(self):
        kinds = (
            ('verify_upload', 'preprocess', 'index')
            if self.embedder
            else ('verify_upload', 'preprocess')
        )
        job = self.queue.claim(kinds)
        if job is None:
            return None
        job_id, token = job['id'], job['lease_token']
        last_heartbeat = time.monotonic()

        def heartbeat():
            nonlocal last_heartbeat
            if time.monotonic() - last_heartbeat >= self.queue.lease_seconds / 3:
                self.queue.heartbeat(job_id, token)
                last_heartbeat = time.monotonic()

        code, retryable = None, False
        result = None
        try:
            if job['kind'] not in {'verify_upload', 'preprocess', 'index'}:
                code = 'unsupported_job'
            else:
                version = self.queue.payload(job['version_id'])
                if not self.files.verify(
                    version['storage_key'], version['sha256'], version['size_bytes'], heartbeat
                ):
                    code = 'original_integrity_failed'
                elif job['kind'] == 'preprocess':
                    result = execute(
                        self.files.resolve(version['storage_key']),
                        version['original_name'],
                        version['id'],
                        self.options,
                        heartbeat,
                        self.timeout,
                    )
                elif job['kind'] == 'index':
                    result = build_index(
                        self.queue.engine,
                        version['id'],
                        self.embedder,
                        heartbeat,
                        self.index_timeout,
                    )
        except EmbeddingError as error:
            code, retryable = error.code, error.retryable
        except (LostLease, DomainError):
            return {'job_id': str(job_id), 'status': 'lease_lost'}
        except FileNotFoundError:
            code = 'original_missing'
        except OSError:
            code, retryable = 'storage_unavailable', True
        except ValueError:
            code = 'invalid_storage_key'

        try:
            if code:
                status = self.queue.fail(job_id, token, code, retryable=retryable)
            elif result is not None and job['kind'] == 'index':
                self.queue.complete_indexing(job_id, token, result)
                status = 'succeeded'
            elif result is not None:
                status = self.queue.complete_preprocessing(job_id, token, result)
            else:
                self.queue.complete(job_id, token)
                status = 'succeeded'
        except LostLease:
            status = 'lease_lost'
        return {'job_id': str(job_id), 'kind': job['kind'], 'status': status}
