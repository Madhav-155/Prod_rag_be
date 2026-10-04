import json
from uuid import UUID, uuid4

from sqlalchemy import Engine, text

from rag_backend.domain.errors import DomainError, LostLease
from rag_backend.indexing.service import IndexResult, persist_index
from rag_backend.ingestion.contracts import PreprocessingResult
from rag_backend.storage.repositories.evidence import persist_result
from rag_backend.storage.repositories.sources import get_version, require_source


class JobQueue:
    def __init__(self, engine: Engine, lease_seconds: int = 60):
        self.engine = engine
        self.lease_seconds = lease_seconds

    def get(self, job_id: UUID):
        with self.engine.connect() as connection:
            row = (
                connection.execute(text('SELECT * FROM jobs WHERE id = :id'), {'id': job_id})
                .mappings()
                .first()
            )
            if row is None:
                raise DomainError('job_not_found', 'Job does not exist.', 404)
            return dict(row)

    def payload(self, version_id: UUID):
        with self.engine.connect() as connection:
            return get_version(connection, version_id)

    def list_for_version(self, version_id: UUID):
        with self.engine.connect() as connection:
            get_version(connection, version_id)
            return [
                dict(row)
                for row in connection.execute(
                    text('SELECT * FROM jobs WHERE version_id = :id ORDER BY created_at, id'),
                    {'id': version_id},
                ).mappings()
            ]

    def _mark_failure(self, connection, row, status, code):
        if row['kind'] == 'verify_upload':
            connection.execute(
                text('UPDATE source_versions SET state = :state WHERE id = :id'),
                {'state': 'failed' if status == 'failed' else 'stored', 'id': row['version_id']},
            )
        elif row['kind'] == 'index':
            connection.execute(
                text('UPDATE source_versions SET indexing_status=:status WHERE id=:id'),
                {'status': 'failed' if status == 'failed' else 'pending', 'id': row['version_id']},
            )
        else:
            connection.execute(
                text("""
                UPDATE source_versions SET preprocessing_status = :state,
                    quality_report = CAST(:report AS jsonb)
                WHERE id = :id
            """),
                {
                    'state': 'failed' if status == 'failed' else 'pending',
                    'id': row['version_id'],
                    'report': json.dumps({'reasons': [code], 'searchable': False}),
                },
            )

    def recover_exhausted(self) -> int:
        with self.engine.begin() as connection:
            rows = (
                connection.execute(
                    text("""
                SELECT id, version_id, kind FROM jobs
                WHERE status = 'running' AND lease_expires_at <= clock_timestamp()
                    AND attempts >= max_attempts
                LIMIT 100 FOR UPDATE SKIP LOCKED
            """)
                )
                .mappings()
                .all()
            )
            for row in rows:
                connection.execute(
                    text("""
                    UPDATE jobs SET status = 'failed', lease_token = NULL,
                        lease_expires_at = NULL, last_error_code = 'lease_expired',
                        updated_at = clock_timestamp()
                    WHERE id = :id
                """),
                    {'id': row['id']},
                )
                self._mark_failure(connection, row, 'failed', 'lease_expired')
            return len(rows)

    def claim(self, kinds=('verify_upload', 'preprocess', 'index')):
        self.recover_exhausted()
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    text("""
                WITH candidate AS (
                    SELECT j.id FROM jobs j
                    JOIN source_versions v ON v.id = j.version_id
                    JOIN sources s ON s.id = v.source_id
                    WHERE s.deleted_at IS NULL AND j.kind = ANY(:kinds)
                        AND j.attempts < j.max_attempts AND (
                        (j.status = 'queued' AND j.available_at <= clock_timestamp()) OR
                        (j.status = 'running' AND j.lease_expires_at <= clock_timestamp())
                    )
                    ORDER BY j.created_at, j.id
                    LIMIT 1 FOR UPDATE OF j SKIP LOCKED
                )
                UPDATE jobs j SET status = 'running', attempts = j.attempts + 1,
                    lease_token = :token,
                    lease_expires_at = clock_timestamp() + make_interval(secs => :seconds),
                    updated_at = clock_timestamp()
                FROM candidate c WHERE j.id = c.id RETURNING j.*
            """),
                    {'token': uuid4(), 'seconds': self.lease_seconds, 'kinds': list(kinds)},
                )
                .mappings()
                .first()
            )
            return dict(row) if row else None

    def heartbeat(self, job_id: UUID, token: UUID) -> None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text("""
                UPDATE jobs j SET
                    lease_expires_at = clock_timestamp() + make_interval(secs => :seconds),
                    updated_at = clock_timestamp()
                WHERE j.id = :id AND j.status = 'running' AND j.lease_token = :token
                    AND j.lease_expires_at > clock_timestamp()
                    AND EXISTS (
                        SELECT 1 FROM source_versions v JOIN sources s ON s.id = v.source_id
                        WHERE v.id = j.version_id AND s.deleted_at IS NULL
                    )
                RETURNING j.id
            """),
                {'id': job_id, 'token': token, 'seconds': self.lease_seconds},
            ).first()
            if row is None:
                raise LostLease()

    def _lock_active_source(self, connection, job_id: UUID):
        source = connection.scalar(
            text("""
            SELECT v.source_id FROM jobs j JOIN source_versions v ON v.id = j.version_id
            WHERE j.id = :id
        """),
            {'id': job_id},
        )
        if source is None:
            raise LostLease()
        try:
            require_source(connection, source, lock=True)
        except DomainError as error:
            raise LostLease() from error

    def complete(self, job_id: UUID, token: UUID) -> None:
        with self.engine.begin() as connection:
            self._lock_active_source(connection, job_id)
            version_id = connection.scalar(
                text("""
                UPDATE jobs SET status = 'succeeded', lease_token = NULL,
                    lease_expires_at = NULL, last_error_code = NULL,
                    updated_at = clock_timestamp()
                WHERE id = :id AND kind = 'verify_upload'
                    AND status = 'running' AND lease_token = :token
                    AND lease_expires_at > clock_timestamp()
                RETURNING version_id
            """),
                {'id': job_id, 'token': token},
            )
            if version_id is None:
                raise LostLease()
            connection.execute(
                text("""
                UPDATE source_versions SET state = 'verified', verified_at = clock_timestamp()
                WHERE id = :id
            """),
                {'id': version_id},
            )
            connection.execute(
                text("""
                INSERT INTO jobs(id, version_id, kind, max_attempts)
                SELECT :id, version_id, 'preprocess', max_attempts FROM jobs WHERE id = :parent
                ON CONFLICT(version_id, kind) DO NOTHING
            """),
                {'id': uuid4(), 'parent': job_id},
            )

    def complete_preprocessing(self, job_id: UUID, token: UUID, result: PreprocessingResult):
        with self.engine.begin() as connection:
            self._lock_active_source(connection, job_id)
            status = 'succeeded' if result.status == 'approved' else 'failed'
            code = result.reasons[0] if result.reasons else None
            version_id = connection.scalar(
                text("""
                UPDATE jobs SET status = :status, lease_token = NULL,
                    lease_expires_at = NULL, last_error_code = :code,
                    updated_at = clock_timestamp()
                WHERE id = :id AND kind = 'preprocess' AND status = 'running'
                    AND lease_token = :token AND lease_expires_at > clock_timestamp()
                RETURNING version_id
            """),
                {'id': job_id, 'token': token, 'status': status, 'code': code},
            )
            if version_id is None:
                raise LostLease()
            persist_result(connection, version_id, result)
            if result.status == 'approved':
                connection.execute(
                    text("""
                    INSERT INTO jobs(id, version_id, kind, max_attempts)
                    SELECT :id, version_id, 'index', max_attempts FROM jobs WHERE id=:parent
                    ON CONFLICT(version_id, kind) DO NOTHING
                """),
                    {'id': uuid4(), 'parent': job_id},
                )
            return status

    def complete_indexing(self, job_id: UUID, token: UUID, result: IndexResult):
        with self.engine.begin() as connection:
            self._lock_active_source(connection, job_id)
            version_id = connection.scalar(
                text("""
                UPDATE jobs SET status='succeeded', lease_token=NULL, lease_expires_at=NULL,
                    last_error_code=NULL, updated_at=clock_timestamp()
                WHERE id=:id AND kind='index' AND status='running' AND lease_token=:token
                    AND lease_expires_at > clock_timestamp() RETURNING version_id
            """),
                {'id': job_id, 'token': token},
            )
            if version_id is None:
                raise LostLease()
            persist_index(connection, version_id, result)

    def fail(self, job_id: UUID, token: UUID, code: str, *, retryable: bool) -> str:
        # Codes are developer-defined identifiers, never raw exception messages.
        if not code.replace('_', '').isalnum() or len(code) > 80:
            raise ValueError('Invalid job error code')
        with self.engine.begin() as connection:
            self._lock_active_source(connection, job_id)
            row = (
                connection.execute(
                    text("""
                UPDATE jobs SET
                    status = CASE WHEN :retryable AND attempts < max_attempts
                        THEN 'queued' ELSE 'failed' END,
                    available_at = clock_timestamp()
                        + make_interval(secs => LEAST(60, CAST(power(2, attempts) AS integer))),
                    lease_token = NULL, lease_expires_at = NULL,
                    last_error_code = :code, updated_at = clock_timestamp()
                WHERE id = :id AND status = 'running' AND lease_token = :token
                    AND lease_expires_at > clock_timestamp()
                RETURNING version_id, status, kind
            """),
                    {'id': job_id, 'token': token, 'code': code, 'retryable': retryable},
                )
                .mappings()
                .first()
            )
            if row is None:
                raise LostLease()
            self._mark_failure(connection, row, row['status'], code)
            return row['status']

    def retry(self, job_id: UUID):
        with self.engine.begin() as connection:
            source = connection.scalar(
                text("""
                SELECT v.source_id FROM jobs j JOIN source_versions v ON v.id = j.version_id
                WHERE j.id = :id
            """),
                {'id': job_id},
            )
            if source is None:
                raise DomainError('job_not_found', 'Job does not exist.', 404)
            require_source(connection, source, lock=True)
            quarantined = connection.scalar(
                text("""
                SELECT v.preprocessing_status = 'quarantined' FROM source_versions v
                JOIN jobs j ON j.version_id = v.id WHERE j.id = :id AND j.kind = 'preprocess'
            """),
                {'id': job_id},
            )
            if quarantined:
                raise DomainError(
                    'quarantined', 'Inspect quality reasons; upload a corrected new version.', 409
                )
            row = (
                connection.execute(
                    text("""
                UPDATE jobs SET status = 'queued', available_at = clock_timestamp(),
                    last_error_code = NULL, updated_at = clock_timestamp()
                WHERE id = :id AND status = 'failed' AND attempts < max_attempts
                RETURNING *
            """),
                    {'id': job_id},
                )
                .mappings()
                .first()
            )
            if row is None:
                raise DomainError(
                    'retry_not_allowed', 'Only failed jobs with attempts remaining can retry.', 409
                )
            self._mark_failure(connection, row, 'queued', 'retry_requested')
            return dict(row)
