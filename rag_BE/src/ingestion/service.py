import hashlib
import json
from uuid import UUID, uuid4

from sqlalchemy import Engine, text

from rag_backend.domain.errors import DomainError
from rag_backend.storage.files import LocalFileStore, StagedFile, validate_filename
from rag_backend.storage.repositories.sources import get_version, replay_for_key, require_source


class SourceService:
    def __init__(self, engine: Engine, files: LocalFileStore, max_attempts: int = 3):
        self.engine = engine
        self.files = files
        self.max_attempts = max_attempts

    def accept(self, staged: StagedFile, filename: str, key: UUID, source_id: UUID | None = None):
        validate_filename(filename)
        fingerprint = hashlib.sha256(
            json.dumps(
                [str(source_id) if source_id else None, filename, staged.sha256, staged.size_bytes],
                ensure_ascii=False,
            ).encode('utf-8')
        ).hexdigest()
        # Serialize only identical request keys, then serialize updates to one source.
        # Transaction-scoped advisory locks are released even after a process crash.
        lock_id = key.int & ((1 << 63) - 1)
        with self.engine.begin() as connection:
            connection.execute(text('SELECT pg_advisory_xact_lock(:key)'), {'key': lock_id})
            replay = replay_for_key(connection, key)
            if replay:
                require_source(connection, replay['source_id'])
                if replay['request_fingerprint'] != fingerprint:
                    raise DomainError(
                        'idempotency_conflict', 'This key was used for another upload.', 409
                    )
                return {
                    'source_id': replay['source_id'],
                    'version_id': replay['id'],
                    'version_number': replay['number'],
                    'job_id': replay['job_id'],
                    'replayed': True,
                }
            if source_id is None:
                source_id = uuid4()
                connection.execute(
                    text('INSERT INTO sources(id, title) VALUES (:id, :title)'),
                    {'id': source_id, 'title': filename},
                )
                number = 1
            else:
                require_source(connection, source_id, lock=True)
                number = connection.scalar(
                    text(
                        'SELECT COALESCE(MAX(number), 0) + 1 '
                        'FROM source_versions WHERE source_id = :id'
                    ),
                    {'id': source_id},
                )
            version_id, job_id = uuid4(), uuid4()
            storage_key = f'originals/{version_id.hex}.bin'
            connection.execute(
                text("""
                INSERT INTO source_versions (
                    id, source_id, number, original_name, storage_key,
                    sha256, size_bytes, upload_key, request_fingerprint
                ) VALUES (:id, :source, :number, :name, :storage, :sha, :size, :key, :fingerprint)
            """),
                {
                    'id': version_id,
                    'source': source_id,
                    'number': number,
                    'name': filename,
                    'storage': storage_key,
                    'sha': staged.sha256,
                    'size': staged.size_bytes,
                    'key': key,
                    'fingerprint': fingerprint,
                },
            )
            connection.execute(
                text("""
                INSERT INTO jobs(id, version_id, kind, max_attempts)
                VALUES (:id, :version, 'verify_upload', :attempts)
            """),
                {'id': job_id, 'version': version_id, 'attempts': self.max_attempts},
            )
            # Rename a flushed file BEFORE the database commit makes the job visible.
            # If commit has an ambiguous outcome, retain the file: an orphan is safer
            # than deleting a possibly committed original. Reconciliation is deferred.
            self.files.promote(staged, version_id)
        return {
            'source_id': source_id,
            'version_id': version_id,
            'version_number': number,
            'job_id': job_id,
            'replayed': False,
        }

    def list_sources(self, limit: int, offset: int):
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text("""
                SELECT * FROM sources WHERE deleted_at IS NULL
                ORDER BY created_at DESC, id LIMIT :limit OFFSET :offset
            """),
                    {'limit': limit, 'offset': offset},
                ).mappings()
            ]

    def get_source(self, source_id: UUID):
        with self.engine.connect() as connection:
            return require_source(connection, source_id)

    def versions(self, source_id: UUID, limit: int, offset: int):
        with self.engine.connect() as connection:
            require_source(connection, source_id)
            return [
                dict(row)
                for row in connection.execute(
                    text("""
                SELECT * FROM source_versions WHERE source_id = :id
                ORDER BY number DESC LIMIT :limit OFFSET :offset
            """),
                    {'id': source_id, 'limit': limit, 'offset': offset},
                ).mappings()
            ]

    def evidence(self, source_id: UUID, version_id: UUID, kind: str, limit: int, offset: int):
        table = {'blocks': 'canonical_blocks', 'chunks': 'evidence_chunks'}[kind]
        with self.engine.connect() as connection:
            version = get_version(connection, version_id)
            if version['source_id'] != source_id:
                raise DomainError('version_not_found', 'Version is unavailable.', 404)
            if kind == 'chunks' and version['preprocessing_status'] != 'approved':
                return []
            return [
                dict(row)
                for row in connection.execute(
                    text(
                        f'SELECT * FROM {table} WHERE version_id = :id '
                        'ORDER BY ordinal LIMIT :limit OFFSET :offset'
                    ),
                    {'id': version_id, 'limit': limit, 'offset': offset},
                ).mappings()
            ]

    def delete(self, source_id: UUID):
        with self.engine.begin() as connection:
            # Lock ordering matches job completion/retry: source, then job.
            require_source(connection, source_id, lock=True)
            connection.execute(
                text('UPDATE sources SET deleted_at = clock_timestamp() WHERE id = :id'),
                {'id': source_id},
            )
            connection.execute(
                text("""
                UPDATE jobs SET status = 'cancelled', lease_token = NULL,
                    lease_expires_at = NULL, updated_at = clock_timestamp(),
                    last_error_code = 'source_deleted'
                WHERE version_id IN (SELECT id FROM source_versions WHERE source_id = :id)
                    AND status IN ('queued', 'running', 'failed')
            """),
                {'id': source_id},
            )
        return {'deleted': True, 'original_files_retained': True}
