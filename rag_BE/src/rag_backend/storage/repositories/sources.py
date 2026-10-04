from uuid import UUID

from sqlalchemy import Connection, text

from rag_backend.domain.errors import DomainError


def require_source(connection: Connection, source_id: UUID, *, lock: bool = False):
    suffix = ' FOR UPDATE' if lock else ''
    row = (
        connection.execute(
            text('SELECT * FROM sources WHERE id = :id AND deleted_at IS NULL' + suffix),
            {'id': source_id},
        )
        .mappings()
        .first()
    )
    if row is None:
        raise DomainError('source_not_found', 'Source does not exist or was deleted.', 404)
    return dict(row)


def replay_for_key(connection: Connection, key: UUID):
    row = (
        connection.execute(
            text("""
        SELECT v.*, j.id AS job_id
        FROM source_versions v JOIN jobs j ON j.version_id = v.id
        WHERE v.upload_key = :key AND j.kind = 'verify_upload'
    """),
            {'key': key},
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


def get_version(connection: Connection, version_id: UUID):
    row = (
        connection.execute(
            text("""
        SELECT v.* FROM source_versions v JOIN sources s ON s.id = v.source_id
        WHERE v.id = :id AND s.deleted_at IS NULL
    """),
            {'id': version_id},
        )
        .mappings()
        .first()
    )
    if row is None:
        raise DomainError('version_not_found', 'Version is unavailable.', 404)
    return dict(row)
