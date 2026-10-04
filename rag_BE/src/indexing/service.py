import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from uuid import UUID

from sqlalchemy import text

from rag_backend.indexing.embeddings import Embedder, EmbeddingError, Profile, validate_vectors


@dataclass
class IndexResult:
    version_id: UUID
    profile: Profile
    chunk_ids: list[UUID]
    vectors: list[list[float]]


def build_index(engine, version_id, embedder: Embedder, heartbeat, timeout=300):
    with engine.connect() as connection:
        rows = list(
            connection.execute(
                text("""
            SELECT c.id, c.text FROM evidence_chunks c
            JOIN source_versions v ON v.id = c.version_id
            JOIN sources s ON s.id = v.source_id
            WHERE v.id = :id AND v.state = 'verified'
                AND v.preprocessing_status = 'approved' AND s.deleted_at IS NULL
            ORDER BY c.ordinal
        """),
                {'id': version_id},
            ).mappings()
        )
    if not rows or len(rows) > 20000:
        raise EmbeddingError('index_evidence_unavailable_or_limit')
    deadline = time.monotonic() + timeout
    pool = ThreadPoolExecutor(max_workers=1)

    def bounded(call, *args):
        future = pool.submit(call, *args)
        while True:
            heartbeat()
            if time.monotonic() >= deadline:
                raise EmbeddingError('index_timeout', True)
            try:
                return future.result(timeout=0.5)
            except TimeoutError:
                continue

    try:
        profile = bounded(embedder.profile)
        if profile.dimensions != 768:
            raise EmbeddingError('embedding_dimension_mismatch')
        vectors = []
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            values = bounded(embedder.embed, [row['text'] for row in batch], profile)
            vectors.extend(validate_vectors(values, len(batch)))
        if bounded(embedder.profile) != profile:
            raise EmbeddingError('embedding_model_changed')
        heartbeat()
        return IndexResult(version_id, profile, [row['id'] for row in rows], vectors)
    finally:
        # In-flight HTTP calls have their own timeout and cannot write to the database.
        pool.shutdown(wait=False, cancel_futures=True)


def persist_index(connection, version_id, result: IndexResult):
    version = (
        connection.execute(text('SELECT * FROM source_versions WHERE id=:id'), {'id': version_id})
        .mappings()
        .one()
    )
    if (
        version_id != result.version_id
        or version['state'] != 'verified'
        or version['preprocessing_status'] != 'approved'
    ):
        raise ValueError('Invalid indexing version')
    chunks = list(
        connection.execute(
            text('SELECT id, text FROM evidence_chunks WHERE version_id=:id ORDER BY ordinal'),
            {'id': version_id},
        ).mappings()
    )
    if not chunks or [r['id'] for r in chunks] != result.chunk_ids:
        raise ValueError('Index must cover the complete canonical chunk set')
    vectors = validate_vectors(result.vectors, len(chunks))
    profile = result.profile
    if profile.dimensions != 768 or not profile.digest:
        raise ValueError('Invalid embedding profile')
    connection.execute(
        text("""
        INSERT INTO embedding_profiles(id, provider, model, digest, dimensions, representation)
        VALUES (:id, :provider, :model, :digest, :dimensions, :representation)
        ON CONFLICT(id) DO NOTHING
    """),
        {'id': profile.id, **asdict(profile)},
    )
    connection.execute(
        text("""
        INSERT INTO retrieval_records(chunk_id, version_id, profile_id, page_content, embedding)
        VALUES (:chunk, :version, :profile, :content, CAST(:vector AS vector))
    """),
        [
            {
                'chunk': c['id'],
                'version': version_id,
                'profile': profile.id,
                'content': c['text'],
                'vector': json.dumps(v),
            }
            for c, v in zip(chunks, vectors, strict=True)
        ],
    )
    connection.execute(
        text("""
        UPDATE source_versions SET indexing_status='indexed', embedding_profile=:profile,
            indexed_at=clock_timestamp() WHERE id=:id
    """),
        {'id': version_id, 'profile': profile.id},
    )
    # Source row is locked by the caller. A late older job cannot roll publication back.
    connection.execute(
        text("""
        UPDATE sources s SET published_version_id=:version WHERE s.id=:source
            AND (s.published_version_id IS NULL OR
                (SELECT number FROM source_versions WHERE id=s.published_version_id) < :number)
    """),
        {'version': version_id, 'source': version['source_id'], 'number': version['number']},
    )
