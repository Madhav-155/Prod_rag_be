"""Transactional search records and publication pointers."""

from alembic import op

revision = '0004_indexing'
down_revision = '0003_preprocessing'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE source_versions
            ADD COLUMN indexing_status text NOT NULL DEFAULT 'pending'
                CHECK (indexing_status IN ('pending', 'indexed', 'failed')),
            ADD COLUMN embedding_profile text,
            ADD COLUMN indexed_at timestamptz,
            ADD CONSTRAINT version_source_pair UNIQUE(id, source_id)
    """)
    op.execute("""
        ALTER TABLE sources ADD COLUMN published_version_id uuid,
            ADD CONSTRAINT published_version_source
            FOREIGN KEY (published_version_id, id) REFERENCES source_versions(id, source_id)
    """)
    op.execute('ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check')
    op.execute(
        'ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check '
        "CHECK (kind IN ('verify_upload', 'preprocess', 'index'))"
    )
    op.execute("""
        CREATE TABLE embedding_profiles (
            id text PRIMARY KEY, provider text NOT NULL, model text NOT NULL,
            digest text NOT NULL, dimensions integer NOT NULL CHECK (dimensions = 768),
            representation text NOT NULL
        )
    """)
    op.execute(
        'ALTER TABLE evidence_chunks ADD CONSTRAINT chunk_version_pair UNIQUE(id, version_id)'
    )
    op.execute("""
        CREATE TABLE retrieval_records (
            chunk_id uuid PRIMARY KEY,
            version_id uuid NOT NULL REFERENCES source_versions(id),
            profile_id text NOT NULL REFERENCES embedding_profiles(id),
            page_content text NOT NULL CHECK (length(page_content) > 0),
            embedding vector(768) NOT NULL,
            search_terms tsvector GENERATED ALWAYS AS
                (to_tsvector('simple'::regconfig, page_content)) STORED,
            FOREIGN KEY (chunk_id, version_id) REFERENCES evidence_chunks(id, version_id)
        )
    """)
    op.execute('CREATE INDEX retrieval_fts ON retrieval_records USING gin(search_terms)')
    op.execute('CREATE INDEX retrieval_version ON retrieval_records(version_id)')
    op.execute('CREATE INDEX retrieval_profile ON retrieval_records(profile_id)')
    # Exact vector scan is intentional at MVP scale; avoid approximate recall loss.
    op.execute("""
        CREATE VIEW published_retrieval_records AS
        SELECT r.*, v.source_id FROM retrieval_records r
        JOIN source_versions v ON v.id = r.version_id
        JOIN sources s ON s.id = v.source_id AND s.published_version_id = v.id
        WHERE s.deleted_at IS NULL AND v.state = 'verified'
            AND v.preprocessing_status = 'approved' AND v.indexing_status = 'indexed'
            AND r.profile_id = v.embedding_profile
    """)
    op.execute("""
        INSERT INTO jobs(id, version_id, kind, max_attempts)
        SELECT gen_random_uuid(), v.id, 'index', j.max_attempts
        FROM source_versions v JOIN sources s ON s.id = v.source_id
        JOIN jobs j ON j.version_id = v.id AND j.kind = 'preprocess'
        WHERE v.preprocessing_status = 'approved' AND v.state = 'verified'
            AND s.deleted_at IS NULL AND j.status = 'succeeded'
        ON CONFLICT(version_id, kind) DO NOTHING
    """)


def downgrade():
    op.execute('DROP VIEW published_retrieval_records')
    op.drop_table('retrieval_records')
    op.drop_table('embedding_profiles')
    op.execute('ALTER TABLE evidence_chunks DROP CONSTRAINT chunk_version_pair')
    op.execute('ALTER TABLE sources DROP COLUMN published_version_id')
    op.execute("DELETE FROM jobs WHERE kind = 'index'")
    op.execute('ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check')
    op.execute(
        'ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check '
        "CHECK (kind IN ('verify_upload', 'preprocess'))"
    )
    op.execute(
        'ALTER TABLE source_versions DROP COLUMN indexing_status, '
        'DROP COLUMN embedding_profile, DROP COLUMN indexed_at, '
        'DROP CONSTRAINT version_source_pair'
    )
