"""Canonical original blocks and traceable candidate chunks."""

from alembic import op

revision = '0003_preprocessing'
down_revision = '0002_sources_jobs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE source_versions
            ADD COLUMN preprocessing_status text NOT NULL DEFAULT 'pending'
                CHECK (preprocessing_status IN ('pending', 'approved', 'quarantined', 'failed')),
            ADD COLUMN preprocessing_profile text,
            ADD COLUMN quality_report jsonb NOT NULL DEFAULT '{}'::jsonb,
            ADD COLUMN preprocessed_at timestamptz
    """)
    op.execute('ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check')
    op.execute(
        'ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check '
        "CHECK (kind IN ('verify_upload', 'preprocess'))"
    )
    op.execute("""
        CREATE TABLE canonical_blocks (
            id uuid PRIMARY KEY,
            version_id uuid NOT NULL REFERENCES source_versions(id),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            kind text NOT NULL,
            raw_text text NOT NULL,
            text text NOT NULL,
            locator jsonb NOT NULL,
            attributes jsonb NOT NULL,
            heading_path jsonb NOT NULL,
            UNIQUE(version_id, ordinal)
        )
    """)
    op.execute("""
        CREATE TABLE evidence_chunks (
            id uuid PRIMARY KEY,
            version_id uuid NOT NULL REFERENCES source_versions(id),
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            kind text NOT NULL,
            text text NOT NULL,
            heading_path jsonb NOT NULL,
            spans jsonb NOT NULL,
            UNIQUE(version_id, ordinal)
        )
    """)
    # Resume originals verified by Stage 2 without changing their immutable versions.
    op.execute("""
        INSERT INTO jobs(id, version_id, kind, max_attempts)
        SELECT gen_random_uuid(), v.id, 'preprocess', j.max_attempts
        FROM source_versions v
        JOIN sources s ON s.id = v.source_id
        JOIN jobs j ON j.version_id = v.id AND j.kind = 'verify_upload'
        WHERE v.state = 'verified' AND s.deleted_at IS NULL AND j.status = 'succeeded'
        ON CONFLICT (version_id, kind) DO NOTHING
    """)


def downgrade() -> None:
    op.drop_table('evidence_chunks')
    op.drop_table('canonical_blocks')
    op.execute("DELETE FROM jobs WHERE kind = 'preprocess'")
    op.execute('ALTER TABLE jobs DROP CONSTRAINT jobs_kind_check')
    op.execute("ALTER TABLE jobs ADD CONSTRAINT jobs_kind_check CHECK (kind = 'verify_upload')")
    op.execute("""
        ALTER TABLE source_versions
            DROP COLUMN preprocessing_status, DROP COLUMN preprocessing_profile,
            DROP COLUMN quality_report, DROP COLUMN preprocessed_at
    """)
