"""Original uploads, immutable versions, and leased background jobs."""

from alembic import op

revision = '0002_sources_jobs'
down_revision = '0001_enable_vector'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE sources (
            id uuid PRIMARY KEY,
            title text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            deleted_at timestamptz
        )
    """)
    op.execute("""
        CREATE TABLE source_versions (
            id uuid PRIMARY KEY,
            source_id uuid NOT NULL REFERENCES sources(id),
            number integer NOT NULL CHECK (number > 0),
            original_name text NOT NULL,
            storage_key text NOT NULL UNIQUE,
            sha256 varchar(64) NOT NULL CHECK (length(sha256) = 64),
            size_bytes bigint NOT NULL CHECK (size_bytes > 0),
            upload_key uuid NOT NULL UNIQUE,
            request_fingerprint varchar(64) NOT NULL,
            state text NOT NULL DEFAULT 'stored'
                CHECK (state IN ('stored', 'verified', 'failed')),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            verified_at timestamptz,
            UNIQUE(source_id, number)
        )
    """)
    op.execute("""
        CREATE TABLE jobs (
            id uuid PRIMARY KEY,
            version_id uuid NOT NULL REFERENCES source_versions(id),
            kind text NOT NULL CHECK (kind = 'verify_upload'),
            status text NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
            attempts integer NOT NULL DEFAULT 0,
            max_attempts integer NOT NULL CHECK (max_attempts BETWEEN 1 AND 10),
            available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            lease_token uuid,
            lease_expires_at timestamptz,
            last_error_code varchar(80),
            created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            UNIQUE(version_id, kind),
            CHECK (attempts >= 0 AND attempts <= max_attempts),
            CHECK (
                (status = 'running' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
                OR
                (status <> 'running' AND lease_token IS NULL AND lease_expires_at IS NULL)
            )
        )
    """)
    op.execute('CREATE INDEX ix_sources_active ON sources(created_at, id) WHERE deleted_at IS NULL')
    op.execute(
        'CREATE INDEX ix_jobs_poll ON jobs(status, available_at, lease_expires_at, created_at)'
    )


def downgrade() -> None:
    op.drop_table('jobs')
    op.drop_table('source_versions')
    op.drop_table('sources')
