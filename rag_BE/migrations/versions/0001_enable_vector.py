"""Enable pgvector inside the authoritative PostgreSQL database."""

from alembic import op

revision = '0001_enable_vector'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')


def downgrade() -> None:
    # Leave the shared extension installed; later migrations own their tables.
    # Never use DROP EXTENSION ... CASCADE to undo application schema changes.
    pass
