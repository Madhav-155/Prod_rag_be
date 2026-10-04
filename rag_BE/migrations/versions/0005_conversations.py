"""Local conversation history; generated answers are not evidence."""

from alembic import op

revision = '0005_conversations'
down_revision = '0004_indexing'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE conversations (
            id uuid PRIMARY KEY,
            created_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz NOT NULL DEFAULT now() + interval '30 days',
            active_turn uuid,
            active_until timestamptz
        )
    """)
    op.execute("""
        CREATE TABLE conversation_turns (
            id uuid PRIMARY KEY,
            conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            ordinal integer NOT NULL,
            idempotency_key uuid NOT NULL,
            request_hash text NOT NULL,
            query text NOT NULL,
            status text NOT NULL CHECK(status IN ('pending', 'completed', 'failed')),
            response jsonb,
            error_code text,
            created_at timestamptz NOT NULL DEFAULT now(),
            completed_at timestamptz,
            UNIQUE(conversation_id, idempotency_key),
            UNIQUE(conversation_id, ordinal)
        )
    """)
    op.execute("""
        CREATE TABLE answer_events (
            id uuid PRIMARY KEY,
            conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            turn_id uuid NOT NULL REFERENCES conversation_turns(id) ON DELETE CASCADE,
            outcome text NOT NULL,
            telemetry jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute('CREATE INDEX conversation_expiry ON conversations(expires_at)')
    op.execute('CREATE INDEX answer_event_time ON answer_events(created_at)')


def downgrade():
    op.drop_table('answer_events')
    op.drop_table('conversation_turns')
    op.drop_table('conversations')
