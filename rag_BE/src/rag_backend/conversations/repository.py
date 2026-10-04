import hashlib
import json
from uuid import uuid4

from sqlalchemy import text

from rag_backend.domain.errors import DomainError


class ConversationRepository:
    def __init__(self, engine):
        self.engine = engine

    def create(self):
        with self.engine.begin() as connection:
            return dict(
                connection.execute(
                    text(
                        'INSERT INTO conversations(id) VALUES (:id) '
                        'RETURNING id, created_at, expires_at'
                    ),
                    {'id': uuid4()},
                )
                .mappings()
                .one()
            )

    def list(self, limit=50, offset=0):
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text(
                        'SELECT id, created_at, expires_at FROM conversations '
                        'WHERE expires_at > now() ORDER BY created_at DESC, id '
                        'LIMIT :limit OFFSET :offset'
                    ),
                    {'limit': limit, 'offset': offset},
                ).mappings()
            ]

    def _lock(self, connection, conversation_id):
        row = (
            connection.execute(
                text(
                    'SELECT *, active_until > now() AS busy FROM conversations '
                    'WHERE id=:id AND expires_at > now() FOR UPDATE'
                ),
                {'id': conversation_id},
            )
            .mappings()
            .first()
        )
        if row is None:
            raise DomainError('conversation_not_found', 'Conversation is unavailable.', 404)
        return row

    def reserve(self, conversation_id, key, request):
        fingerprint = hashlib.sha256(
            json.dumps(request.model_dump(mode='json'), sort_keys=True).encode()
        ).hexdigest()
        with self.engine.begin() as connection:
            conversation = self._lock(connection, conversation_id)
            existing = (
                connection.execute(
                    text(
                        'SELECT * FROM conversation_turns WHERE conversation_id=:id '
                        'AND idempotency_key=:key'
                    ),
                    {'id': conversation_id, 'key': key},
                )
                .mappings()
                .first()
            )
            if existing:
                if existing['request_hash'] != fingerprint:
                    raise DomainError(
                        'idempotency_conflict', 'Key belongs to another request.', 409
                    )
                if existing['status'] == 'completed':
                    return existing['id'], existing['response'], ''
                raise DomainError(
                    'turn_not_replayable',
                    'Turn pending or failed; inspect history before retrying.',
                    409,
                )
            if conversation['active_turn'] and conversation['busy']:
                raise DomainError('conversation_busy', 'A turn is already in progress.', 409)
            if conversation['active_turn']:
                connection.execute(
                    text(
                        "UPDATE conversation_turns SET status='failed', "
                        "error_code='lease_expired', completed_at=now() "
                        "WHERE id=:id AND status='pending'"
                    ),
                    {'id': conversation['active_turn']},
                )
            ordinal = connection.scalar(
                text(
                    'SELECT coalesce(max(ordinal), 0)+1 FROM conversation_turns '
                    'WHERE conversation_id=:id'
                ),
                {'id': conversation_id},
            )
            if ordinal > 100:
                raise DomainError(
                    'conversation_full', 'Start a new conversation after 100 turns.', 409
                )
            recent = list(
                connection.scalars(
                    text(
                        'SELECT query FROM conversation_turns WHERE conversation_id=:id '
                        "AND status='completed' ORDER BY ordinal DESC LIMIT 3"
                    ),
                    {'id': conversation_id},
                )
            )
            # Only complete prior user questions; assistant claims never become evidence.
            context = ''
            for query in reversed(recent):
                if len(context) + len(query) + 1 <= 1000:
                    context += query + '\n'
            turn_id = uuid4()
            connection.execute(
                text(
                    'INSERT INTO conversation_turns(id, conversation_id, ordinal, '
                    'idempotency_key, request_hash, query, status) '
                    "VALUES (:turn, :id, :ordinal, :key, :hash, :query, 'pending')"
                ),
                {
                    'turn': turn_id,
                    'id': conversation_id,
                    'ordinal': ordinal,
                    'key': key,
                    'hash': fingerprint,
                    'query': request.query,
                },
            )
            connection.execute(
                text(
                    'UPDATE conversations SET active_turn=:turn, '
                    "active_until=now()+interval '15 minutes' WHERE id=:id"
                ),
                {'turn': turn_id, 'id': conversation_id},
            )
            return turn_id, None, context

    def complete(self, conversation_id, turn_id, snapshot):
        with self.engine.begin() as connection:
            conversation = self._lock(connection, conversation_id)
            if conversation['active_turn'] != turn_id or not conversation['busy']:
                raise DomainError('turn_fenced', 'Turn expired or was superseded.', 409)
            connection.execute(
                text(
                    "UPDATE conversation_turns SET status='completed', "
                    'response=CAST(:response AS jsonb), '
                    "completed_at=now() WHERE id=:turn AND status='pending'"
                ),
                {'turn': turn_id, 'response': json.dumps(snapshot)},
            )
            connection.execute(
                text(
                    'INSERT INTO answer_events(id, conversation_id, turn_id, outcome, telemetry) '
                    'VALUES (:event, :id, :turn, :outcome, CAST(:trace AS jsonb))'
                ),
                {
                    'event': uuid4(),
                    'id': conversation_id,
                    'turn': turn_id,
                    'outcome': snapshot['status'],
                    'trace': json.dumps(snapshot['trace']),
                },
            )
            connection.execute(
                text('UPDATE conversations SET active_turn=NULL, active_until=NULL WHERE id=:id'),
                {'id': conversation_id},
            )

    def fail(self, conversation_id, turn_id):
        with self.engine.begin() as connection:
            # Same lock ordering as reserve/complete/delete. No error bodies or queries in events.
            row = connection.execute(
                text('SELECT active_turn FROM conversations WHERE id=:id FOR UPDATE'),
                {'id': conversation_id},
            ).first()
            if row is None or row.active_turn != turn_id:
                return
            connection.execute(
                text(
                    "UPDATE conversation_turns SET status='failed', error_code='answer_failed', "
                    "completed_at=now() WHERE id=:turn AND status='pending'"
                ),
                {'turn': turn_id},
            )
            connection.execute(
                text('UPDATE conversations SET active_turn=NULL, active_until=NULL WHERE id=:id'),
                {'id': conversation_id},
            )
            connection.execute(
                text(
                    'INSERT INTO answer_events(id, conversation_id, turn_id, outcome, telemetry) '
                    "VALUES (:event, :id, :turn, 'failed', '{}'::jsonb)"
                ),
                {'event': uuid4(), 'id': conversation_id, 'turn': turn_id},
            )

    def history(self, conversation_id):
        with self.engine.begin() as connection:
            self._lock(connection, conversation_id)
            return [
                dict(row)
                for row in connection.execute(
                    text(
                        'SELECT id, ordinal, query, status, response, error_code, '
                        'created_at, completed_at '
                        'FROM conversation_turns WHERE conversation_id=:id ORDER BY ordinal'
                    ),
                    {'id': conversation_id},
                ).mappings()
            ]

    def availability(self, chunk_ids):
        if not chunk_ids:
            return set()
        with self.engine.connect() as connection:
            return {
                str(value)
                for value in connection.scalars(
                    text(
                        'SELECT chunk_id FROM published_retrieval_records '
                        'WHERE chunk_id=ANY(CAST(:ids AS uuid[]))'
                    ),
                    {'ids': list(chunk_ids)},
                )
            }

    def delete(self, conversation_id):
        with self.engine.begin() as connection:
            result = connection.execute(
                text('DELETE FROM conversations WHERE id=:id'), {'id': conversation_id}
            )
            return bool(result.rowcount)

    def purge_expired(self):
        with self.engine.begin() as connection:
            return connection.execute(
                text('DELETE FROM conversations WHERE expires_at <= now()')
            ).rowcount
