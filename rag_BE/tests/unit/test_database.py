from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from rag_backend.storage.database import SCHEMA_REVISION, check_database


@pytest.mark.parametrize(
    ('scalars', 'expected'),
    [
        ([True, 'alembic_version', SCHEMA_REVISION], (True, True, True)),
        ([True, None], (True, True, False)),
        ([False, 'alembic_version', SCHEMA_REVISION], (True, False, True)),
        ([True, 'alembic_version', 'old_schema'], (True, True, False)),
    ],
)
def test_database_readiness(scalars, expected):
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.scalar.side_effect = scalars
    checks = check_database(engine)
    assert (checks.database, checks.pgvector, checks.schema) == expected


def test_database_failure_is_sanitized():
    engine = MagicMock()
    engine.connect.side_effect = OperationalError('connection', {}, Exception('secret-password'))
    checks = check_database(engine)
    assert not checks.ready
    assert 'secret-password' not in repr(checks)
