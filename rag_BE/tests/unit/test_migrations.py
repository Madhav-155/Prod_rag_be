import ast
import os
import subprocess
import sys
from pathlib import Path

from mako.template import Template

BACKEND = Path(__file__).resolve().parents[2]


def test_migration_template_generates_valid_python():
    template = Template(filename=str(BACKEND / 'migrations' / 'script.py.mako'))
    rendered = template.render(
        message='A future migration',
        imports='',
        up_revision='test_revision',
        down_revision='0001_enable_vector',
        branch_labels=None,
        depends_on=None,
        upgrades='pass',
        downgrades='pass',
    )
    ast.parse(rendered)


def test_offline_migration_generates_vector_setup_without_a_database():
    environment = {**os.environ, 'RAG_DATABASE_PASSWORD': 'offline-test-only'}
    result = subprocess.run(
        [sys.executable, '-m', 'alembic', 'upgrade', 'head', '--sql'],
        cwd=BACKEND,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert 'CREATE EXTENSION IF NOT EXISTS vector' in result.stdout
    assert '0001_enable_vector' in result.stdout
    assert 'offline-test-only' not in result.stdout
