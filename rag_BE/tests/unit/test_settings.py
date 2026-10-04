import pytest
from pydantic import ValidationError

from rag_backend.config import BACKEND_ROOT, Settings


def test_password_is_required_and_blank_is_rejected(monkeypatch):
    monkeypatch.delenv('RAG_DATABASE_PASSWORD', raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_password=' ')


def test_password_stays_masked_and_special_characters_survive():
    secret = 'local:password@with/slash%25'
    settings = Settings(_env_file=None, database_password=secret)
    assert secret not in repr(settings)
    assert secret not in str(settings.database_url())
    assert settings.database_url().password == secret


def test_data_path_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, database_password='test', data_dir='../.local-data')
    assert settings.data_dir == (BACKEND_ROOT.parent / '.local-data').resolve()
    assert not (tmp_path / '.local-data').exists()


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / '.env'
    env_file.write_text('RAG_DATABASE_PASSWORD=from-file\n', encoding='utf-8')
    monkeypatch.setenv('RAG_DATABASE_PASSWORD', 'from-environment')
    settings = Settings(_env_file=env_file)
    assert settings.database_password.get_secret_value() == 'from-environment'


def test_invalid_port_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_password='test', database_port=70000)
