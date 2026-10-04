"""Configuration for the editable local checkout. No secrets are logged."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='RAG_',
        env_file=BACKEND_ROOT / '.env',
        env_file_encoding='utf-8',
        extra='forbid',
        hide_input_in_errors=True,
    )

    database_host: str = '127.0.0.1'
    database_port: int = Field(default=55432, ge=1, le=65535)
    database_name: str = 'rag'
    database_user: str = 'rag'
    database_password: SecretStr
    data_dir: Path = BACKEND_ROOT.parent / '.local-data'
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    job_max_attempts: int = Field(default=3, ge=1, le=10)
    job_lease_seconds: int = Field(default=60, ge=5, le=3600)
    worker_poll_seconds: float = Field(default=2, gt=0, le=60)
    preprocess_timeout_seconds: int = Field(default=120, ge=1, le=600)
    preprocess_max_pages: int = Field(default=200, ge=1, le=1000)
    preprocess_max_chars: int = Field(default=2_000_000, ge=100, le=10_000_000)
    preprocess_max_blocks: int = Field(default=20_000, ge=10, le=100_000)
    chunk_max_chars: int = Field(default=1800, ge=300, le=8000)
    chunk_overlap_chars: int = Field(default=160, ge=0, le=300)
    embedding_url: str = 'http://127.0.0.1:11434'
    embedding_model: str = 'embeddinggemma:latest'
    embedding_http_timeout: int = Field(default=60, ge=1, le=60)
    index_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    generation_provider: Literal['ollama', 'openai'] = 'ollama'
    generation_url: str = 'http://127.0.0.1:11434'
    generation_model: str = 'qwen3:1.7b'
    generation_http_timeout: int = Field(default=180, ge=1, le=180)
    openai_model: str = ''
    openai_api_key: SecretStr | None = None
    remote_generation_enabled: bool = False
    remote_data_policy_reviewed: bool = False
    remote_kill_switch: bool = True
    remote_call_limit: int = Field(default=0, ge=0, le=100)

    @field_validator('database_password')
    @classmethod
    def require_password(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError('Database password must not be blank')
        return value

    @field_validator('data_dir')
    @classmethod
    def resolve_data_dir(cls, value: Path) -> Path:
        expanded = value.expanduser()
        if not expanded.is_absolute():
            expanded = BACKEND_ROOT / expanded
        return expanded.resolve()

    def database_url(self) -> URL:
        # URL.create safely handles punctuation in credentials without string concatenation.
        return URL.create(
            'postgresql+psycopg',
            username=self.database_user,
            password=self.database_password.get_secret_value(),
            host=self.database_host,
            port=self.database_port,
            database=self.database_name,
        )
