import asyncio
import hashlib
from uuid import uuid4

import pytest

from rag_backend.domain.errors import DomainError
from rag_backend.storage.files import LocalFileStore, validate_filename


async def chunks(*items):
    for item in items:
        yield item


def test_stream_promote_and_verify(tmp_path):
    store = LocalFileStore(tmp_path, 100)
    staged = asyncio.run(store.receive(chunks(b'hello', b' world')))
    assert staged.sha256 == hashlib.sha256(b'hello world').hexdigest()
    assert staged.size_bytes == 11
    key = store.promote(staged, uuid4())
    assert not staged.path.exists()
    assert store.verify(key, staged.sha256, 11, lambda: None)
    assert not store.verify(key, '0' * 64, 11, lambda: None)


@pytest.mark.parametrize('content', [(), (b'123', b'456')])
def test_empty_or_oversized_upload_leaves_no_staging_file(tmp_path, content):
    store = LocalFileStore(tmp_path, 5)
    with pytest.raises(DomainError):
        asyncio.run(store.receive(chunks(*content)))
    assert list(tmp_path.rglob('*.part')) == []


def test_interrupted_stream_cleans_up(tmp_path):
    async def interrupted():
        yield b'part'
        raise ConnectionError('client disconnected')

    store = LocalFileStore(tmp_path, 100)
    with pytest.raises(ConnectionError):
        asyncio.run(store.receive(interrupted()))
    assert list(tmp_path.rglob('*.part')) == []


@pytest.mark.parametrize(
    'name',
    [
        '../file.pdf',
        r'..\file.pdf',
        '/tmp/file.pdf',
        r'C:\file.pdf',
        'file.pdf:stream',
        'bad\x00.txt',
        'file.exe',
        '',
    ],
)
def test_rejects_invalid_names(name):
    with pytest.raises(DomainError):
        validate_filename(name)


def test_accepts_supported_unicode_filename():
    assert validate_filename('report notes.PDF') == 'report notes.PDF'
    assert validate_filename('दस्तावेज.md') == 'दस्तावेज.md'


def test_storage_key_cannot_escape_root(tmp_path):
    store = LocalFileStore(tmp_path, 100)
    with pytest.raises(ValueError):
        store.resolve('../outside')
