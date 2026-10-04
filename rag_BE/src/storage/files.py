"""Stream originals to disk without trusting client filenames as paths."""

import hashlib
import os
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID, uuid4

from starlette.concurrency import run_in_threadpool

from rag_backend.domain.errors import DomainError


def validate_filename(filename: str) -> str:
    if (
        not filename
        or len(filename) > 255
        or any(ord(char) < 32 or ord(char) == 127 for char in filename)
        or PureWindowsPath(filename).name != filename
        or PurePosixPath(filename).name != filename
        or ':' in filename
    ):
        raise DomainError('invalid_filename', 'Use a plain filename without a directory.', 422)
    if Path(filename).suffix.lower() not in {'.pdf', '.txt', '.md'}:
        raise DomainError('unsupported_type', 'Accepted files: PDF, TXT, and Markdown.', 415)
    return filename


@dataclass(frozen=True)
class StagedFile:
    path: Path
    sha256: str
    size_bytes: int


class LocalFileStore:
    def __init__(self, root: Path, max_bytes: int):
        self.root = root.resolve()
        self.max_bytes = max_bytes

    def resolve(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root) or candidate == self.root:
            raise ValueError('Storage key is outside the configured directory')
        return candidate

    async def receive(self, chunks: AsyncIterable[bytes]) -> StagedFile:
        staging = self.resolve('staging')
        staging.mkdir(parents=True, exist_ok=True)
        path = self.resolve(f'staging/{uuid4().hex}.part')
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open('xb') as handle:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise DomainError(
                            'upload_too_large', 'Upload exceeds the configured limit.', 413
                        )
                    if chunk:
                        await run_in_threadpool(handle.write, chunk)
                        digest.update(chunk)
                if not size:
                    raise DomainError('empty_upload', 'The uploaded file is empty.', 422)
                await run_in_threadpool(handle.flush)
                await run_in_threadpool(os.fsync, handle.fileno())
            return StagedFile(path, digest.hexdigest(), size)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def discard(self, staged: StagedFile) -> None:
        path = staged.path.resolve()
        if not path.is_relative_to(self.resolve('staging')):
            raise ValueError('Only staged uploads can be discarded')
        path.unlink(missing_ok=True)

    def promote(self, staged: StagedFile, version_id: UUID) -> str:
        key = f'originals/{version_id.hex}.bin'
        destination = self.resolve(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Generated IDs prevent client-driven overwrites. Originals are never edited.
        if destination.exists():
            raise FileExistsError('Original already exists')
        os.replace(staged.path, destination)
        return key

    def verify(self, key: str, sha256: str, size_bytes: int, heartbeat: Callable[[], None]) -> bool:
        digest = hashlib.sha256()
        size = 0
        with self.resolve(key).open('rb') as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                if size > size_bytes:
                    return False
                digest.update(chunk)
                heartbeat()
        return size == size_bytes and digest.hexdigest() == sha256
