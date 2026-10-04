import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

from rag_backend.ingestion.contracts import Options, PreprocessingResult


def execute(
    path: Path, filename: str, version_id: UUID, options: Options, heartbeat, timeout: int = 120
) -> PreprocessingResult:
    request = json.dumps(
        {
            'path': str(path),
            'filename': filename,
            'version_id': str(version_id),
            'options': options.model_dump(),
        }
    ).encode()
    process = subprocess.Popen(
        [sys.executable, '-m', 'rag_backend.ingestion.parse_worker'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return PreprocessingResult(
                    version_id=version_id,
                    profile='restructure-v1',
                    status='quarantined',
                    reasons=['parser_timeout'],
                )
            try:
                output, _ = process.communicate(input=request, timeout=min(1, remaining))
                break
            except subprocess.TimeoutExpired:
                request = None
                heartbeat()
        if process.returncode:
            return PreprocessingResult(
                version_id=version_id,
                profile='restructure-v1',
                status='quarantined',
                reasons=['parser_process_failed'],
            )
        try:
            data = json.loads(output)
        except (ValueError, UnicodeDecodeError):
            return PreprocessingResult(
                version_id=version_id,
                profile='restructure-v1',
                status='quarantined',
                reasons=['parser_protocol_failed'],
            )
        if data.get('error') == 'storage_unavailable':
            raise OSError('Parser could not read original')
        if data.get('error'):
            return PreprocessingResult(
                version_id=version_id,
                profile='restructure-v1',
                status='quarantined',
                reasons=['parser_failed'],
            )
        return PreprocessingResult.model_validate(data['result'])
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
