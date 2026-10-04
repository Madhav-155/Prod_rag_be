"""Bounded subprocess protocol: local file in, structured evidence out."""

import json
import sys
from pathlib import Path
from uuid import UUID

from rag_backend.ingestion.contracts import Options
from rag_backend.ingestion.pipeline import run


def main() -> int:
    try:
        request = json.loads(sys.stdin.buffer.read())
        result = run(
            Path(request['path']),
            request['filename'],
            UUID(request['version_id']),
            Options.model_validate(request['options']),
        )
        response = {'result': result.model_dump(mode='json')}
    except OSError:
        response = {'error': 'storage_unavailable'}
    except Exception:
        # Never send untrusted parser exceptions, document content, or local paths into job logs.
        response = {'error': 'parser_failed'}
    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=True).encode('utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
