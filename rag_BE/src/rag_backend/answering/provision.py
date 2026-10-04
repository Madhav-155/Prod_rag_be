"""Explicit NLI artifact download; inference never downloads files."""

import os
from uuid import uuid4

import httpx

from rag_backend.answering.nli import MODEL_ID, REVISION
from rag_backend.answering.nli_worker import verify_files
from rag_backend.config import Settings


def main():
    folder = Settings().data_dir / 'models' / 'nli-minilm2'
    folder.mkdir(parents=True, exist_ok=True)
    try:
        verify_files(folder)
        print('NLI already installed and verified.')
        return
    except (OSError, ValueError):
        pass
    for remote, local, maximum in [
        ('onnx/model.onnx', 'model.onnx', 330_000_000),
        ('tokenizer.json', 'tokenizer.json', 1_400_000),
    ]:
        temporary = folder / (uuid4().hex + '.part')
        try:
            with httpx.stream(
                'GET',
                f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{remote}',
                timeout=60,
                follow_redirects=True,
                trust_env=False,
            ) as response:
                response.raise_for_status()
                size = 0
                with temporary.open('xb') as handle:
                    for part in response.iter_bytes():
                        size += len(part)
                        if size > maximum:
                            raise ValueError('Model download exceeds size limit')
                        handle.write(part)
            os.replace(temporary, folder / local)
            print(f'Downloaded {local}: {size} bytes', flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    verify_files(folder)
    print('NLI checksums verified.')


if __name__ == '__main__':
    try:
        main()
    except (httpx.HTTPError, OSError, ValueError):
        # Redirect URLs can contain signed tokens: never dump a download traceback.
        raise SystemExit('NLI download/verification failed; retry provisioning.') from None
