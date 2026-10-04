"""Explicit download only: python -m rag_backend.retrieval.provision."""

import os
from uuid import uuid4

import httpx

from rag_backend.config import Settings
from rag_backend.retrieval.rerank_worker import verify_files
from rag_backend.retrieval.reranking import MODEL_ID, REVISION


def main():
    folder = Settings().data_dir / 'models' / 'minilm-l6-v2'
    folder.mkdir(parents=True, exist_ok=True)
    try:
        verify_files(folder)
        print('Reranker already provisioned and verified.')
        return
    except ValueError:
        pass
    for remote, local, maximum in [
        ('onnx/model.onnx', 'model.onnx', 100_000_000),
        ('tokenizer.json', 'tokenizer.json', 1_000_000),
    ]:
        staging = folder / (uuid4().hex + '.part')
        try:
            with httpx.stream(
                'GET',
                f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{remote}',
                follow_redirects=True,
                timeout=60,
                trust_env=False,
            ) as response:
                response.raise_for_status()
                size = 0
                with staging.open('xb') as handle:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > maximum:
                            raise ValueError('Download exceeds pinned artifact size limit')
                        handle.write(chunk)
            os.replace(staging, folder / local)
            print(f'Downloaded {local}: {size} bytes', flush=True)
        finally:
            staging.unlink(missing_ok=True)
    verify_files(folder)
    print('Reranker model and tokenizer checksums verified.')


if __name__ == '__main__':
    main()
