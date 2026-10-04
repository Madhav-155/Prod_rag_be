import json
import subprocess
import sys

MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L6-v2'
REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
MODEL_SHA256 = '5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a'
TOKENIZER_BLOB = '688882a79f44442ddc1f60d70334a7ff5df0fb47'
PROFILE = f'{MODEL_ID}@{REVISION}/onnx-fp32/window-max-v2'


class RerankError(Exception):
    pass


class LocalReranker:
    profile = PROFILE

    def __init__(self, model_dir, timeout=30):
        self.model_dir, self.timeout = model_dir, timeout

    def score(self, query, documents):
        if not documents:
            return []
        try:
            completed = subprocess.run(
                [sys.executable, '-m', 'rag_backend.retrieval.rerank_worker', str(self.model_dir)],
                input=json.dumps({'query': query, 'documents': documents}),
                text=True,
                encoding='utf-8',
                capture_output=True,
                timeout=self.timeout,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                check=False,
            )
            if completed.returncode != 0:
                raise RerankError('reranker_unavailable')
            payload = json.loads(completed.stdout)
            if 'error' in payload:
                raise RerankError(payload['error'])
            return payload['scores']
        except subprocess.TimeoutExpired as error:
            raise RerankError('reranker_timeout') from error
        except (OSError, ValueError, KeyError) as error:
            raise RerankError('reranker_unavailable') from error
