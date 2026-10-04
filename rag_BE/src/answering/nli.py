import json
import subprocess
import sys

MODEL_ID = 'cross-encoder/nli-MiniLM2-L6-H768'
REVISION = 'b95119ce93d3e065de6214e38cd4a97b0f2f2c6d'
MODEL_SHA256 = '807d33fffefd95ad60e2a91f54eed50a92aa53c077945a6842bb587a49a3fdf3'
TOKENIZER_BLOB = '8df0092a08e1d459f60ec541e08cd35a16362bfe'
PROFILE = f'{MODEL_ID}@{REVISION}/onnx-fp32-v1'


class NLIError(Exception):
    pass


class LocalNLI:
    profile = PROFILE

    def __init__(self, folder, timeout=45):
        self.folder, self.timeout = folder, timeout

    def predict(self, pairs):
        try:
            response = subprocess.run(
                [sys.executable, '-m', 'rag_backend.answering.nli_worker', str(self.folder)],
                input=json.dumps(pairs),
                encoding='utf-8',
                text=True,
                capture_output=True,
                timeout=self.timeout,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            data = json.loads(response.stdout)
            if response.returncode or 'scores' not in data:
                raise NLIError('nli_unavailable_or_input_limit')
            return data['scores']
        except (subprocess.TimeoutExpired, OSError, ValueError) as error:
            raise NLIError('nli_unavailable_or_input_limit') from error
