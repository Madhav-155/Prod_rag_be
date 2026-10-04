import hashlib
import json
import sys
from pathlib import Path

from rag_backend.answering.nli import MODEL_SHA256, TOKENIZER_BLOB


def verify_files(folder):
    with (folder / 'model.onnx').open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != MODEL_SHA256:
            raise ValueError('nli_integrity_failed')
    raw = (folder / 'tokenizer.json').read_bytes()
    if hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != TOKENIZER_BLOB:
        raise ValueError('nli_integrity_failed')


def predict(folder, pairs):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    verify_files(folder)
    if not pairs or len(pairs) > 30:
        raise ValueError('nli_input_limit')
    tokenizer = Tokenizer.from_file(str(folder / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    encodings = tokenizer.encode_batch([(p['premise'], p['hypothesis']) for p in pairs])
    if any(len(e.ids) > 512 for e in encodings):
        raise ValueError('nli_input_limit')
    ort.disable_telemetry_events()
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 2
    settings.inter_op_num_threads = 1
    settings.log_severity_level = 3
    session = ort.InferenceSession(
        str(folder / 'model.onnx'), sess_options=settings, providers=['CPUExecutionProvider']
    )
    output = []
    for start in range(0, len(encodings), 4):
        batch = encodings[start : start + 4]
        length = max(len(e.ids) for e in batch)
        fields = {
            'input_ids': ('ids', tokenizer.token_to_id('<pad>')),
            'attention_mask': ('attention_mask', 0),
            'token_type_ids': ('type_ids', 0),
        }
        inputs = {
            node.name: np.array(
                [
                    getattr(e, fields[node.name][0])
                    + [fields[node.name][1]] * (length - len(e.ids))
                    for e in batch
                ],
                dtype=np.int64,
            )
            for node in session.get_inputs()
        }
        logits = np.asarray(session.run(None, inputs)[0])
        if logits.shape != (len(batch), 3) or not np.isfinite(logits).all():
            raise ValueError('nli_invalid_scores')
        values = np.exp(logits - logits.max(axis=1, keepdims=True))
        values /= values.sum(axis=1, keepdims=True)
        output.extend(
            [
                {
                    'contradiction': float(row[0]),
                    'entailment': float(row[1]),
                    'neutral': float(row[2]),
                }
                for row in values
            ]
        )
    return output


def main():
    try:
        result = {'scores': predict(Path(sys.argv[1]), json.load(sys.stdin))}
    except Exception:
        result = {'error': 'nli_unavailable_or_input_limit'}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
