"""Offline CPU cross-encoder. Only JSON data enters; no remote code/downloads."""

import hashlib
import json
import sys
from pathlib import Path

from rag_backend.retrieval.reranking import MODEL_SHA256, TOKENIZER_BLOB


def verify_files(folder):
    model, tokenizer = folder / 'model.onnx', folder / 'tokenizer.json'
    if not model.is_file() or not tokenizer.is_file():
        raise ValueError('reranker_not_provisioned')
    with model.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != MODEL_SHA256:
            raise ValueError('reranker_integrity_failed')
    data = tokenizer.read_bytes()
    if (
        hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        != TOKENIZER_BLOB
    ):
        raise ValueError('reranker_integrity_failed')
    return model, tokenizer


def build_windows(query_ids, documents, cls_id=101, sep_id=102):
    if len(query_ids) > 128:
        raise ValueError('reranker_query_limit')
    available = 512 - len(query_ids) - 3
    windows, owners = [], []
    for owner, document in enumerate(documents):
        offset, count = 0, 0
        while True:
            part = document[offset : offset + available]
            count += 1
            if count > 4:
                raise ValueError('reranker_window_limit')
            ids = [cls_id, *query_ids, sep_id, *part, sep_id]
            windows.append(
                {
                    'input_ids': ids,
                    'attention_mask': [1] * len(ids),
                    'token_type_ids': [0] * (len(query_ids) + 2) + [1] * (len(part) + 1),
                }
            )
            owners.append(owner)
            if offset + available >= len(document):
                break
            offset += available - 32
    return windows, owners


def score(folder, query, documents):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    if len(documents) > 20 or len(query) > 1000 or any(len(d) > 8000 for d in documents):
        raise ValueError('reranker_input_limit')
    model_path, tokenizer_path = verify_files(folder)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Window the complete token IDs ourselves: tokenizer overflow processing in the
    # installed release lost tail tokens in a regression test. This pinned model
    # uses BERT pair layout [CLS] query [SEP] passage [SEP].
    tokenizer.no_truncation()
    tokenizer.no_padding()
    query_ids = tokenizer.encode(query, add_special_tokens=False).ids
    encoded = tokenizer.encode_batch(documents, add_special_tokens=False)
    windows, owners = build_windows(
        query_ids,
        [item.ids for item in encoded],
        tokenizer.token_to_id('[CLS]'),
        tokenizer.token_to_id('[SEP]'),
    )
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=['CPUExecutionProvider']
    )
    scores = [float('-inf')] * len(documents)
    for start in range(0, len(windows), 8):
        batch = windows[start : start + 8]
        width = max(len(item['input_ids']) for item in batch)
        inputs = {
            node.name: np.asarray(
                [item[node.name] + [0] * (width - len(item['input_ids'])) for item in batch],
                dtype=np.int64,
            )
            for node in session.get_inputs()
        }
        output = np.asarray(session.run(None, inputs)[0]).reshape(-1)
        if len(output) != len(batch) or not np.isfinite(output).all():
            raise ValueError('reranker_invalid_scores')
        for offset, value in enumerate(output):
            owner = owners[start + offset]
            scores[owner] = max(scores[owner], float(value))
    return scores


def main():
    try:
        payload = json.load(sys.stdin)
        result = {'scores': score(Path(sys.argv[1]), payload['query'], payload['documents'])}
    except ValueError as error:
        allowed = {
            'reranker_not_provisioned',
            'reranker_integrity_failed',
            'reranker_input_limit',
            'reranker_query_limit',
            'reranker_window_limit',
            'reranker_invalid_scores',
        }
        result = {'error': str(error) if str(error) in allowed else 'reranker_unavailable'}
    except Exception:
        result = {'error': 'reranker_unavailable'}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
