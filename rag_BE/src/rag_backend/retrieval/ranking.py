import re
from difflib import SequenceMatcher


def fuse(rows, constant=60):
    merged = {}
    for row in rows:
        item = merged.setdefault(row['chunk_id'], {**row, 'rrf_score': 0.0, 'rerank_score': None})
        item['rrf_score'] += 1.0 / (constant + row['rank'])
    return sorted(merged.values(), key=lambda x: (-x['rrf_score'], str(x['chunk_id'])))


def deduplicate(candidates):
    kept, exact = [], set()
    for candidate in candidates:
        content = ' '.join(candidate['text'].split())
        if content in exact:
            continue
        # Conservative near-duplicate suppression within one immutable version only.
        numbers = re.findall(r'\d+(?:[.,]\d+)*', content)
        near = any(
            old['version_id'] == candidate['version_id']
            and covered(candidate.get('spans', []), old.get('spans', []))
            and re.findall(r'\d+(?:[.,]\d+)*', old['text']) == numbers
            and SequenceMatcher(
                None, content, ' '.join(old['text'].split()), autojunk=False
            ).ratio()
            >= 0.97
            for old in kept
        )
        if not near:
            exact.add(content)
            kept.append(candidate)
    return kept


def covered(spans, previous):
    content = [s for s in spans if s.get('role') == 'content']
    return bool(content) and all(
        any(
            prior['block_id'] == span['block_id']
            and prior['start'] <= span['start']
            and prior['end'] >= span['end']
            for prior in previous
        )
        for span in content
    )


def diverse(candidates, limit):
    # First prefer at most two hits from a source; fill remaining slots if needed.
    chosen, deferred, counts = [], [], {}
    for item in candidates:
        source = item['source_id']
        if counts.get(source, 0) < 2:
            chosen.append(item)
            counts[source] = counts.get(source, 0) + 1
        else:
            deferred.append(item)
    return (chosen + deferred)[:limit]
