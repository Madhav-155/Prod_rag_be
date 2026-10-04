from collections import Counter

from sqlalchemy import text

from rag_backend.evaluation.report import percentile


def summarize(engine, limit=1000):
    # Small bounded local dashboard dataset; no queries, responses or evidence returned.
    with engine.connect() as connection:
        rows = list(
            connection.execute(
                text(
                    'SELECT e.outcome, e.telemetry FROM answer_events e '
                    'JOIN conversations c ON c.id=e.conversation_id WHERE c.expires_at > now() '
                    'ORDER BY e.created_at DESC, e.id LIMIT :limit'
                ),
                {'limit': limit},
            ).mappings()
        )
    timings, costs, stages = [], [], {}
    unknown_cost = 0
    for row in rows:
        trace = row['telemetry']
        if 'total_ms' in trace:
            timings.append(trace['total_ms'])
        if 'planning_ms' in trace:
            stages.setdefault('planning', []).append(trace['planning_ms'])
        for key, value in trace.get('retrieval', {}).get('stage_ms', {}).items():
            stages.setdefault('retrieval.' + key, []).append(value)
        for tool in trace.get('tools', []):
            stages.setdefault('tool.' + tool['tool'], []).append(tool['latency_ms'])
        generations = trace.get('generations', [])
        stages.setdefault('generation', []).extend(g['latency_ms'] for g in generations)
        usage = [g.get('usage', {}).get('api_cost_usd') for g in generations]
        if row['outcome'] == 'failed' or any(cost is None for cost in usage):
            unknown_cost += 1
        else:
            costs.append(sum(usage))
    return {
        'scope': 'Latest retained conversation events only; stateless /answers excluded.',
        'event_limit': limit,
        'events': len(rows),
        'outcomes': dict(Counter(r['outcome'] for r in rows)),
        'latency_samples': len(timings),
        'p50_latency_ms': percentile(timings, 0.5),
        'p95_latency_ms': percentile(timings, 0.95),
        'mean_known_api_cost_usd': sum(costs) / len(costs) if costs else None,
        'unknown_cost_events': unknown_cost,
        'stage_latency_ms': {
            key: {
                'samples': len(values),
                'p50': percentile(values, 0.5),
                'p95': percentile(values, 0.95),
            }
            for key, values in stages.items()
        },
        'quality_note': 'Precision/recall require reviewed offline labels, not runtime confidence.',
    }
