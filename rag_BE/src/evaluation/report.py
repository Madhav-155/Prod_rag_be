"""Offline scoring only: no model calls, downloads or external writes."""

import argparse
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GoldCase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str
    dataset_version: str
    query: str = Field(min_length=1, max_length=1000)
    task: str
    modality: str
    category: Literal['normal', 'adversarial'] = 'normal'
    role: Literal['local_single_user'] = 'local_single_user'
    reviewed: bool = False
    relevant_evidence_ids: list[str]
    expected_outcome: Literal['answer', 'search', 'extracts', 'clarify', 'abstain']


class Observation(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    id: str
    dataset_version: str
    retrieved_evidence_ids: list[str]
    outcome: Literal['answer', 'search', 'extracts', 'clarify', 'abstain']
    latency_ms: float = Field(ge=0)
    api_cost_usd: float | None = Field(default=None, ge=0)
    # These are reviewer labels, never the response validator's self-assessment.
    reviewed: bool = False
    total_claims: int = Field(ge=0)
    supported_claims: int = Field(ge=0)
    total_citations: int = Field(ge=0)
    supported_citations: int = Field(ge=0)
    task_success: bool
    unsafe_output: bool = False

    @model_validator(mode='after')
    def counts(self):
        if self.supported_claims > self.total_claims:
            raise ValueError('Supported claims exceed total')
        if self.supported_citations > self.total_citations:
            raise ValueError('Supported citations exceed total')
        return self


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(pairs, k):
    precision, recall, outcomes = [], [], []
    for gold, observation in pairs:
        known = set(gold.relevant_evidence_ids)
        retrieved = list(dict.fromkeys(observation.retrieved_evidence_ids))[:k]
        if known:
            hits = len(known.intersection(retrieved))
            precision.append(hits / k)
            recall.append(hits / len(known))
        outcomes.append(observation.outcome == gold.expected_outcome)
    observations = [row for _, row in pairs]
    costs = [row.api_cost_usd for row in observations if row.api_cost_usd is not None]
    abstentions = [
        row.outcome == gold.expected_outcome
        for gold, row in pairs
        if gold.expected_outcome in {'abstain', 'clarify'}
    ]
    return {
        'cases': len(pairs),
        'retrieval_precision_at_k': divide(sum(precision), len(precision)),
        'retrieval_recall_at_k': divide(sum(recall), len(recall)),
        'grounded_claim_precision': divide(
            sum(r.supported_claims for r in observations), sum(r.total_claims for r in observations)
        ),
        'citation_precision': divide(
            sum(r.supported_citations for r in observations),
            sum(r.total_citations for r in observations),
        ),
        'expected_abstention_or_clarification_correctness': divide(
            sum(abstentions), len(abstentions)
        ),
        'outcome_accuracy': divide(sum(outcomes), len(outcomes)),
        'task_success': divide(sum(r.task_success for r in observations), len(observations)),
        'unsafe_outputs': sum(r.unsafe_output for r in observations),
        'p50_latency_ms': percentile([r.latency_ms for r in observations], 0.5),
        'p95_latency_ms': percentile([r.latency_ms for r in observations], 0.95),
        'mean_api_cost_usd': divide(sum(costs), len(costs)),
        'unknown_cost_cases': len(observations) - len(costs),
    }


def evaluate(gold, observations, k=5):
    if not 1 <= k <= 10 or not gold:
        raise ValueError('Nonempty gold set and k between 1 and 10 required')
    expected = {row.id: row for row in gold}
    actual = {row.id: row for row in observations}
    if len(expected) != len(gold) or len(actual) != len(observations):
        raise ValueError('Duplicate case IDs')
    if set(expected) != set(actual):
        raise ValueError('Missing or unexpected observations')
    if len({row.dataset_version for row in gold}) != 1:
        raise ValueError('Gold set must use one dataset version')
    pairs = [(case, actual[case.id]) for case in gold]
    if any(case.dataset_version != result.dataset_version for case, result in pairs):
        raise ValueError('Dataset version mismatch')
    report = metrics(pairs, k)
    reviewed = all(case.reviewed and row.reviewed for case, row in pairs)
    # Starting local MVP gates, not a certification or production SLA.
    thresholds = {
        'retrieval_recall_at_k': 0.90,
        'grounded_claim_precision': 0.95,
        'citation_precision': 0.98,
        'task_success': 0.90,
        'expected_abstention_or_clarification_correctness': 0.95,
    }
    failures = [
        name for name, floor in thresholds.items() if report[name] is None or report[name] < floor
    ]
    if not reviewed:
        failures.append('human_review_incomplete')
    if len(pairs) < 30:
        failures.append('minimum_30_cases')
    if sum(case.category == 'adversarial' for case in gold) < 5:
        failures.append('minimum_5_adversarial_cases')
    if report['unsafe_outputs']:
        failures.append('unsafe_output')
    if report['unknown_cost_cases'] or report['mean_api_cost_usd'] != 0:
        failures.append('zero_api_cost_required')
    if report['p95_latency_ms'] > 120000:
        failures.append('local_p95_latency_budget')
    slices = {}
    for dimension in ('task', 'modality', 'role', 'category'):
        slices[dimension] = {
            value: metrics([(g, r) for g, r in pairs if getattr(g, dimension) == value], k)
            for value in sorted({getattr(g, dimension) for g, _ in pairs})
        }
    return {
        'dataset_version': gold[0].dataset_version,
        'k': k,
        'reviewed': reviewed,
        'metrics': report,
        'slices': slices,
        'release_gate': {'pass': not failures, 'failures': failures},
        'note': 'Offline labels only; no per-response precision/recall guarantee.',
    }


def read_rows(path, model):
    if path.stat().st_size > 10_000_000:
        raise ValueError('Evaluation file exceeds 10 MB')
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    try:
        result = evaluate(
            read_rows(args.gold, GoldCase), read_rows(args.observations, Observation), args.k
        )
    except (ValueError, OSError):
        raise SystemExit('Invalid evaluation input; check schemas, IDs and versions.') from None
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
