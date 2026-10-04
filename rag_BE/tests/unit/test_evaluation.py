import pytest
from pydantic import ValidationError

from rag_backend.evaluation.report import GoldCase, Observation, evaluate, percentile


def case(id='one', expected='answer', reviewed=False):
    return GoldCase(
        id=id,
        dataset_version='synthetic-v1',
        query='Synthetic question',
        task='answer',
        modality='text',
        reviewed=reviewed,
        relevant_evidence_ids=['a', 'b'],
        expected_outcome=expected,
    )


def observation(id='one', **values):
    data = dict(
        id=id,
        dataset_version='synthetic-v1',
        retrieved_evidence_ids=['a', 'a', 'x'],
        outcome='answer',
        latency_ms=100,
        api_cost_usd=0,
        total_claims=2,
        supported_claims=1,
        total_citations=2,
        supported_citations=1,
        task_success=False,
    )
    data.update(values)
    return Observation(**data)


def test_precision_recall_deduplicate_and_use_distinct_denominators():
    report = evaluate([case()], [observation()], k=5)
    assert report['metrics']['retrieval_precision_at_k'] == 1 / 5
    assert report['metrics']['retrieval_recall_at_k'] == 1 / 2
    assert report['metrics']['grounded_claim_precision'] == 1 / 2
    assert report['slices']['modality']['text']['cases'] == 1
    assert 'human_review_incomplete' in report['release_gate']['failures']


def test_no_evidence_denominators_are_null_not_perfect_accuracy():
    gold = case(expected='abstain')
    gold.relevant_evidence_ids = []
    report = evaluate(
        [gold],
        [
            observation(
                outcome='abstain',
                total_claims=0,
                supported_claims=0,
                total_citations=0,
                supported_citations=0,
            )
        ],
    )
    assert report['metrics']['retrieval_recall_at_k'] is None
    assert report['metrics']['grounded_claim_precision'] is None
    assert report['metrics']['expected_abstention_or_clarification_correctness'] == 1
    assert not report['release_gate']['pass']


@pytest.mark.parametrize('change', [{'id': 'missing'}, {'dataset_version': 'wrong'}])
def test_case_alignment_and_version_mismatch_rejected(change):
    with pytest.raises(ValueError):
        evaluate([case()], [observation(**change)])


def test_duplicate_and_nonfinite_metrics_rejected():
    with pytest.raises(ValueError):
        evaluate([case(), case()], [observation()])
    with pytest.raises(ValidationError):
        observation(latency_ms=float('nan'))
    with pytest.raises(ValidationError):
        observation(supported_claims=3)
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([10, 20, 30], 0.95) == pytest.approx(29)


def test_release_gate_requires_review_safety_cost_and_measured_outcomes():
    gold, results = [], []
    for i in range(30):
        outcome = 'abstain' if i > 24 else 'answer'
        gold.append(case(str(i), outcome, True))
        if i > 24:
            gold[-1].category = 'adversarial'
        results.append(
            observation(
                str(i),
                outcome=outcome,
                reviewed=True,
                retrieved_evidence_ids=['a', 'b'],
                supported_claims=2,
                supported_citations=2,
                task_success=True,
            )
        )
    assert evaluate(gold, results)['release_gate']['pass']
    results[0].unsafe_output = True
    results[1].api_cost_usd = None
    failures = evaluate(gold, results)['release_gate']['failures']
    assert 'unsafe_output' in failures and 'zero_api_cost_required' in failures
