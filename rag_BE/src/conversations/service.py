from copy import deepcopy
from typing import Literal

from rag_backend.answering.contracts import AnswerRequest


class TurnRequest(AnswerRequest):
    # Stored history is selected server-side; clients cannot inject a history transcript.
    conversation_context: Literal[''] = ''


def snapshot(response):
    payload = response.model_dump(mode='json')
    # Keep exact citation references, not a second copy of entire original passages.
    for evidence in payload['evidence']:
        evidence['hit'].pop('text', None)
        for citation in evidence['hit']['citations']:
            citation.pop('excerpt', None)
    return payload


class ConversationService:
    def __init__(self, repository, answering):
        self.repository, self.answering = repository, answering

    def annotate_many(self, payloads):
        values = deepcopy(payloads)
        ids = {e['hit']['chunk_id'] for p in values if p for e in p['evidence']}
        available = self.repository.availability(ids)
        for payload in values:
            if payload:
                for evidence in payload['evidence']:
                    evidence['available'] = evidence['hit']['chunk_id'] in available
                    evidence['availability_reason'] = (
                        'current_publication'
                        if evidence['available']
                        else 'deleted_replaced_or_unavailable'
                    )
        return values

    def turn(self, conversation_id, key, request):
        turn_id, existing, context = self.repository.reserve(conversation_id, key, request)
        if existing is not None:
            return {
                'turn_id': turn_id,
                'replayed': True,
                'response': self.annotate_many([existing])[0],
            }
        try:
            answer_request = AnswerRequest.model_validate(
                {**request.model_dump(), 'conversation_context': context}
            )
            response = self.answering.answer(answer_request)
            payload = snapshot(response)
            self.repository.complete(conversation_id, turn_id, payload)
        except Exception:
            self.repository.fail(conversation_id, turn_id)
            raise
        return {'turn_id': turn_id, 'replayed': False, 'response': self.annotate_many([payload])[0]}

    def history(self, conversation_id):
        turns = self.repository.history(conversation_id)
        annotated = self.annotate_many([turn['response'] for turn in turns])
        for turn, response in zip(turns, annotated, strict=True):
            turn['response'] = response
        return {'conversation_id': conversation_id, 'turns': turns}
