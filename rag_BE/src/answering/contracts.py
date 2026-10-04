from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from rag_backend.retrieval.contracts import SearchHit, SearchRequest

Task = Literal['auto', 'answer', 'search', 'summarize', 'compare', 'generate_section']


class AnswerRequest(SearchRequest):
    limit: int = Field(default=5, ge=1, le=5)
    task: Task = 'auto'
    conversation_context: str = Field(default='', max_length=1000)


class Plan(BaseModel):
    task: str
    information_need: str
    allowed_tools: list[str]
    sequence: list[str]
    evidence_requirements: list[str]
    stop_criteria: list[str]
    max_tool_calls: int = 6
    max_generation_calls: int = 2
    max_claims: int = 6


class Claim(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, max_length=600)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)


class Draft(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['answer', 'abstain', 'clarify']
    claims: list[Claim] = Field(max_length=6)


class Evidence(BaseModel):
    id: str
    hit: SearchHit


class AnswerResponse(BaseModel):
    request_id: UUID
    status: Literal['answer', 'extracts', 'search', 'clarify', 'abstain']
    text: str
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    plan: Plan
    trace: dict
