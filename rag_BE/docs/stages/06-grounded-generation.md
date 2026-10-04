# Stage 6: grounded answering and bounded reasoning

## What changed, and why

Stage 5 returned evidence candidates. Stage 6 adds POST /answers: a fixed planner,
context assembly, replaceable generation adapter and a response-validation gate.
It never streams an unvalidated draft. No database migration is needed.
The service remains private, single-user and loopback-only: source filters are
NOT authorization. Login, workspace administration and the frontend are not added.

| Stage | Implementation | Reason |
| --- | --- | --- |
| Plan/router | Deterministic task selection: answer, search, summarize, compare, generate_section, clarify or abstain. | Closed workflows are inspectable and cannot invent tools or actions. |
| Tools | Allowlisted retrieval, original-evidence hydration and citation validation. | No shell, web browsing, writes, code execution or external actions. |
| Retrieve | Reuse Stage 5 FTS + pgvector + RRF, deduplication and local reranker. | One retrieval implementation and matching embedding profiles. |
| Select context | At most five complete original chunks, within a 6,000-byte serialized-message budget. | Never silently truncate a cited chunk or send raw search candidates. |
| Recheck | Rehydrate the same publication/profile/source scope before generation and before return. | A replaced or deleted source must not support a newly returned answer. |
| Generate | Local Qwen3 1.7B through Ollama by default; JSON schema, no thinking/tool execution. | Small local starting point, zero inference API charge, replaceable provider contract. |
| Validate | Schema/IDs, exact numeric values/backtick symbols, relevance, local NLI support/conflict. | A fluent draft is not enough to render an answer. |
| Repair/fallback | Repair citation IDs/schema once using the same evidence; otherwise clarify, abstain or labeled extracts. | No repeated attempts to persuade a validator to accept unsupported facts. |
| Trace | Decision codes, tools/timings, profiles and token usage when available. | Debug decisions without logging query text, evidence, drafts or hidden reasoning. |

Planner output is a structured task plan, NOT chain-of-thought. It reports information
need, allowed tools, sequence, evidence requirements, stop criteria and budgets.
The HTTP response contains that user-facing plan; the logger does not log its query.
Fixed heuristics detect a small set of ambiguous, high-risk and injection requests.
These are conservative checks, not comprehensive security or domain-risk classifiers.

## API and task boundaries

POST /answers accepts query, optional source_ids, limit (1-5, default 5), task
(default auto), and optional conversation_context (at most 1,000 characters).
Query/source validation is shared with /search. Extra fields are rejected.

~~~json
{"query":"What are Saturn rings made of?","task":"answer","limit":5}
~~~

- answer: selected passages and generated claims need a nonnegative local relevance
  score before claim-support validation.
- search: returns labeled original extracts, without generation.
- summarize / generate_section: require explicit nonempty source_ids; outputs
  explicitly cover retrieved excerpts only, not a complete document.
- compare: requires exactly two distinct selected sources and cited evidence from
  both. Statements are source-backed, not unsupported rankings or calculations.
- clarify / abstain: internal router outcomes, not arbitrary model tools.
- Omitted/null source_ids searches published sources; [] means no source scope.
- Conversation context is untrusted interpretation context, never evidence.
  Persistent history and reliable conversational query rewriting remain Stage 7.

Response includes request_id, status, text, claims, evidence, plan and trace.
Status is answer, search, extracts, clarify or abstain. Only validated claims produce
answer status. Each claim lists E1-style evidence IDs; the evidence list includes
original source/version/chunk/block IDs, locators and excerpts. Clients must render
text safely rather than interpreting source/model text as HTML.

## Context and response validation

Context contains the question, selected original evidence and grounding/citation
instructions. It excludes whole retrieval result lists and internal storage paths.
Recognized secrets or instruction-injection patterns cause chunks to be omitted.
This regex screening is defense-in-depth, NOT a comprehensive secret/PII scanner;
private data must not be uploaded to this unauthenticated demo by untrusted users.

The strict draft schema permits only status and up to six atomic claims. Each claim
has up to 600 characters and one to three evidence IDs. Unknown IDs/invalid format
get one repair. Unsupported numbers or code symbols, unsafe claims, missing
comparison evidence, irrelevant claims or uncertain support do not get repaired.

Local NLI uses cross-encoder/nli-MiniLM2-L6-H768, pinned revision
b95119ce93d3e065de6214e38cd4a97b0f2f2c6d, checksum-verified ONNX and tokenizer files.
The upstream label order is contradiction, entailment, neutral. CPU inference is
offline, two threads, batches of four, at most 30 premise/claim pairs and a
45-second subprocess timeout. Full pairs exceeding 512 tokens are rejected;
the end of evidence is never silently discarded.

Each cited passage must support its claim with entailment score at least 0.90.
Contradiction score at least 0.90 in any selected passage rejects the response.
Scores are model outputs, NOT calibrated probabilities or proof. English quality,
relevance threshold zero and NLI thresholds require gold-set calibration in Stage 7.
This first slice deliberately does not combine multiple passages into inferred
multi-hop facts. Some correct complex answers will therefore abstain.

Outcomes:

- Pass: render only validated claims and their original citations.
- Correctable schema/citation issue: regenerate once, then run the whole gate again.
- Unsupported, conflicting, unsafe or weak evidence: fixed clarification/abstention.
- Generator/NLI outage or NLI input limit: labeled original extracts, no generated
  claims. Source/publication checks still run. If those checks fail, abstain.
- Retrieval/DB failure or admission/tool budget exceeded: sanitized HTTP 503.
- Optional high-risk LLM judge is NOT enabled; uncertain cases safely abstain.

No validation step can prove absence of all hallucinations, secret leaks, hidden
contradictions or task omissions. Conflict checks cover selected evidence only.
Whole-document summarization, arithmetic, complete comparison coverage and
multilingual support are not established by the current smoke tests.

## Providers, budgets and zero-cost default

Local generation uses the installed model only: no inference-time download, no
cloud-tag/remote model, no proxy/redirect or automatic provider fallback. The
adapter checks completion capability, records the digest, rejects a changed digest,
disables thinking where supported and accepts only completed text responses.
Default: qwen3:1.7b, temperature 0, 8,192-token context configuration, 512 output
tokens. The conservative message byte cap leaves room for schema/template overhead;
it is not a universal tokenizer guarantee for future adapters/models.

One concurrent answer per API process; six tool calls and at most two generation
attempts. A 240-second generation-start budget and 300-second between-tool budget
are checked between stages, NOT a hard wall-clock cancellation. Each local HTTP
operation has its own configured timeout (default 180 seconds); NLI/reranker
subprocesses have hard timeouts. Multiple API processes multiply concurrency.
There is no global RAM/CPU quota.

The optional OpenAI Responses adapter is disabled by default. It requires explicit
provider/model/key configuration, remote enablement, reviewed data policy,
kill-switch off and a positive call allowance. It uses strict structured output,
no tools, no retry and requests store=false. This is not a claim of zero provider
retention: review the selected provider/model's policy before sending any data.

Keys are server-side SecretStr settings only. No key is sent to a frontend or
included in trace/error messages. Provider error bodies and hidden reasoning are
not returned. Remote calls send selected evidence and conversation context, so
explicit opt-in is required. Local NLI remains the validator even with remote
generation.

The remote cap is per API process, counts failed attempts and resets on restart;
it is NOT an account-wide, persistent or monetary spending limit. Output is bounded
to 512 tokens but input and model prices vary. Keep remote disabled for the zero-cost
MVP. Persistent cost accounting and budget enforcement precede any public/shared
paid deployment. No remote inference call is part of automated tests.

## Setup

From rag_BE, with Ollama and the existing database available:

~~~powershell
ollama pull qwen3:1.7b
.\.venv\Scripts\python.exe -m rag_backend.answering.provision
.\.venv\Scripts\python.exe -m uvicorn rag_backend.main:create_app --factory --host 127.0.0.1 --port 8000
~~~

Explicit model provisioning downloads public artifacts only, never project documents.
The NLI files live in ignored .local-data/models/nli-minilm2. Generation weights
are held in Ollama's model cache. Inference has no automatic download fallback.

~~~powershell
$body = @{ query = 'What are Saturn rings made of?'; task = 'answer' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/answers -ContentType application/json -Body $body
~~~

Sources must first complete verification, preprocessing and indexing. /health/ready
still checks database/schema infrastructure, not model readiness. Missing local
generation/NLI can cause labeled-extract fallback; missing embeddings causes 503.

## Tests and next stage

Final regression run (2026-10-04): all 191 tests passed with database and real-model
tests enabled, in 84.64 seconds. Ruff lint and formatting passed; pip reported no
broken requirements. One existing upstream Starlette test-client deprecation
warning remains. The API/worker were not left running as background services.

Live verification on 2026-10-04: the four answering integration tests passed,
including actual Qwen generation plus local NLI acceptance of the cited synthetic
upload (84.10 seconds for the targeted run). This test requires answer status;
an extracts/abstention fallback does not count as success. Both models were
provisioned locally. Optional remote inference was tested only with mock transport;
no paid inference calls or project-document transfer were made.

Unit tests cover grounding, repair-once, unsupported values, unknown citations,
injection screening, NLI failure/conflicts, source changes, bounded context/tools,
local provider contracts and mocked opt-in remote gates.
PostgreSQL integration covers upload through publication to cited answer and deletion
during generation. Opt-in real tests require RAG_RUN_DB_TESTS=1 and
RAG_RUN_MODEL_TESTS=1; they check NLI labels/token limits and actual local
upload/retrieve/generate/validate success, not just fallback success.

This is functional testing, not a gold-set accuracy benchmark. Stage 7 remains
persistent conversations, retention/deletion semantics, durable telemetry/audit,
offline reviewed gold/adversarial sets and aggregate evaluation. No frontend,
authentication, broad multi-agent tools, multimodal enrichment or flowchart comparison
was introduced here.

Primary references:
[Ollama chat API](https://docs.ollama.com/api/chat),
[Qwen3 local model](https://ollama.com/library/qwen3:1.7b),
[NLI model](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768),
[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
