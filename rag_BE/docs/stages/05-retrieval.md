# Stage 5: hybrid retrieval and original evidence

This stage finds and ranks evidence. It does not generate answers or establish that
every result supports a claim. The frontend and authentication remain deferred;
run this local single-user API on loopback only. Source filters are not authorization.

## Process and reasons

| Step | What happens | Why |
| --- | --- | --- |
| Normalize | NFC Unicode and whitespace cleanup; bounded query and source IDs. | Consistent input without LLM query rewriting or changing negation/numbers. |
| Query embedding | Use the installed EmbeddingGemma digest and retrieval-query prompt. | Match the embedding space used for document indexing. |
| Hybrid search | Top 40 FTS results plus top 40 exact pgvector cosine results. | Keywords preserve exact terms; vectors find semantic matches. |
| RRF | Add 1/(60 + rank) from each list; deterministic UUID tie-break. | Raw lexical and vector scores are not directly comparable. |
| Deduplicate | Collapse whitespace-normalized exact duplicates; conservative near-duplicates within a version only when original content spans are covered and numbers unchanged. | Avoid repeated evidence without broadly merging contradictory statements. |
| Rerank | Local cross-encoder scores the top 20 question/passage pairs. | Assess relevance jointly instead of relying only on independent embeddings. |
| Hydrate | Read canonical text, headings and block/line/page locators from PostgreSQL again. | Search representations are not authoritative citation evidence. |
| Diversify | Prefer at most two hits per source, then fill remaining slots if needed. | Give multiple sources a chance without inventing extra matches. |

Each database search branch filters the published, nondeleted, verified, approved,
indexed versions and the exact embedding profile BEFORE top-k selection. Optional
source IDs constrain both branches. Hydration uses a new SQL snapshot to re-check
publication and deletion, so a source deleted/replaced during reranking is dropped.
Results describe the hydration snapshot; a later generation request must re-check
again. Arbitrary future revocations cannot erase evidence already delivered to a client.
No-match results may mean no compatible published profile, not an empty database.

Canonical span references are verified against original blocks from the same version;
invalid/missing references cause the hit to be omitted. The response includes normalized
chunk text plus exact source locators and excerpts. PDF locators are page/geometry,
not byte offsets; text/Markdown locations are original line ranges.

## Reranker and operational boundaries

Model: cross-encoder/ms-marco-MiniLM-L6-v2, pinned revision
233902d25c440f23af6f7d6e94d2946bac0bee0a. The original FP32 ONNX artifact is about
91 MB. It and tokenizer.json are verified against pinned upstream SHA-256/Git blob
identities before loading. Only local ONNX CPU inference and local tokenizer files
are used. No remote model code, PyTorch pickle or inference API is loaded.

This reranker is English-focused; multilingual ranking quality is not established.
It is a replaceable adapter, not a promise that this model is best for every domain.
Its raw logits are relevance scores, NOT calibrated confidence/support probabilities.

Reranking runs in a fresh subprocess per request with a 30-second hard timeout, two
CPU inference threads, batches of eight, and no automatic download. Long passages
use overlapping 512-token pair windows with 32-token overlap and max-window scoring;
at most four windows per passage and 128 query tokens. If limits are exceeded, we
explicitly fall back to RRF for the whole candidate list instead of silently dropping
the end of a passage. A fresh process trades model startup cost for bounded lifetime.

At most two concurrent searches per API process. Additional requests get HTTP 503.
The 120-second overall budget is checked between stages, not a hard cancellation of
an in-flight network/SQL call. HTTP requests have their own configured 60-second
timeout; SQL has the existing statement timeout. Multi-process deployment multiplies
the concurrency allowance. This MVP has no hard RAM sandbox or public rate limiter.

Embedding failure -> sanitized HTTP 503 (no silent lexical-only mode).
Reranker missing/corrupt/unavailable/timed out -> original RRF order plus explicit
warnings and null rerank scores. Publication/evidence checks still run. No remote
fallback or paid API is introduced. Electricity/hardware are not counted as API cost.

## API

POST /search with JSON:

~~~json
{"query": "What are Saturn rings made of?", "limit": 5}
~~~

Optional source_ids is a list of at most 50 source UUIDs. Omitted/null means all
published sources; [] means no sources. limit is 1-10 (default 5). Query is at most
1,000 characters and must contain searchable text. Extra request fields are rejected.

Response: request_id, status (ok/no_results), hits, warnings and trace. Each hit has
source/version/chunk IDs, filename, version number, original chunk text, headings,
citation block IDs/locators/spans/excerpts, RRF score and optional rerank score.
Trace includes model profiles, candidate/returned counts, timings by stage, total
latency, ranking mode, zero API cost and cache_hit=false. There is no retrieval cache.

No query text, evidence text, filename or model prompt is added to retrieval log
events. The Python logger emits structured extra fields; a durable audit sink and
dashboard belong to the later observability stage. Precision/recall are not computed
for individual requests. Scores do not implement answer sufficiency or abstention;
those checks belong to Stage 6.

## Setup and verification

Verified on 2026-10-03: all 145 tests passed with database and real-model tests
enabled (41.46 seconds in the final run). Ruff lint/format checks passed and pip
reported no broken requirements. One existing upstream Starlette test-client
deprecation warning remains. No API or worker was left running in the background.

The first long-passage test exposed tail loss in tokenizer overflow handling.
Window construction now uses the complete untruncated token IDs with explicit
BERT pair layout. A unit test proves token coverage/overlap, and the live model
test verifies the relevant tail changes ranking. The reranker profile is versioned
as window-max-v2 to identify this behavior.

From rag_BE, using the existing environment and database:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -c requirements-dev.lock -e '.[dev]'
.\.venv\Scripts\python.exe -m rag_backend.retrieval.provision
.\.venv\Scripts\python.exe -m uvicorn rag_backend.main:create_app --factory --host 127.0.0.1 --port 8000
~~~

Provisioning is an explicit public model-file download; it sends no documents or
queries. Files live under .local-data/models/minilm-l6-v2 (ignored by Git). Restart
the API after changing application code. No new database migration is needed.
Existing Stage 4 published records are searchable with their matching model digest.

PowerShell request (after uploading and indexing sources):

~~~powershell
$searchBody = @{ query = 'What are Saturn rings made of?'; limit = 5 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -ContentType application/json -Body $searchBody
~~~

For the full suite including real models:

~~~powershell
$env:RAG_RUN_DB_TESTS = '1'
$env:RAG_RUN_MODEL_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest
Remove-Item Env:RAG_RUN_MODEL_TESTS
Remove-Item Env:RAG_RUN_DB_TESTS
~~~

Tests use temporary synthetic originals and isolated PostgreSQL schemas, not user
documents. Real tests cover local embedding, publication, hybrid search, cross-encoder
ordering, citation hydration and evidence beyond the first reranker token window.
This is functional verification, not a calibrated gold-set accuracy result.

Next: Stage 6 bounded reasoning/router, context assembly, grounded generation and
response validation. No flowchart comparison was performed, as requested.

Primary references: [PostgreSQL text search functions](https://www.postgresql.org/docs/current/functions-textsearch.html),
[cross-encoder model](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
[ONNX Runtime](https://onnxruntime.ai/docs/api/python/api_summary.html),
[Tokenizers](https://huggingface.co/docs/tokenizers/api/tokenizer).
