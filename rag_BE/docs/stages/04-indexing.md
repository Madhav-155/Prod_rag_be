# Stage 4: local embeddings, keyword index and safe publication

## What changes and why

1. Approved preprocessing queues an **index** job in the same transaction. Existing
   approved versions receive a job through migration 0004. Quarantine never queues indexing.
2. The worker verifies original-file integrity again and loads the complete chunk set.
3. A provider-neutral Embedder interface calls the local Ollama adapter in batches
   of eight. The default model is embeddinggemma:latest; it must already be installed.
   No auto-pull, cloud fallback, proxy use or redirects. Only literal loopback HTTP
   origins are allowed. Model selection is configuration, not application-wide code.
4. The adapter records the model digest and checks it around embedding calls.
   Count, dimension (768), finite/nonzero values and unit normalization are checked.
   Requests set truncate=false so oversized evidence fails rather than losing text.
   EmbeddingGemma documents receive `title: none | text: ` and queries receive
   `task: search result | query: `. This formatting is included in the profile hash
   as embeddinggemma-retrieval-v2. Stored page_content remains the original chunk
   text, without these task prefixes. Other model adapters must define their own
   input conventions; changing conventions requires re-embedding, not mixing vectors.
5. PostgreSQL receives one retrieval record per original chunk: page_content is
   canonical chunk text, with chunk/version/profile references and a pgvector vector.
   Original blocks and citation locations remain authoritative, not copied into metadata.
6. A generated simple-configuration tsvector plus GIN index provides keyword search.
   This avoids assuming English stemming; language-specific analyzers are future work.
7. Only after every chunk passes do records, indexed status, publication pointer and
   job success commit together. A failed update leaves the previous version published.

## Publication rules

The sources.published_version_id pointer chooses one version per source. Source
row locking serializes publication and deletion. A late older job may finish indexing
but cannot replace a newer published version. Fenced/expired leases cannot publish.
The published_retrieval_records view excludes deleted, unverified, unapproved and
unindexed versions. Later retrieval must use this view and the matching embedding
profile, then hydrate canonical evidence; it must not search unrestricted raw records.

Old vectors are retained for history/diagnostics, but are not exposed by the published
view. Deletion is still logical, not physical erasure. There is no partial lexical-only
publication when embeddings fail. Index failures do not overwrite extraction approval.

Vectors use exact pgvector similarity at MVP scale, not an approximate HNSW/IVFFlat
index. The GIN keyword index is physical; the vector column supports exact search.
This is an explicit accuracy-first scale tradeoff. Profile/model migration, dimensions
other than 768, multiple representations per chunk, embedding caches and approximate
vector indexes require a later migration/reindex design. Do not mix profiles in Stage 5.

## Setup and operation

Run from rag_BE with the existing environment/database:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -c requirements-dev.lock -e '.[dev]'
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m rag_backend.worker
~~~

Before processing index jobs, install/start Ollama and explicitly provision the chosen
local model (for example, ollama pull embeddinggemma). That is a separate model
download, not something this application performs. See .env.example for configuration.
The configured tag must match its name in Ollama's /api/tags response.

Use GET /jobs?version_id=... for all three jobs. Version responses now expose
indexing_status, embedding_profile and indexed_at; source responses expose
published_version_id. Indexed older versions need not be currently published.
Infrastructure /health/ready does not assert model availability.

Failures expose safe codes, not provider response text. Transport/429/server failures
use the existing bounded retry budget. Missing model, invalid dimensions or content
overflow fail without automatic retries; after correcting configuration, manually retry
if attempts remain. Exhausted attempts require a new uploaded version in this MVP.

Default per-HTTP-operation timeout is 60 seconds; indexing has a 300-second acceptance
deadline. The main worker renews leases while model calls run. Cancellation/timeout
prevents publication; an already-running HTTP call may finish under its transport
timeout and has no database-write authority. This is not a hard model-server kill or
memory sandbox. Large documents may need tuned local limits. No paid API is used.

## Verification and remaining scope

Final live verification (2026-10-03): all 116 tests passed with both database and
local-model tests enabled. Ruff lint/format checks passed. Effective configuration:
embeddinggemma:latest, 768 dimensions, 60-second HTTP timeout. Installed model digest:
`85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1`.
Stage 4 local-model setup and smoke verification are complete. This confirms the
pipeline works, not that retrieval accuracy has been established on a gold dataset.

Verified on 2026-10-03: 114 tests passed (72 unit, 42 integration), Ruff lint and
format checks passed, pip check found no broken requirements, and the project database
is at 0004_indexing. One existing upstream Starlette test-client deprecation warning
remains. The API and worker were not left running in the background.

Tests exercise real PostgreSQL FTS/vector operations and use synthetic vectors only
inside tests. They check publication, failed updates, retry, stale leases, deletion,
out-of-order completion, missing models, invalid vectors and local-only transport.
Synthetic embeddings are never an application fallback and do not establish retrieval
quality. The initial implementation lacked a local runtime. On 2026-10-03,
Ollama 0.35.1 and embeddinggemma:latest were installed and the real-model
end-to-end test passed (66.16 seconds including initial model loading).
Ollama verified the model download checksum. First model-server startup took
approximately 35 seconds on this machine; a parallel 15-second adapter check
timed out, so the normal request default was increased to 60 seconds. Explicit
environment overrides still take precedence. Lease heartbeats and the overall
indexing deadline remain enforced.

The opt-in real-model test is now available. It uses synthetic text documents and
the same isolated PostgreSQL schema/temporary-file fixtures as the integration suite:

~~~powershell
$env:RAG_RUN_DB_TESTS = '1'
$env:RAG_RUN_MODEL_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest tests/integration/test_local_model.py -v
Remove-Item Env:RAG_RUN_MODEL_TESTS
Remove-Item Env:RAG_RUN_DB_TESTS
~~~

The test indexes astronomy and cooking documents, embeds a Saturn question, and
checks that pgvector ranks the astronomy source first with 768-dimensional vectors.
This is an end-to-end smoke check, not a retrieval-accuracy benchmark. Following
the formatting update, 115 regression tests passed; the separate live test has
now also passed. The isolated test sources/schema are cleaned up by the fixtures;
the model remains installed locally. No real user documents were used in this test.

Stage 5 adds query embeddings, hybrid FTS/vector retrieval, RRF and reranking. Stage 6
adds grounded generation and response validation. OCR/mixed enrichment stay deferred.
No flowchart comparison or frontend/auth implementation was included in this stage.

Official references: [Ollama embed API](https://docs.ollama.com/api/embed),
[EmbeddingGemma retrieval prompts](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers),
[model digests](https://docs.ollama.com/api/tags), [pgvector](https://github.com/pgvector/pgvector),
[PostgreSQL FTS indexes](https://www.postgresql.org/docs/current/textsearch-tables.html).
