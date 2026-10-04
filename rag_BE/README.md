# RAG backend

This is the backend in `C:\Projects\prod_rag\rag_BE`. The frontend lives in
`../rag_FE`. We build and explain the backend in stages.

## Current stage: 7 — conversation history and evaluation foundation

Upload, verification, text restructuring/chunking, and transactional indexing are implemented.
See [Stage 2: storage and jobs](docs/stages/02-storage-jobs.md),
[Stage 3: parsing and chunking](docs/stages/03-restructuring-chunking.md), and
[Stage 4: indexing, publication, setup and limitations](docs/stages/04-indexing.md).
Run `python -m rag_backend.worker --once` to process one job, or omit `--once` to keep polling.
A normal document requires verification, preprocessing, then indexing jobs.
Indexing requires the configured local Ollama model. It never downloads a model automatically.
POST /search now returns ranked original evidence and citation locations.
See [Stage 5: retrieval decisions, API and limits](docs/stages/05-retrieval.md).
POST /answers adds bounded planning, grounded generation and response validation.
See [Stage 6: answering, safety boundaries and setup](docs/stages/06-grounded-generation.md).
Search results alone remain evidence candidates, not answers.
Conversation endpoints now persist validated responses with idempotent replay and
current citation availability. GET /metrics/answers reports content-free runtime metrics.
See [Stage 7: history, retention, evaluation and remaining review](docs/stages/07-conversations-evaluation.md).
The offline evaluator is implemented; a representative human-reviewed accuracy benchmark
is still pending. Synthetic templates are not accuracy results.

## Stage 1: application and database foundation

Implemented:

- An installable Python package under `src/rag_backend`.
- A FastAPI application factory and separate health router.
- Validated settings loaded from this directory's `.env` and environment variables.
- PostgreSQL connections with bounded connection, pool and statement timeouts.
- A Compose service for PostgreSQL 17 with pgvector, bound to loopback.
- An explicit Alembic migration that enables pgvector.
- Unit tests and an opt-in, read-only database integration check.

Why these choices:

| Choice | Reason |
| --- | --- |
| One package, separate modules | The API and later worker share the same business logic. |
| src layout | Tests exercise the installed package, reducing accidental imports from the working directory. |
| Application factory | Tests can supply settings without changing process-wide state. |
| PostgreSQL plus pgvector | Original evidence, versions, jobs, full-text search and vectors can share transactions. |
| Explicit migrations | Starting the API does not silently change the database. |
| Separate liveness/readiness | A running process can report a missing database or schema accurately. |
| Local file storage | Original PDFs remain intact while PostgreSQL stores extracted evidence and references. |

Stage 1 introduced the foundation; subsequent stages add the ingestion/indexing worker.
Readiness currently describes database
infrastructure only; it does not claim that a RAG pipeline or model is ready.

## Local setup (PowerShell)

Prerequisites: Python 3.12 or 3.13 and Docker with Linux containers.
Run Python commands below from `rag_BE`.

```powershell
cd C:\Projects\prod_rag\rag_BE
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements-dev.lock -e '.[dev]'
Copy-Item .env.example .env
```

The constraints file records the dependency versions tested on Windows/Python 3.12.
Revalidate dependencies when changing Python or operating systems.
Skip environment creation when `.venv` already exists. Set a local database
password in `.env` before starting PostgreSQL. Keep `.env` private.
Environment variables take precedence over `.env`.

From the project root:

```powershell
cd C:\Projects\prod_rag
docker compose --env-file rag_BE/.env up -d postgres
```

Then migrate and start the API:

```powershell
cd C:\Projects\prod_rag\rag_BE
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn rag_backend.main:create_app --factory --host 127.0.0.1 --port 8000
```

- `http://127.0.0.1:8000/docs`: API contracts.
- `GET /health/live`: HTTP 200 when the process responds.
- `GET /health/ready`: HTTP 200 only when the database, vector extension and
  expected migration revision are present; otherwise HTTP 503 with boolean checks.

This first release is for local/private use without authentication.
Keep the API bound to loopback. Relative `RAG_DATA_DIR` paths are resolved from
`rag_BE`, defaulting to `../.local-data`; uploaded files will never be committed.
Settings currently target an editable local checkout. A future wheel deployment
must make configuration and data paths explicit.

The host database port defaults to 55432 because another local service already
uses 5432. PostgreSQL keeps port 5432 inside its container.
The Docker database is persistent in a named volume. Ordinary
`docker compose --env-file rag_BE/.env down` preserves it.
Changing the environment password after initialization does not change an
existing PostgreSQL user's password. The `pg17` image tag is mutable; image
digest pinning belongs to the later reproducible deployment stage.

## Checks

```powershell
cd C:\Projects\prod_rag\rag_BE
.\.venv\Scripts\python.exe -m pytest -m 'not integration'
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

After starting and migrating the local database:

```powershell
$env:RAG_RUN_DB_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest -m integration
Remove-Item Env:RAG_RUN_DB_TESTS
```

The integration check verifies the real schema and a vector-distance operation.
Unit tests cover invalid configuration, secret masking, database outages,
missing migrations/extensions, and API status codes. Tests use synthetic data.

## Accepted design and build sequence

### Stage 1 verification (2026-10-01)

- Created a project-local `.venv` using the available bundled Python 3.12.14.
  The Windows Python launcher had no registered runtime.
- Created an ignored `.env` with a generated local database password.
- Started `prod-rag-postgres-1` on `127.0.0.1:55432` and applied the first migration.
- Passed all 20 tests, including the real pgvector operation and API readiness
  against PostgreSQL. Ruff checks passed; pip reported no broken requirements.
- One upstream Starlette warning recommends its newer HTTP test client.
  The current pinned test client passed the suite.

The database is running. The API was exercised through its test client and is
not left running as a background server. Use the Uvicorn command above to start it.

1. **Foundation:** package, settings, API lifecycle, database configuration.
2. **Storage and jobs:** sources, immutable versions, original files,
   PostgreSQL job claiming, heartbeat, bounded retries and crash recovery.
3. **Ingestion:** searchable PDF/TXT/Markdown parsing, normalization, extraction
   quality, structure-aware chunks with exact source locations.
4. **Indexing:** local embeddings, PostgreSQL FTS, atomic publication of complete
   versions. Failed updates leave the previous completed version searchable.
5. **Retrieval:** optional source filters, FTS plus pgvector, RRF, deduplication,
   local cross-encoder reranking, diverse original evidence.
6. **Answering:** bounded planner/router, approved read-only tools, Ollama by
   default, optional OpenAI adapter, provider-neutral contracts, context assembly,
   citations and local NLI. Repair once or return extracts/clarification/abstention.
7. **Conversations and evaluation:** persistent history, deletion semantics,
   per-stage telemetry, unit/integration/end-to-end checks and offline gold sets.

The first milestone detects scanned PDFs and reports that OCR is not yet
supported. Text uses original evidence directly. Mixed-content enrichment,
OCR, image/table/code/spreadsheet handling and their validators follow later.

Chat searches all published sources unless filtered. Generation uses selected
original evidence; history helps interpret questions but does not support factual
claims by itself. Answers are shown only after validation, with no draft streaming.
After source deletion, historical messages remain, their citations are marked
unavailable, and deleted evidence cannot support future answers.

## Sources behind the foundation

- [FastAPI modular applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [Pydantic settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Alembic migrations](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
