# Stage 2: sources, original files, and background jobs

Historical stage record. Parsing is now described in [Stage 3](03-restructuring-chunking.md)
and indexing/publication in [Stage 4](04-indexing.md).

## What works now

Upload a PDF/TXT/Markdown candidate, receive source/version/job IDs, poll its job,
and run the worker to verify that the stored original matches the uploaded bytes.
Upload a new version explicitly under the same source. List sources/versions,
retry eligible failed jobs, or delete a source logically.

A version marked **verified** has passed size/checksum verification only.
Content parsing, scanned-PDF detection, chunking, embedding and searchable
publication arrive in Stages 3 and 4. Extension checks do not establish file validity.

## Why the records are separate

| Record | Purpose |
| --- | --- |
| Source | Stable document identity and deletion state. |
| Version | Immutable file reference, original name, SHA-256, size and version number; processing state can change. |
| Job | Status, attempt count, lease and sanitized error code. |

A new version keeps the previous original. Re-upload by source ID; matching
filenames alone does not merge documents.

## Upload flow

1. Check filename, supported extension and declared size.
2. Stream the raw request into a uniquely named staging file, enforcing the byte
   limit while reading. Flush and sync it, and calculate SHA-256.
3. Begin a PostgreSQL transaction. Serialize identical idempotency keys; lock an
   existing source while assigning its next version number.
4. Insert the source/version and queued verification job.
5. Atomically rename the staged file to its generated original-file path.
6. Commit the transaction and return HTTP 202.

The job cannot be claimed until commit. A file-move failure rolls back the database.
File storage and PostgreSQL do not form one distributed transaction: a crash or
uncertain commit after rename can leave an orphan original. We retain it rather
than risk removing a committed original. Automatic orphan reconciliation and
physical purging are later work. Ordinary failed requests clean up staging files;
abrupt process termination can leave staging files. Do not manually remove
originals referenced by versions.

The client sends a UUID **Idempotency-Key**. Repeating the same key, filename,
content and target source returns the existing version/job. Reusing it for a
different request returns HTTP 409. A deliberate new version uses a new key.

## Worker and failure behavior

- PostgreSQL row locking with SKIP LOCKED lets workers claim different jobs.
- Each claim increments the attempt count and issues a fresh lease token.
- Heartbeats renew the lease during file verification.
- Expired work can be reclaimed; the old token cannot complete, fail or renew it.
- Transient storage failures retry with exponential backoff, capped at 60 seconds.
- Missing/corrupted originals fail with a specific error code.
- Manual retry preserves attempts; the default total limit is three.
- A final expired attempt becomes failed. Exhausted jobs require a new upload/version.
- Ctrl+C stops the worker; unfinished leases are recoverable on restart.

This is at-least-once execution with guarded state changes. Future parsing and
indexing handlers must make writes idempotent. There is no exactly-once promise
for future external model calls.

See the [PostgreSQL locking documentation](https://www.postgresql.org/docs/current/sql-select.html)
for the SKIP LOCKED behavior used by the queue.

## API contract

| Method and path | Behavior |
| --- | --- |
| POST /sources?filename=notes.txt | Create source, version and queued job. |
| POST /sources/{id}/versions?filename=notes.txt | Add a version to an active source. |
| GET /sources?limit=50&offset=0 | List active sources; limit is at most 100. |
| GET /sources/{id} | Read an active source. |
| GET /sources/{id}/versions | List versions newest first; supports limit/offset. |
| DELETE /sources/{id} | Hide source and cancel unfinished jobs. |
| GET /jobs/{id} | Read job status and safe error code. |
| POST /jobs/{id}/retry | Retry a failed job if attempts remain. |

Uploads use **application/octet-stream**, with file bytes as the request body.
This provides a bounded stream without another multipart buffering layer.
The Streamlit client will send selected bytes with the filename as an encoded
query parameter and a generated request key.

Responses exclude storage paths, request fingerprints, lease tokens and raw
database exceptions. OpenAPI documents the binary upload body.

## Run it

The Stage 1 environment and database are already set up. From rag_BE:

~~~powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn rag_backend.main:create_app --factory --host 127.0.0.1 --port 8000
~~~

In another terminal, upload an existing file; replace the example file path:

~~~powershell
$uploadKey = [Guid]::NewGuid().ToString()
curl.exe -X POST 'http://127.0.0.1:8000/sources?filename=notes.txt' -H 'Content-Type: application/octet-stream' -H "Idempotency-Key: $uploadKey" --data-binary '@C:\path\notes.txt'
~~~

Run one job, or omit --once to keep polling:

~~~powershell
cd C:\Projects\prod_rag\rag_BE
.\.venv\Scripts\python.exe -m rag_backend.worker --once
~~~

Use the returned job_id with GET /jobs/{id}. A succeeded job currently means the
original was verified, not that the document is indexed.

## Limits and deletion

Defaults: 25 MiB per upload, three attempts, a 60-second lease, two-second idle
poll. All are configurable through the RAG_ settings in .env.example.
No model dependency or paid provider call is introduced in this stage.

DELETE is a **logical deletion**. It hides the source and its versions and
cancels pending/running/failed work, including fencing an active worker.
The response explicitly says original_files_retained=true. Files, version
records and completed job history remain stored. Physical retention cleanup and
conversation citation handling will arrive with those later features.

## Verification

On 2026-10-01, all 58 tests passed: 32 unit tests and 26 PostgreSQL integration
tests. Formatting, lint and dependency checks passed. The worker --once startup
check succeeded. One existing upstream test-client deprecation warning remains.
Migration 0002_sources_jobs is applied to the local project database.

Tests migrate a fresh generated PostgreSQL schema per integration test and remove
only that schema afterward. File operations use temporary test directories.

Covered: streamed size limits; interrupted uploads; path traversal; replay and
conflicting keys; concurrent uploads/version numbers; file-move rollback;
exclusive claims; heartbeat/lease takeover; retry exhaustion; integrity failure;
deletion during work; API contracts; internal-reference privacy.

PostgreSQL-specific queries live in the source repository and job service.
Additional table mapping modules will be added when useful; there is no parallel
ORM model to keep synchronized with the migrations yet.
