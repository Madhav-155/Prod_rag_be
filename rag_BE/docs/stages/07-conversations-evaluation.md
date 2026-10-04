# Stage 7: local conversation history and evaluation foundation

## What changed and why

| Change | Reason |
| --- | --- |
| PostgreSQL conversations and ordered turns | History survives API restarts and shares the existing transactional database. |
| Idempotency key on each turn | Retrying a successful send returns the saved answer without another model call. |
| One pending turn per conversation; 15-minute lease and completion fencing | Concurrent sends cannot mix ordering, and a crashed/late worker cannot overwrite a newer turn. |
| Atomic completed turn + content-free event | A saved answer and its telemetry commit together. |
| Compact citation snapshots | Save IDs, source/version/locators and the rendered response, not an extra copy of raw evidence fields. |
| Availability checked on every history read/replay | Deleted/replaced/unpublished evidence is visibly unavailable without rewriting historical answer text. |
| Last three completed user questions, bounded to 1,000 characters | Server-selected context assists interpretation; previous assistant claims never become evidence. |
| Separate runtime metrics and offline scoring | Runtime confidence is not ground truth or per-request precision/recall. |

The backend remains a private, single-user installation. Everyone able to reach
the API can access its history. There is no login, ownership isolation or public
deployment authorization. Keep it bound to 127.0.0.1. No frontend was changed.

Migration 0005_conversations adds conversations, conversation_turns and answer_events.
It does not remove sources, versions, original files, embeddings or retrieval indexes.
Health readiness now expects this migration. /answers remains stateless; clients use
the conversation endpoints when they want persistence.

## API

| Endpoint | Result |
| --- | --- |
| POST /conversations | Create a conversation; returns ID and creation/expiry timestamps. |
| GET /conversations?limit=50&offset=0 | Active conversations, newest first; max limit 100. |
| POST /conversations/{id}/turns | Answer and persist one turn. Required UUID Idempotency-Key header. |
| GET /conversations/{id} | Ordered turns, response snapshots, failures and citation availability. |
| DELETE /conversations/{id} | Permanently delete that conversation, its turns and associated events; idempotent 204. |
| GET /metrics/answers?limit=1000 | Bounded runtime summary of latest retained conversation events. |

A turn accepts the Stage 6 query/task/source_ids/limit fields. Nonempty client-provided
conversation_context is rejected: the server selects history. Response includes
turn_id, replayed and response. Unlike stateless /answers, persisted response.evidence
omits hit.text and citation.excerpt; source/version/chunk/block references, locators
and availability remain. Source extracts displayed as an answer are still part of
saved answer text; this is not a claim that all original words are excluded from history.

Minimal PowerShell example after starting the API:

~~~powershell
$conversation = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/conversations
$headers = @{ 'Idempotency-Key' = [guid]::NewGuid().ToString() }
$body = @{ query = 'What are Saturn rings made of?'; task = 'answer' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/conversations/$($conversation.id)/turns" -Headers $headers -ContentType application/json -Body $body
Invoke-RestMethod -Uri "http://127.0.0.1:8000/conversations/$($conversation.id)"
~~~

Retain the same key and identical body when retrying a successful/uncertain delivery.
Reusing a key with changed inputs returns 409. A pending/failed key also returns 409;
inspect history rather than automatically regenerating. Failed turns need a new key
for a deliberate retry. Other sends while the lease is live return 409.
After 15 minutes a new key can fence the abandoned turn, marking it lease_expired.
A pending row may remain pending until that next send: no recovery scheduler is claimed.
There is a 100-turn cap per conversation. Start a new conversation after that.

No database transaction is held during retrieval/model execution. Deleting the
conversation while generation runs prevents completion from recreating it.
An in-flight local model call is not forcibly canceled by conversation deletion.
Citation availability reflects its read snapshot; deletion cannot retract a response
already delivered. Historical response text remains historical, never new evidence.
A replaced version is unavailable under current-publication rules even if its binary
is retained for audit. The Stage 6 router still asks clarification for vague follow-ups;
automatic conversational query rewriting is not implemented.

## Retention, privacy and operational boundaries

Retention is 30 days from creation, not a sliding period from last use. Expired
conversations cannot be listed, read or appended to. Physical deletion is explicit:

~~~powershell
.\.venv\Scripts\python.exe -m rag_backend.conversations.retention --purge-expired
~~~

This permanently removes expired conversations and cascading turns/events only.
It does not delete originals or source records. It has not been scheduled automatically.
Manual conversation deletion is also permanent in the live database; recovery requires
a separately maintained database backup. Backups/WAL may retain old data, so this is not
a secure-erasure or compliance claim. No existing user conversations were purged during
implementation; deletion tests use isolated synthetic schemas.

Questions and rendered answers are intentionally persisted as sensitive local history.
There is no application-layer encryption or multi-user access control. Protect the
local database/OS account; avoid uploading secrets. Content-free telemetry is not an
immutable compliance audit: it follows conversation deletion/expiry.

Successful conversation turns store Stage 6 decision codes, model/provider profiles,
token counts where available, cache/ranking outcomes and timings. Failures store a
generic code rather than sensitive exception bodies. A database outage can prevent
a failure event from being persisted; we do not claim exactly-once inference after
a process crash. Completed-key replay prevents duplicate calls after a committed result.

Runtime metrics include outcome counts, latency sample count, p50/p95 total and
stage timing, known mean API cost and unknown-cost count. They sample at most the latest
1,000 retained conversation events. No query, answer, evidence, secret or hidden reasoning
is returned by the metrics endpoint. Failed-event latency is currently absent and its
cost unknown. Stateless /answers is not included in this persistent dashboard dataset.
Nested stage timings overlap; do not sum them as independent wall-clock durations.
There is no dashboard UI, automatic alert delivery or distributed tracing backend yet.

## Offline evaluation: use reviewed labels, not model confidence

The report CLI performs no inference, downloads or remote calls. It joins gold cases
and observations by exact ID/dataset version and rejects duplicates, missing cases,
invalid counts and nonfinite metrics. It returns aggregate metrics and task/modality/
local-role/adversarial-category slices.

~~~powershell
.\.venv\Scripts\python.exe -m rag_backend.evaluation.report --gold tests/fixtures/evaluation/gold.template.jsonl --observations tests/fixtures/evaluation/observations.template.jsonl --k 5
~~~

The shipped files are UNREVIEWED SYNTHETIC FORMAT EXAMPLES. Their times, evidence IDs
and outcomes are placeholders, not model measurements. Their gate deliberately fails.
A reviewed=true flag records an assertion by the reviewer, not cryptographic proof
of human review; do not flip it without the actual review.

Metric definitions:

- Retrieval precision@k: relevant unique retrieved IDs / k, then macro-average over
  cases with known evidence. Fewer than k results leave unfilled slots incorrect.
- Recall@k: relevant unique retrieved IDs / all known relevant IDs, macro-averaged
  over evidence-bearing cases. No-evidence cases are excluded, not scored perfect.
- Grounded claim precision: supported factual claims / total factual claims,
  micro-averaged from reviewed claim labels.
- Citation precision: citations supporting their associated claims / total citations,
  micro-averaged from reviewed labels, not merely existing citation IDs.
- Expected abstention/clarification correctness: correct specific safe outcome among
  cases expecting abstain/clarify. Overall outcome accuracy and task success separately
  penalize unnecessary abstention on answerable tasks.
- Task success: reviewer-approved task completion fraction. The local validator's
  pass status must not substitute for this label.
- Latency: p50/p95 using linear interpolation of actual request durations.
- Cost: mean known API cost with unknown-cost count; unknown never means free.
- Zero-denominator metrics are null, not 100%.

Starting local-MVP gates: at least 30 reviewed cases including five adversarial cases;
recall >= .90, grounded precision >= .95, citation precision >= .98, task success >= .90,
expected safe-outcome correctness >= .95, zero unsafe outputs, all known API costs zero,
and p95 <= 120,000 ms on the recorded local hardware/model profile.
These are initial engineering targets, not calibrated production SLOs or certification.
Gate success on this sample is not evidence of coverage beyond its tasks/modalities.

### Building the real gold set: pending human review

Project owner is the initial reviewer; a second reviewer should resolve ambiguous
or safety-critical labels. Use only synthetic/public/approved data. Start with
answer, search, excerpt-summary, comparison and generated-section cases; include
TXT/Markdown/searchable PDF, exact numbers, distractors, absent evidence, conflicting
sources, instruction injection and stale/deleted citations. Unsupported scanned/mixed
inputs should be checked for quarantine separately, not labeled as supported modalities.

1. Upload the approved fixture documents and let their versions publish.
2. Record questions, expected outcomes and exact relevant chunk IDs against those
   immutable versions. Replace placeholder IDs; preserve a source/version manifest.
3. Run queries using the documented APIs. Record ordered retrieved chunk IDs from
   /search for retrieval metrics and the corresponding /answers outcome/timing/usage.
   Do not treat final cited IDs alone as the full retrieved candidate list.
4. Review every answer claim and citation against original evidence; label support,
   task completion and unsafe output. Retain private response artifacts for review.
5. Record model/digest profiles and hardware with the run. Mark both files reviewed
   only after review, then run the report.
6. Refresh/version the set when sources, embedding profile, chunking, reranker,
   prompt, generator or validator changes. Keep a held-out set and add regression
   cases after failures. Repeat before release and after material data/model changes.

Actual reviewed corpus creation and real accuracy measurement are NOT complete.
No human labels, accuracy percentage or model calibration were invented here.
An automated batch inference runner, LLM judge calibration, confidence intervals,
full observability/alerts and larger representative benchmarks remain follow-up work.

## Verification and next step

Verified on 2026-10-04: all 207 tests passed with PostgreSQL and real local model
tests enabled (102.33 seconds). Ruff lint and formatting passed; pip found no
broken requirements. Alembic reports 0005_conversations at head. One existing
upstream Starlette test-client deprecation warning remains. The synthetic report
correctly failed its gate for incomplete human review, fewer than 30 cases and
fewer than five adversarial cases; its placeholder scores are not benchmark results.

Tests cover persistence/replay, history after service restart, key conflicts,
busy/expired lease fencing, generation failure, deletion during generation,
retention cleanup, source deletion and citation availability, API bounds, telemetry
separation, metric denominators and failed release gates. The existing real-model
smoke test also persists/replays its actual validated response.

The additive migration has been applied to the local PostgreSQL database. Docker
Desktop was restarted because its engine was initially unavailable. No unrelated
containers or database volumes were removed. No API/worker was left running.

Next recommended checkpoint: review the implemented backend against the frozen design,
identify the deliberately deferred modalities, and build/review the real evaluation
corpus before describing the system as production-ready. The simple Streamlit
upload/chat page remains a separate later task.
