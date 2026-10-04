# Stage 3: parsing, normalization, quality and chunks

Completed for searchable PDF, UTF-8 TXT and Markdown. This is the agreed text
milestone, not completion of OCR or the full mixed-modality architecture.

| Step | Change | Why |
| --- | --- | --- |
| Detect | Check extension and PDF signature. | A filename does not prove format. |
| Parse | pdfplumber extracts PDF text/layout; markdown-it-py preserves Markdown structure; strict UTF-8 for text. | Retain page/box or line locations for citations. |
| Analyze | Identify headings, group PDF paragraphs, retain recognized tables/code/image references. | Chunk by structure instead of arbitrary substrings. |
| Normalize | Unicode NFC and newline normalization, retaining raw extracted text. | Stable processing without losing original evidence. |
| Quality | Reject empty/unreadable/unsupported or complex content and preserve reason codes. | Avoid indexing unreliable or partial extraction. |
| Chunk | Heading-aware prose with sentence/paragraph boundaries; bounded contextual overlap. | Keep nearby meaning together and trace spans to originals. |
| Persist | Canonical blocks, chunk spans, profile and quality report commit with job completion. | No partial chunk sets or stale worker writes. |

Default limits: 200 PDF pages, 2 million characters, 20,000 blocks, 120 seconds.
Prose chunks target 1,800 characters with up to 160 characters of adjacent prose
context. These are character caps, not model-token guarantees. Indexing refuses
model input overflow rather than silently truncating.

Detected tables, images, code and HTML quarantine the version in this milestone;
canonical structure is retained where extraction succeeded, but no chunks are
published. Atomic table/code chunk handling is tested; row windows and mixed
semantic enrichment are later work. Scanned PDF/OCR, rotated/complex layouts,
encrypted and malformed PDFs have explicit failure/quarantine behavior.
PDF reading order, heading and table detection remain heuristics, not guarantees.

Original binaries remain in local storage. PDF raw_text means extracted text,
not the complete binary original. Chunk offsets reference normalized block text;
source citations use original page/bounding-box or line locations.

Parsing runs in a separate process with time/input limits and lease heartbeats.
It is not an OS security sandbox or hard memory quota. No code is executed, no
images fetched, and no LLM/provider is called here.

API additions: GET /jobs?version_id=..., and GET
/sources/{source_id}/versions/{version_id}/blocks or /chunks. Version responses
expose preprocessing_status, profile, quality_report and timestamp. Deletion and
source/version mismatch prevent evidence reads. Authentication remains deferred
for this loopback-only local MVP.

The quality_report is an extraction-time snapshot (searchable=false), not live
publication status. Use indexing_status and the source published_version_id once
Stage 4 runs. Extraction quality is distinct from future enrichment/answer validators.

Stage 3 verification: 88 tests passed, including PostgreSQL integration, PDF
subprocess parsing, quarantine, provenance, lease fencing and rollback.

References: [pdfplumber](https://github.com/jsvine/pdfplumber) and
[markdown-it-py](https://markdown-it-py.readthedocs.io/en/latest/using.html).
