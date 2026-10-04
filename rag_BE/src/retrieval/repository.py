import json

from sqlalchemy import text


class RetrievalRepository:
    def __init__(self, engine):
        self.engine = engine

    def candidates(self, query, vector, profile, source_ids):
        if source_ids == []:
            return []
        scope = 'AND r.source_id = ANY(:sources)' if source_ids is not None else ''
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text(f"""
                WITH scoped AS NOT MATERIALIZED (
                    SELECT r.* FROM published_retrieval_records r
                    WHERE r.profile_id=:profile {scope}
                ), lexical AS (
                    SELECT chunk_id, ts_rank_cd(search_terms,
                        websearch_to_tsquery('simple', :query)) AS score
                    FROM scoped WHERE search_terms @@ websearch_to_tsquery('simple', :query)
                    ORDER BY score DESC, chunk_id LIMIT 40
                ), semantic AS (
                    SELECT chunk_id, embedding <=> CAST(:vector AS vector) AS distance
                    FROM scoped ORDER BY distance, chunk_id LIMIT 40
                ), ranked AS (
                    SELECT chunk_id, 'lexical' AS channel,
                        row_number() OVER (ORDER BY score DESC, chunk_id) AS rank FROM lexical
                    UNION ALL
                    SELECT chunk_id, 'semantic' AS channel,
                        row_number() OVER (ORDER BY distance, chunk_id) AS rank FROM semantic
                )
                SELECT k.*, r.source_id, r.version_id, r.page_content AS text, c.spans
                FROM ranked k JOIN scoped r USING(chunk_id)
                JOIN evidence_chunks c ON c.id=r.chunk_id
                ORDER BY k.channel, k.rank
            """),
                    {
                        'query': query,
                        'vector': json.dumps(vector),
                        'profile': profile,
                        'sources': source_ids,
                    },
                ).mappings()
            ]

    def hydrate(self, candidates, profile, source_ids):
        if not candidates or source_ids == []:
            return []
        scope = 'AND r.source_id = ANY(:sources)' if source_ids is not None else ''
        with self.engine.connect() as connection:
            # One new SQL snapshot checks current publication AND loads canonical blocks.
            rows = list(
                connection.execute(
                    text(f"""
                SELECT r.chunk_id, r.source_id, r.version_id, c.text, c.heading_path, c.spans,
                    v.number AS version_number, v.original_name AS filename,
                    b.id AS block_id, b.kind, b.text AS block_text, b.locator
                FROM published_retrieval_records r
                JOIN evidence_chunks c ON c.id=r.chunk_id AND c.version_id=r.version_id
                JOIN source_versions v ON v.id=r.version_id
                CROSS JOIN LATERAL jsonb_array_elements(c.spans) span
                JOIN canonical_blocks b ON b.id=CAST(span->>'block_id' AS uuid)
                    AND b.version_id=r.version_id
                WHERE r.chunk_id = ANY(:chunks) AND r.profile_id=:profile {scope}
            """),
                    {
                        'chunks': [c['chunk_id'] for c in candidates],
                        'profile': profile,
                        'sources': source_ids,
                    },
                ).mappings()
            )
        grouped = {}
        for row in rows:
            item = grouped.setdefault(row['chunk_id'], {'row': row, 'blocks': {}})
            item['blocks'][str(row['block_id'])] = row
        hydrated = []
        for candidate in candidates:
            item = grouped.get(candidate['chunk_id'])
            if item is None:
                continue
            row, blocks = item['row'], item['blocks']
            citations = []
            for span in row['spans']:
                block = blocks.get(span['block_id'])
                if (
                    block is None
                    or not block['locator']
                    or not 0 <= span['start'] < span['end'] <= len(block['block_text'])
                ):
                    citations = []
                    break
                citations.append(
                    {
                        'block_id': block['block_id'],
                        'kind': block['kind'],
                        'locator': block['locator'],
                        'start': span['start'],
                        'end': span['end'],
                        'role': span['role'],
                        'excerpt': block['block_text'][span['start'] : span['end']],
                    }
                )
            if citations:
                hydrated.append(
                    {
                        **{
                            key: row[key]
                            for key in (
                                'chunk_id',
                                'source_id',
                                'version_id',
                                'version_number',
                                'filename',
                                'text',
                                'heading_path',
                            )
                        },
                        'citations': citations,
                        'rrf_score': candidate['rrf_score'],
                        'rerank_score': candidate['rerank_score'],
                    }
                )
        return hydrated
