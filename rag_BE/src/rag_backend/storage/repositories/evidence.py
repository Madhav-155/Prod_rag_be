import json

from sqlalchemy import text

from rag_backend.ingestion.contracts import PreprocessingResult


def persist_result(connection, version_id, result: PreprocessingResult):
    if result.version_id != version_id or (result.status != 'approved' and result.chunks):
        raise ValueError('Invalid evidence result')
    if result.status == 'approved' and (result.reasons or not result.blocks or not result.chunks):
        raise ValueError('Approved evidence must have supported blocks and chunks without errors')
    blocks_by_id = {block.id: block for block in result.blocks}
    for chunk in result.chunks:
        for span in chunk.spans:
            block = blocks_by_id.get(span.block_id)
            if block is None or not 0 <= span.start < span.end <= len(block.text):
                raise ValueError('Invalid original evidence reference')
    # Lease/source checks occur in the caller's transaction. All evidence and job
    # completion commit together; no partial chunk set becomes visible.
    if result.blocks:
        connection.execute(
            text("""
            INSERT INTO canonical_blocks(
                id, version_id, ordinal, kind, raw_text, text, locator, attributes, heading_path
            ) VALUES (:id, :version, :ordinal, :kind, :raw_text, :text,
                CAST(:locator AS jsonb), CAST(:attributes AS jsonb), CAST(:heading_path AS jsonb))
            ON CONFLICT(id) DO NOTHING
        """),
            [
                {
                    **block.model_dump(exclude={'locator', 'attributes', 'heading_path'}),
                    'version': version_id,
                    'locator': json.dumps(block.locator),
                    'attributes': json.dumps(block.attributes),
                    'heading_path': json.dumps(block.heading_path),
                }
                for block in result.blocks
            ],
        )
    if result.chunks:
        connection.execute(
            text("""
            INSERT INTO evidence_chunks(id, version_id, ordinal, kind, text, heading_path, spans)
            VALUES (:id, :version, :ordinal, :kind, :text,
                CAST(:heading_path AS jsonb), CAST(:spans AS jsonb))
            ON CONFLICT(id) DO NOTHING
        """),
            [
                {
                    'id': chunk.id,
                    'version': version_id,
                    'ordinal': chunk.ordinal,
                    'kind': chunk.kind,
                    'text': chunk.text,
                    'heading_path': json.dumps(chunk.heading_path),
                    'spans': json.dumps([span.model_dump(mode='json') for span in chunk.spans]),
                }
                for chunk in result.chunks
            ],
        )
    report = {
        'reasons': result.reasons,
        'metrics': result.metrics,
        'completed_stages': result.completed_stages,
        'block_count': len(result.blocks),
        'chunk_count': len(result.chunks),
        'searchable': False,
    }
    connection.execute(
        text("""
        UPDATE source_versions SET preprocessing_status = :status,
            preprocessing_profile = :profile, quality_report = CAST(:report AS jsonb),
            preprocessed_at = clock_timestamp()
        WHERE id = :version
    """),
        {
            'status': result.status,
            'profile': result.profile,
            'report': json.dumps(report),
            'version': version_id,
        },
    )
