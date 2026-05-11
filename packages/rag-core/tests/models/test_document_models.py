from uuid import UUID

from rag_core.models.document import Chunk, SearchResult


def test_chunk_generates_id_and_default_metadata() -> None:
    chunk = Chunk(
        document_id="doc-1",
        chunk_index=0,
        content="Example chunk",
        start_char=0,
        end_char=13,
    )

    assert UUID(chunk.id)
    assert chunk.metadata == {}


def test_search_result_wraps_chunk_with_scores() -> None:
    chunk = Chunk(
        document_id="doc-1",
        chunk_index=0,
        content="Example chunk",
        start_char=0,
        end_char=13,
    )

    result = SearchResult(chunk=chunk, similarity_score=0.92, rank_position=0)

    assert result.chunk == chunk
    assert result.similarity_score == 0.92
    assert result.rank_position == 0
    assert result.rerank_score is None
