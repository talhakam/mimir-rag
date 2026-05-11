from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """Text and metadata extracted from a source document."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A smaller text segment created from a parsed document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A retrieved chunk with ranking scores."""

    chunk: Chunk
    similarity_score: float
    rank_position: int
    rerank_score: float | None = None
