# packages/rag-core/src/rag_core/retrieval/base.py
from abc import ABC, abstractmethod
from rag_core.models.document import Chunk, SearchResult


class BaseVectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        ...

    @abstractmethod
    async def search(self, vector: list[float], top_k: int = 10,
                     filters: dict | None = None) -> list[SearchResult]:
        ...

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        ...