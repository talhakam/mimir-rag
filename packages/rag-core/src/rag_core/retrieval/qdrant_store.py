# packages/rag-core/src/rag_core/retrieval/qdrant_store.py
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from rag_core.retrieval.base import BaseVectorStore
from rag_core.models.document import Chunk, SearchResult


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, url: str = "localhost", port: int = 6333,
                 collection_name: str = "documents", vector_size: int = 384):
        self._client = AsyncQdrantClient(url=url, port=port)
        self._collection = collection_name
        self._vector_size = vector_size

    async def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = await self._client.get_collections()
        exists = any(c.name == self._collection for c in collections.collections)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(self, vector: list[float], top_k: int = 10,
                     filters: dict | None = None) -> list[SearchResult]:
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        search_results = []
        for i, point in enumerate(results.points):
            payload = point.payload or {}
            chunk = Chunk(
                id=str(point.id),
                document_id=payload.get("document_id", ""),
                chunk_index=payload.get("chunk_index", 0),
                content=payload.get("content", ""),
                start_char=0,
                end_char=0,
                metadata={k: v for k, v in payload.items()
                          if k not in ("document_id", "chunk_index", "content")},
            )
            search_results.append(SearchResult(
                chunk=chunk,
                similarity_score=point.score,
                rank_position=i,
            ))
        return search_results

    async def delete(self, document_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]),
        )