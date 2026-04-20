# packages/rag-core/src/rag_core/chunking/base.py
from abc import ABC, abstractmethod
from rag_core.models.document import Chunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, document_id: str, metadata: dict | None = None) -> list[Chunk]:
        ...