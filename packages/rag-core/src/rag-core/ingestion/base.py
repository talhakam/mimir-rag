from abc import ABC, abstractmethod
from pathlib import Path
from rag_core.models.document import ParsedDocument


class BaseParser(ABC):
    """Interface that all document parsers must implement."""

    @abstractmethod
    def parse(self, file_path: Path, metadata: dict | None = None) -> ParsedDocument:
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        ...