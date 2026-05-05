# packages/rag-core/src/rag_core/ingestion/registry.py
from pathlib import Path
from rag_core.ingestion.base import BaseParser
from rag_core.ingestion.pdf_parser import PDFParser
from rag_core.ingestion.text_parser import TextParser


class ParserRegistry:
    """Maps file extensions to parser instances."""

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:
        for ext in parser.supported_extensions():
            self._parsers[ext.lower()] = parser

    def get_parser(self, file_path: Path) -> BaseParser:
        ext = file_path.suffix.lower()
        if ext not in self._parsers:
            raise ValueError(f"No parser registered for extension: {ext}")
        return self._parsers[ext]


def create_default_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register(PDFParser())
    registry.register(TextParser())
    return registry