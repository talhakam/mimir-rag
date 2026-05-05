# packages/rag-core/src/rag_core/ingestion/text_parser.py
from pathlib import Path

from rag_core.ingestion.base import BaseParser
from rag_core.models.document import ParsedDocument


class TextParser(BaseParser):
    """
    Parser for plain text and Markdown documents.
    Reads the file as UTF-8 text and collects basic file metadata.
    """

    def supported_extensions(self) -> list[str]:
        return [".txt", ".md"]

    def parse(self, file_path: Path, metadata: dict | None = None) -> ParsedDocument:
        content = file_path.read_text(encoding="utf-8")

        doc_metadata = metadata or {}
        doc_metadata.update({
            "title": file_path.stem,
            "author": "",
            "page_count": 1,
            "filename": file_path.name,
            "file_type": file_path.suffix.lstrip(".").lower(),
        })

        return ParsedDocument(content=content, metadata=doc_metadata)