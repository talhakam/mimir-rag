# packages/rag-core/src/rag_core/ingestion/pdf_parser.py
from email.parser import Parser
from pathlib import Path
import fitz  # PyMuPDF

from rag_core.ingestion.base import BaseParser
from rag_core.models.document import ParsedDocument


class PDFParser(BaseParser):
    """
       Parser for PDF documents using PyMuPDF. 
       Extracts text from each page and combines it into a single string, 
       while also collecting metadata. 
    """
    
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: Path, metadata: dict | None = None) -> ParsedDocument:
        doc = fitz.open(file_path)
        pages_text: list[str] = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages_text.append(f"[Page {page_num}]\n{text}")

        combined_text = "\n\n".join(pages_text)

        doc_metadata = metadata or {}
        doc_metadata.update({
            "title": doc.metadata.get("title", file_path.stem),
            "author": doc.metadata.get("author", ""),
            "page_count": len(doc),
            "filename": file_path.name,
            "file_type": "pdf",
        })
        doc.close()

        return ParsedDocument(content=combined_text, metadata=doc_metadata)