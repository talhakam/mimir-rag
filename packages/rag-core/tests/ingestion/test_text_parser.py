from pathlib import Path

import pytest

from rag_core.ingestion.registry import create_default_registry
from rag_core.ingestion.text_parser import TextParser


def test_text_parser_reads_txt_content(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Hello from a text file.\nSecond line.", encoding="utf-8")

    parsed = TextParser().parse(file_path)

    assert parsed.content == "Hello from a text file.\nSecond line."
    assert parsed.metadata["filename"] == "notes.txt"
    assert parsed.metadata["file_type"] == "txt"


def test_text_parser_reads_markdown_content(tmp_path: Path) -> None:
    file_path = tmp_path / "guide.md"
    file_path.write_text("# Guide\n\nThis is markdown.", encoding="utf-8")

    parsed = TextParser().parse(file_path)

    assert parsed.content == "# Guide\n\nThis is markdown."
    assert parsed.metadata["filename"] == "guide.md"
    assert parsed.metadata["file_type"] == "md"


def test_text_parser_preserves_input_metadata_without_mutating_it(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Metadata should be copied.", encoding="utf-8")
    metadata = {"source": "manual-upload"}

    parsed = TextParser().parse(file_path, metadata=metadata)

    assert parsed.metadata["source"] == "manual-upload"
    assert parsed.metadata["filename"] == "notes.txt"
    assert metadata == {"source": "manual-upload"}


def test_parser_registry_routes_txt_to_text_parser() -> None:
    registry = create_default_registry()

    parser = registry.get_parser(Path("notes.txt"))

    assert isinstance(parser, TextParser)


def test_parser_registry_rejects_unsupported_docx() -> None:
    registry = create_default_registry()

    with pytest.raises(ValueError, match="No parser registered for extension: .docx"):
        registry.get_parser(Path("document.docx"))
