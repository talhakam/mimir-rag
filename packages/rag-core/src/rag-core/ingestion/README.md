# Ingestion Service

Handles document parsing; converting raw files (PDF, TXT, MD) into a `ParsedDocument` ready for chunking. Lives in `packages/rag-core/src/rag-core/ingestion/`.

## Current Status

1. `BaseParser` abstract interface 
2. PDF parsing via PyMuPDF 
3. Plain text and Markdown parsing 
4. `ParserRegistry` router to map file extensions for parsing

## Planned

1. Parsers for images, audio etc.
2. Media type format validation.
3. Fallback pipeline for ingestion service.

## File-by-File Breakdown

---

### `base.py` — `BaseParser` Interface

The interface that every parser must follow. It just enforces that all parsers expose the same two methods.

| Method | What you pass in | What you get back |
|--------|-----------------|-------------------|
| `parse(file_path, metadata)` | Path to the file + optional extra metadata dict | `ParsedDocument` with extracted text and metadata |
| `supported_extensions()` | nothing | List of extensions this parser handles, e.g. `[".pdf"]` |

Any class that inherits `BaseParser` **must** implement both methods, or Python will raise an error at instantiation.

---

### `pdf_parser.py` — `PDFParser`

Extracts text from PDF files, page by page, using the **PyMuPDF** library (`import fitz`).

**Supports:** `.pdf`

**Step-by-step:**

1. Opens the PDF file with `fitz.open(file_path)`.
2. Loops through every page, starting at page number 1.
3. Extracts the plain text of each page using `page.get_text("text")`.
4. Skips any page that is blank (whitespace only).
5. Prefixes each kept page's text with `[Page N]` so the source page is always traceable later.
6. Joins all pages into one continuous string separated by blank lines.
7. Builds a metadata dict: pulls `title` and `author` from the PDF's own embedded metadata if present, otherwise falls back to the filename stem and an empty string.
8. Closes the file handle and returns a `ParsedDocument`. File always closes even if error occurs mid-parse.

---

### `text_parser.py` — `TextParser`

Reads plain text and Markdown files as-is. Uses Python's standard library only — no third-party packages needed.

**Supports:** `.txt`, `.md`

**Step-by-step:**

1. Reads the entire file content as a UTF-8 string using `file_path.read_text(encoding="utf-8")`.
2. Builds a metadata dict using the filename stem as the title and the file suffix to set `file_type` (so `.md` → `"md"`, `.txt` → `"txt"`).
3. Sets `page_count` to `1` since text files have no page concept.
4. Returns a `ParsedDocument` with the raw content and metadata.

---

### `registry.py` — `ParserRegistry`

The router. Its only job is to look at an uploaded file's extension and hand back the right parser for it. Nothing else in the pipeline needs to know which parser handles which format.

**How the registry is built (`create_default_registry`):**

| Extension | Parser assigned |
|-----------|----------------|
| `.pdf` | `PDFParser` |
| `.txt` | `TextParser` |
| `.md` | `TextParser` |

**How a file gets routed (`get_parser`):**

1. Reads the file extension from the path and lowercases it.
2. Looks it up in the internal map.
3. Returns the matching parser if found.
4. Raises a `ValueError` with a clear message if the extension is not registered.

| Method | Input | Output |
|--------|-------|--------|
| `register(parser)` | Any `BaseParser` instance | Updates internal map — returns nothing |
| `get_parser(file_path)` | `Path` to the file | Correct parser instance, or `ValueError` |
| `create_default_registry()` | — | Ready-to-use registry with PDF + Text parsers loaded |

---

## End-to-End Ingestion Flow

```
User uploads file
      │
      ▼
registry = create_default_registry()
      │
      ▼
parser = registry.get_parser(file_path)
      │                │
      │                └─ raises ValueError if extension unsupported
      ▼
parsed_doc = parser.parse(file_path, metadata={...})
      │
      ▼
ParsedDocument
  ├── content   → full extracted text (goes to chunking)
  └── metadata  → title, author, filename, file_type, page_count, ...
```

## Adding a New Parser

1. Create `your_parser.py` in this folder.
2. Inherit from `BaseParser` and implement both methods.
3. Register it in `create_default_registry()` in `registry.py`.

```python
# example: docx_parser.py
class DocxParser(BaseParser):
    def supported_extensions(self) -> list[str]:
        return [".docx"]

    def parse(self, file_path: Path, metadata: dict | None = None) -> ParsedDocument:
        ...
```

```python
# registry.py — create_default_registry
registry.register(DocxParser())
```

No other changes needed anywhere in the pipeline.
