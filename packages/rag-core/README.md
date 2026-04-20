# rag-core

The core RAG pipeline library for RAG Engine. It can be imported by the API layer or used directly in scripts and notebooks.

## Overview

`rag-core` provides all the abstractions and implementations needed to run a full Retrieval-Augmented Generation pipeline:

| Stage | Interface | Status |
|-------|-----------|--------|
| Ingestion (parsing) | `BaseParser` | Implemented (PDF, TXT, MD) |
| Chunking | `BaseChunker` |  Interface only |
| Embedding | `BaseEmbedder` |  Interface only |
| Retrieval / Vector Store | `BaseVectorStore` | Interface only |
| Generation (LLM) | `BaseLLM` | Interface only |

## Package Structure

```
packages/rag-core/
├── src/
│   └── rag-core/
│       ├── ingestion/          # Document parsing services
│       │   ├── base.py         # BaseParser interface
│       │   ├── pdf_parser.py   # PDFParser (PyMuPDF)
│       │   ├── text_parser.py  # TextParser (.txt, .md)
│       │   └── registry.py     # ParserRegistry router
│       ├── chunking/
│       │   └── base.py         # BaseChunker interface
│       ├── embedding/
│       │   └── base.py         # BaseEmbedder interface
│       ├── retrieval/
│       │   └── base.py         # BaseVectorStore interface
│       ├── generation/
│       │   └── base.py         # BaseLLM interface
│       └── models/
│           └── document.py     # ParsedDocument, Chunk, SearchResult
├── tests/
└── pyproject.toml
```

## Installation

From the monorepo root (recommended via uv workspace):

```bash
uv sync
```

Or directly in editable mode:

```bash
pip install -e packages/rag-core
```

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `pymupdf` | >=1.24 | PDF text extraction |
| `sentence-transformers` | >=3 | Embeddings + cross-encoder reranking |
| `qdrant-client` | >=1.12 | Vector store client |
| `httpx` | >=0.27 | Ollama HTTP client |
| `jinja2` | >=3.1 | Prompt templates |
| `pydantic` | >=2.0 | Data models |

Requires Python >= 3.12.

## Design Principles

- **Interface-first** — every pipeline stage is an abstract base class. Swap any component without touching the pipeline.
- **No framework coupling** — zero FastAPI, no HTTP servers. Pure library.
- **Pydantic models throughout** — all data flowing between stages is typed and validated.
- **Extensible by registration** — add a new parser/embedder/store by implementing the interface and registering it.
