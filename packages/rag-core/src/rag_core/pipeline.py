# packages/rag-core/src/rag_core/pipeline.py
from pathlib import Path
from collections.abc import AsyncIterator
from jinja2 import Environment, FileSystemLoader

from rag_core.ingestion.base import BaseParser
from rag_core.ingestion.registry import ParserRegistry
from rag_core.chunking.base import BaseChunker
from rag_core.embedding.base import BaseEmbedder
from rag_core.retrieval.base import BaseVectorStore
from rag_core.retrieval.reranker import CrossEncoderReranker
from rag_core.generation.base import BaseLLM
from rag_core.models.document import Chunk, SearchResult


PROMPTS_DIR = Path(__file__).parent / "generation" / "prompts"


class RAGPipeline:
    """
    Orchestrates the full RAG flow.
    
    This class doesn't DO any of the work — it delegates to
    injected components (parser, chunker, embedder, etc.)
    and coordinates the data flow between them.
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        reranker: CrossEncoderReranker | None = None,
        retrieval_top_k: int = 20,
        final_top_k: int = 5,
    ):
        self.parser_registry = parser_registry
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k

        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            autoescape=False,
        )

    # ── WRITE PATH ─────────────────────────────────────────────
    async def ingest(self, file_path: Path, document_id: str,
                     metadata: dict | None = None) -> list[Chunk]:
        """Parse → Chunk → Embed → Store. Returns created chunks."""
        # 1. Parse
        parser = self.parser_registry.get_parser(file_path)
        parsed = parser.parse(file_path, metadata)

        # 2. Chunk
        chunks = self.chunker.chunk(parsed.content, document_id, parsed.metadata)

        # 3. Embed
        texts = [c.content for c in chunks]
        vectors = self.embedder.embed(texts)

        # 4. Store
        await self.vector_store.upsert(chunks, vectors)

        return chunks

    # ── READ PATH ──────────────────────────────────────────────
    async def query(self, question: str,
                    filters: dict | None = None) -> tuple[str, list[SearchResult]]:
        """Full RAG: embed question → retrieve → rerank → generate answer."""
        retrieved = await self.retrieve(question, filters)
        answer = await self._generate(question, retrieved)
        return answer, retrieved

    async def query_stream(self, question: str,
                           filters: dict | None = None) -> tuple[AsyncIterator[str], list[SearchResult]]:
        """Streaming version — yields tokens as they come."""
        retrieved = await self.retrieve(question, filters)
        token_stream = self._generate_stream(question, retrieved)
        return token_stream, retrieved

    async def retrieve(self, question: str,
                       filters: dict | None = None) -> list[SearchResult]:
        """Embed → Vector search → Rerank → Return top chunks."""
        # 1. Embed the question
        query_vector = self.embedder.embed([question])[0]

        # 2. Vector similarity search (broad retrieval)
        candidates = await self.vector_store.search(
            query_vector, top_k=self.retrieval_top_k, filters=filters
        )

        # 3. Rerank (narrow to best chunks)
        if self.reranker and candidates:
            return self.reranker.rerank(question, candidates, top_k=self.final_top_k)

        return candidates[:self.final_top_k]

    async def _generate(self, question: str, chunks: list[SearchResult]) -> str:
        prompt = self._build_prompt(question, chunks)
        return await self.llm.generate(prompt)

    async def _generate_stream(self, question: str,
                                chunks: list[SearchResult]) -> AsyncIterator[str]:
        prompt = self._build_prompt(question, chunks)
        async for token in self.llm.generate_stream(prompt):
            yield token

    def _build_prompt(self, question: str, chunks: list[SearchResult]) -> str:
        template = self._jinja_env.get_template("qa.jinja2")
        return template.render(question=question, chunks=[
            {
                "content": r.chunk.content,
                "document_name": r.chunk.metadata.get("filename", "Unknown"),
                "page_number": r.chunk.metadata.get("page_number", "?"),
            }
            for r in chunks
        ])