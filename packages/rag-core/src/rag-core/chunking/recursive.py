# packages/rag-core/src/rag_core/chunking/recursive.py
from rag_core.chunking.base import BaseChunker
from rag_core.models.document import Chunk


class RecursiveChunker(BaseChunker):
    """
    Split text recursively by trying separators in order:
    paragraphs → sentences → words → characters.
    
    The idea: prefer splitting at semantically meaningful boundaries.
    A paragraph break is a better split point than mid-sentence.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, document_id: str, metadata: dict | None = None) -> list[Chunk]:
        raw_chunks = self._split_recursive(text, self.separators)
        
        chunks: list[Chunk] = []
        for i, (content, start, end) in enumerate(raw_chunks):
            chunks.append(Chunk(
                document_id=document_id,
                chunk_index=i,
                content=content,
                start_char=start,
                end_char=end,
                metadata=metadata or {},
            ))
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[tuple[str, int, int]]:
        """Returns list of (chunk_text, start_char, end_char)."""
        if not text:
            return []

        # If text fits in one chunk, return it
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        # Try each separator starting with the most meaningful
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Last resort: hard character split
            return self._hard_split(text)

        parts = text.split(separator)

        # Merge parts into chunks that fit within chunk_size
        result: list[tuple[str, int, int]] = []
        current_chunk = ""
        current_start = 0

        for part in parts:
            candidate = current_chunk + separator + part if current_chunk else part
            
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    end = current_start + len(current_chunk)
                    result.append((current_chunk, current_start, end))
                    # Overlap: step back by overlap chars
                    overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap else ""
                    current_start = end - len(overlap_text)
                    current_chunk = overlap_text + separator + part if overlap_text else part
                else:
                    # Single part bigger than chunk_size — recurse with finer separator
                    if remaining_separators:
                        sub_chunks = self._split_recursive(part, remaining_separators)
                        for content, s, e in sub_chunks:
                            result.append((content, current_start + s, current_start + e))
                        current_start += len(part) + len(separator)
                        current_chunk = ""
                    else:
                        result.extend(self._hard_split(part))
                        current_start += len(part) + len(separator)
                        current_chunk = ""

        if current_chunk.strip():
            result.append((current_chunk, current_start, current_start + len(current_chunk)))

        return result

    def _hard_split(self, text: str) -> list[tuple[str, int, int]]:
        result = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            end = min(i + self.chunk_size, len(text))
            result.append((text[i:end], i, end))
            if end == len(text):
                break
        return result