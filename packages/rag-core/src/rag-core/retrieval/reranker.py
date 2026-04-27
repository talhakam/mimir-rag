# packages/rag-core/src/rag_core/retrieval/reranker.py
from sentence_transformers import CrossEncoder
from rag_core.models.document import SearchResult


class CrossEncoderReranker:
    """
    Rerank search results using a cross-encoder model.
    
    Why? Bi-encoders (embedding models) encode query and document 
    SEPARATELY — fast but approximate. Cross-encoders encode the
    (query, document) PAIR together — slower but much more accurate.
    
    Typical flow: bi-encoder retrieves top-20, cross-encoder reranks to top-5.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not results:
            return []

        # Score each (query, chunk) pair
        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)

        # Attach rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        # Sort by rerank score descending, take top_k
        reranked = sorted(results, key=lambda r: r.rerank_score or 0, reverse=True)[:top_k]

        # Update rank positions
        for i, result in enumerate(reranked):
            result.rank_position = i

        return reranked