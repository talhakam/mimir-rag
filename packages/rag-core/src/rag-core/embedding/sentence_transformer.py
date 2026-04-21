# packages/rag-core/src/rag_core/embedding/sentence_transformer.py
from sentence_transformers import SentenceTransformer
from rag_core.embedding.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """ 
        The Embedding service uses the Sentence Transformer library
        to convert text into 384-dim vectors. We have planned to also
        implement embeddings for images and tables in the future.    
    
        We are using L2 normalization so that cosine similarity can be 
        computed as a simple dot product in the vector DB.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # L2 norm → cosine similarity = dot product
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dimensions(self) -> int:
        return self._dimensions