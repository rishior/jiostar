"""
Embedding generation for document chunks.

Wraps sentence-transformers to generate dense embeddings for chunks and queries.
Embeddings are generated in batches for efficiency.
"""

import numpy as np
from typing import List, Optional

from src.config import EmbeddingConfig


class Embedder:
    """
    Generates dense embeddings using a sentence-transformers model.
    
    The model is loaded lazily on first use and cached for subsequent calls.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into dense vectors.
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.config.normalize,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query text.
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.config.normalize,
        )
        return np.array(embedding[0], dtype=np.float32)

