"""
Vector index for dense retrieval.

Uses FAISS for efficient similarity search over document chunk embeddings.
The index stores chunk metadata alongside vectors so retrieved results include
document paths and character offsets.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import asdict

from src.indexing.chunker import Chunk
from src.config import INDEX_DIR


class VectorIndex:
    """
    FAISS-based vector index with chunk metadata storage.
    
    Stores both the FAISS index for fast similarity search and a parallel
    metadata list that maps each vector to its source chunk information.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # inner product (cosine sim for normalized vectors)
        self.chunks: List[dict] = []  # metadata for each indexed chunk

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        """
        Add chunk embeddings and their metadata to the index.
        
        The embeddings array and chunks list must have the same length.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) does not match "
                f"chunks count ({len(chunks)})"
            )

        self.index.add(embeddings.astype(np.float32))
        for chunk in chunks:
            self.chunks.append(asdict(chunk))

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[dict, float]]:
        """
        Search the index for the most similar chunks to the query.
        
        Returns:
            List of (chunk_metadata_dict, similarity_score) tuples,
            sorted by descending similarity.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, path: Optional[Path] = None):
        """Save the index and metadata to disk."""
        if path is None:
            path = INDEX_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "VectorIndex":
        """Load a previously saved index from disk."""
        if path is None:
            path = INDEX_DIR
        path = Path(path)

        index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        vi = cls(dimension=index.d)
        vi.index = index
        vi.chunks = chunks
        return vi

