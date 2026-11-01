"""
Ranking utilities and Cross-Encoder reranking.

Combines or reranks retrieval candidates to provide calibrated relevance scores.
"""

from typing import List, Optional
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks candidate retrieval results using a pretrained cross-encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List, top_k: Optional[int] = None) -> List:
        """
        Score (query, document_text) pairs and return sorted results.
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs)

        from src.retrieval.pipeline import RetrievalResult
        import numpy as np

        reranked = []
        for r, score in zip(results, scores):
            # Sigmoid normalization maps logits to calibrated [0, 1] probabilities
            prob = float(1.0 / (1.0 + np.exp(-float(score))))
            reranked.append(
                RetrievalResult(
                    text=r.text,
                    file_path=r.file_path,
                    start=r.start,
                    end=r.end,
                    score=prob,
                    chunk_id=r.chunk_id,
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked


def rerank(results: List, query: str) -> List:
    """Legacy pass-through function for backward compatibility."""
    return results
