"""
Ranking utilities (placeholder for future reranking).

Currently a pass-through; results are ranked by the scores from the retrieval
pipeline. This module exists as the extension point for adding cross-encoder
rerankers or LLM-based reranking in later iterations.
"""

from typing import List


def rerank(results: List, query: str) -> List:
    """
    Rerank retrieval results.
    
    Currently returns results unchanged. Future iterations may add:
    - Cross-encoder reranking
    - LLM-based reranking
    - Feature-based reranking with multiple signals
    """
    return results

