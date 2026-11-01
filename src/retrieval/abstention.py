"""
Abstention logic for the retrieval pipeline.

Determines when the system should return no results because no passage is
relevant enough to answer the query. This is critical for handling unanswerable
queries (questions about clauses that do not exist in the indexed contracts).
"""

from typing import List
from src.config import AbstentionConfig


def _get_score(item) -> float:
    return item.score if hasattr(item, "score") else float(item)


def should_abstain(results: List, config: AbstentionConfig) -> bool:
    """
    Decide whether to abstain from returning results.
    
    The system abstains when the top retrieval score falls below the configured
    threshold. This catches queries about topics not covered by the indexed
    contracts.
    
    Args:
        results: List of RetrievalResult objects or score items, sorted by descending score.
        config: Abstention configuration with threshold and gap parameters.
    
    Returns:
        True if the system should return an empty result set.
    """
    if not results:
        return True

    if not config.enabled:
        return False

    top_score = _get_score(results[0])

    # Select threshold based on configuration
    if getattr(config, "use_reranker_score", False):
        threshold = config.reranker_threshold
    else:
        threshold = config.score_threshold

    # Abstain if the best score is below the threshold
    if top_score < threshold:
        return True

    # Abstain if the gap between top score and threshold is too small
    if config.min_gap > 0:
        gap = top_score - threshold
        if gap < config.min_gap:
            return True

    # Abstain if margin between top-1 and top-2 candidate is below margin_threshold
    margin_threshold = getattr(config, "margin_threshold", 0.0)
    if margin_threshold > 0 and len(results) >= 2:
        margin = _get_score(results[0]) - _get_score(results[1])
        if margin < margin_threshold:
            return True

    return False

