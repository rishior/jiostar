"""
Abstention logic for the retrieval pipeline.

Determines when the system should return no results because no passage is
relevant enough to answer the query. This is critical for handling unanswerable
queries (questions about clauses that do not exist in the indexed contracts).
"""

from typing import List
from src.config import AbstentionConfig


def should_abstain(results: List, config: AbstentionConfig) -> bool:
    """
    Decide whether to abstain from returning results.
    
    The system abstains when the top retrieval score falls below the configured
    threshold. This catches queries about topics not covered by the indexed
    contracts.
    
    Args:
        results: List of RetrievalResult objects, sorted by descending score.
        config: Abstention configuration with threshold and gap parameters.
    
    Returns:
        True if the system should return an empty result set.
    """
    if not results:
        return True

    if not config.enabled:
        return False

    top_score = results[0].score

    # Abstain if the best score is below the threshold
    if top_score < config.score_threshold:
        return True

    # Abstain if the gap between top score and threshold is too small
    if config.min_gap > 0:
        gap = top_score - config.score_threshold
        if gap < config.min_gap:
            return True

    return False

