"""
Evaluation metrics for the LegalBench-RAG benchmark.

Implements character-level precision and recall as defined in the LegalBench-RAG
paper (arxiv.org/abs/2408.10343). These metrics measure how well retrieved
passages overlap with ground-truth answer spans at the character level.
"""

from typing import List, Tuple, Set


def char_level_precision(
    retrieved_spans: List[Tuple[str, int, int]],
    gold_spans: List[Tuple[str, int, int]],
) -> float:
    """
    Compute character-level precision.
    
    Precision = |retrieved_chars AND gold_chars| / |retrieved_chars|
    
    Each span is a tuple of (file_path, start_offset, end_offset).
    Only characters from matching documents are considered overlapping.
    
    Args:
        retrieved_spans: List of (file_path, start, end) tuples from the system.
        gold_spans: List of (file_path, start, end) tuples from ground truth.
    
    Returns:
        Precision score between 0.0 and 1.0.
    """
    if not retrieved_spans:
        return 0.0

    retrieved_chars = _spans_to_char_set(retrieved_spans)
    gold_chars = _spans_to_char_set(gold_spans)

    if not retrieved_chars:
        return 0.0

    overlap = retrieved_chars & gold_chars
    return len(overlap) / len(retrieved_chars)


def char_level_recall(
    retrieved_spans: List[Tuple[str, int, int]],
    gold_spans: List[Tuple[str, int, int]],
) -> float:
    """
    Compute character-level recall.
    
    Recall = |retrieved_chars AND gold_chars| / |gold_chars|
    
    Each span is a tuple of (file_path, start_offset, end_offset).
    
    Args:
        retrieved_spans: List of (file_path, start, end) tuples from the system.
        gold_spans: List of (file_path, start, end) tuples from ground truth.
    
    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not gold_spans:
        return 1.0  # nothing to recall

    retrieved_chars = _spans_to_char_set(retrieved_spans)
    gold_chars = _spans_to_char_set(gold_spans)

    if not gold_chars:
        return 1.0

    overlap = retrieved_chars & gold_chars
    return len(overlap) / len(gold_chars)


def char_level_f1(
    retrieved_spans: List[Tuple[str, int, int]],
    gold_spans: List[Tuple[str, int, int]],
) -> float:
    """
    Compute character-level F1 score.
    
    F1 = 2 * precision * recall / (precision + recall)
    """
    p = char_level_precision(retrieved_spans, gold_spans)
    r = char_level_recall(retrieved_spans, gold_spans)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _spans_to_char_set(spans: List[Tuple[str, int, int]]) -> Set[Tuple[str, int]]:
    """
    Convert a list of spans to a set of (file_path, char_position) tuples.
    
    This allows set operations for computing overlap between retrieved and
    gold spans across multiple documents.
    """
    char_set = set()
    for file_path, start, end in spans:
        for pos in range(start, end):
            char_set.add((file_path, pos))
    return char_set


def precision_at_k(
    retrieved_spans: List[Tuple[str, int, int]],
    gold_spans: List[Tuple[str, int, int]],
    k: int,
) -> float:
    """Compute character-level precision considering only the top-k retrieved spans."""
    return char_level_precision(retrieved_spans[:k], gold_spans)


def recall_at_k(
    retrieved_spans: List[Tuple[str, int, int]],
    gold_spans: List[Tuple[str, int, int]],
    k: int,
) -> float:
    """Compute character-level recall considering only the top-k retrieved spans."""
    return char_level_recall(retrieved_spans[:k], gold_spans)


def f1_at_k(
    retrieved_spans: List[Tuple[str, int, int]],
    gold_spans: List[Tuple[str, int, int]],
    k: int,
) -> float:
    """Compute character-level F1 considering only the top-k retrieved spans."""
    return char_level_f1(retrieved_spans[:k], gold_spans)

