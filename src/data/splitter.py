"""
Train/test split for LegalBench-RAG queries.

Uses the deterministic hash-based split function specified in the assignment
to ensure results are comparable across candidates.
"""

import hashlib
from typing import List, Dict, Tuple


def assign_split(query: str) -> str:
    """
    Assign a query to either 'train' or 'test' split.
    
    This is the exact function provided in the assignment specification.
    Uses SHA-256 hash of the query text to deterministically assign splits,
    placing roughly 30% of queries in test.
    """
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"


def split_queries(queries: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Split queries into train and test sets using the assignment's split function.
    
    Returns:
        (train_queries, test_queries)
    """
    train = []
    test = []

    for q in queries:
        if assign_split(q["query"]) == "train":
            train.append(q)
        else:
            test.append(q)

    return train, test


def split_summary(queries: List[dict]) -> Dict[str, int]:
    """
    Return a summary of the train/test split sizes.
    """
    train, test = split_queries(queries)
    return {
        "total": len(queries),
        "train": len(train),
        "test": len(test),
        "train_pct": round(100 * len(train) / len(queries), 1),
        "test_pct": round(100 * len(test) / len(queries), 1),
    }

