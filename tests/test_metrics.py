"""
Tests for the evaluation metrics module.

Verifies character-level precision, recall, and F1 calculations against
known examples.
"""

import pytest
from src.evaluation.metrics import (
    char_level_precision,
    char_level_recall,
    char_level_f1,
    precision_at_k,
    recall_at_k,
)


class TestCharLevelPrecision:
    """Tests for character-level precision."""

    def test_perfect_match(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 10, 20)]
        assert char_level_precision(retrieved, gold) == 1.0

    def test_no_overlap(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 30, 40)]
        assert char_level_precision(retrieved, gold) == 0.0

    def test_partial_overlap(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 15, 25)]
        assert char_level_precision(retrieved, gold) == 0.5

    def test_empty_retrieved(self):
        gold = [("doc.txt", 10, 20)]
        assert char_level_precision([], gold) == 0.0

    def test_different_documents(self):
        gold = [("doc1.txt", 10, 20)]
        retrieved = [("doc2.txt", 10, 20)]
        assert char_level_precision(retrieved, gold) == 0.0

    def test_superset_retrieval(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 5, 25)]
        assert char_level_precision(retrieved, gold) == 0.5


class TestCharLevelRecall:
    """Tests for character-level recall."""

    def test_perfect_match(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 10, 20)]
        assert char_level_recall(retrieved, gold) == 1.0

    def test_no_overlap(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 30, 40)]
        assert char_level_recall(retrieved, gold) == 0.0

    def test_partial_recall(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 15, 25)]
        assert char_level_recall(retrieved, gold) == 0.5

    def test_empty_gold(self):
        assert char_level_recall([("doc.txt", 10, 20)], []) == 1.0

    def test_superset_retrieval(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 5, 25)]
        assert char_level_recall(retrieved, gold) == 1.0

    def test_multiple_gold_spans(self):
        gold = [("doc.txt", 10, 20), ("doc.txt", 30, 40)]
        retrieved = [("doc.txt", 10, 20)]
        assert char_level_recall(retrieved, gold) == 0.5


class TestCharLevelF1:
    """Tests for character-level F1 score."""

    def test_perfect_match(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 10, 20)]
        assert char_level_f1(retrieved, gold) == 1.0

    def test_no_overlap(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 30, 40)]
        assert char_level_f1(retrieved, gold) == 0.0

    def test_f1_between_precision_and_recall(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [("doc.txt", 15, 25)]
        f1 = char_level_f1(retrieved, gold)
        assert abs(f1 - 0.5) < 1e-10


class TestMetricsAtK:
    """Tests for @k metric variants."""

    def test_precision_at_k(self):
        gold = [("doc.txt", 10, 20)]
        retrieved = [
            ("doc.txt", 10, 20),
            ("doc.txt", 100, 200),
        ]
        p1 = precision_at_k(retrieved, gold, k=1)
        p2 = precision_at_k(retrieved, gold, k=2)
        assert p1 == 1.0
        assert p2 < 1.0

    def test_recall_at_k(self):
        gold = [("doc.txt", 10, 20), ("doc.txt", 30, 40)]
        retrieved = [
            ("doc.txt", 10, 20),
            ("doc.txt", 30, 40),
        ]
        r1 = recall_at_k(retrieved, gold, k=1)
        r2 = recall_at_k(retrieved, gold, k=2)
        assert r1 == 0.5
        assert r2 == 1.0

