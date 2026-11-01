"""
Tests for the abstention logic.

Verifies that the system correctly decides when to return no results.
"""

import pytest
from src.retrieval.abstention import should_abstain
from src.config import AbstentionConfig


class MockResult:
    """Mock retrieval result for testing."""
    def __init__(self, score):
        self.score = score


class TestAbstention:
    """Tests for the should_abstain function."""

    def test_abstain_on_low_scores(self):
        config = AbstentionConfig(enabled=True, score_threshold=0.3)
        results = [MockResult(0.1), MockResult(0.05)]
        assert should_abstain(results, config) is True

    def test_no_abstain_on_high_scores(self):
        config = AbstentionConfig(enabled=True, score_threshold=0.3)
        results = [MockResult(0.5), MockResult(0.2)]
        assert should_abstain(results, config) is False

    def test_abstain_on_empty_results(self):
        config = AbstentionConfig(enabled=True, score_threshold=0.3)
        assert should_abstain([], config) is True

    def test_disabled_abstention(self):
        config = AbstentionConfig(enabled=False, score_threshold=0.3)
        results = [MockResult(0.1)]
        assert should_abstain(results, config) is False

    def test_threshold_boundary(self):
        config = AbstentionConfig(enabled=True, score_threshold=0.3, min_gap=0.0)
        results = [MockResult(0.3)]
        assert should_abstain(results, config) is False

    def test_min_gap_triggers_abstention(self):
        config = AbstentionConfig(enabled=True, score_threshold=0.3, min_gap=0.1)
        results = [MockResult(0.35)]
        assert should_abstain(results, config) is True

    def test_min_gap_satisfied(self):
        config = AbstentionConfig(enabled=True, score_threshold=0.3, min_gap=0.1)
        results = [MockResult(0.5)]
        assert should_abstain(results, config) is False

    def test_reranker_threshold_abstention(self):
        config = AbstentionConfig(enabled=True, use_reranker_score=True, reranker_threshold=-1.0)
        results = [MockResult(-2.5)]
        assert should_abstain(results, config) is True

    def test_reranker_threshold_no_abstention(self):
        config = AbstentionConfig(enabled=True, use_reranker_score=True, reranker_threshold=-1.0)
        results = [MockResult(2.5)]
        assert should_abstain(results, config) is False


