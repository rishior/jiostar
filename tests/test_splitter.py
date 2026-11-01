"""
Tests for the train/test split module.

Verifies that the split function is deterministic and produces the expected
distribution.
"""

import pytest
from src.data.splitter import assign_split, split_queries


class TestAssignSplit:
    """Tests for the assign_split function."""

    def test_deterministic(self):
        query = "Does this agreement allow sublicensing?"
        result1 = assign_split(query)
        result2 = assign_split(query)
        assert result1 == result2

    def test_returns_valid_split(self):
        for q in ["query1", "query2", "query3", "another one"]:
            result = assign_split(q)
            assert result in ("train", "test")

    def test_whitespace_handling(self):
        result1 = assign_split("  test query  ")
        result2 = assign_split("test query")
        assert result1 == result2

    def test_approximate_distribution(self):
        queries = [f"query number {i}" for i in range(1000)]
        test_count = sum(1 for q in queries if assign_split(q) == "test")
        assert 200 < test_count < 400, f"Test count {test_count} outside expected range"


class TestSplitQueries:
    """Tests for the split_queries function."""

    def test_split_partitions(self):
        queries = [{"query": f"q{i}"} for i in range(100)]
        train, test = split_queries(queries)
        assert len(train) + len(test) == len(queries)

    def test_no_overlap(self):
        queries = [{"query": f"q{i}"} for i in range(100)]
        train, test = split_queries(queries)
        train_queries = set(q["query"] for q in train)
        test_queries = set(q["query"] for q in test)
        assert len(train_queries & test_queries) == 0

