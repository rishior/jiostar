"""
Unit tests for v2 enhancements:
1. strip_contract_prefix logic
2. tokenize_bm25 tokenization and stopword removal
3. extract_headings regex parser
4. heading-aware chunking with verbatim offset preservation
5. margin_threshold in abstention decision
"""

import pytest
from src.retrieval.pipeline import strip_contract_prefix, tokenize_bm25
from src.indexing.chunker import (
    extract_headings,
    chunk_document,
    chunk_document_fixed,
    chunk_document_clause,
)
from src.config import AbstentionConfig, ChunkingConfig
from src.retrieval.abstention import should_abstain


class TestStripContractPrefix:
    """Tests for query prefix stripping."""

    def test_semicolon_prefix(self):
        query = "Consider the Agreement between Party A and Party B; what is the governing law?"
        stripped = strip_contract_prefix(query)
        assert stripped == "what is the governing law?"

    def test_colon_prefix(self):
        query = "In the Duos Technologies Strategic Alliance Agreement: does either party have audit rights?"
        stripped = strip_contract_prefix(query)
        assert stripped == "does either party have audit rights?"

    def test_according_to_prefix(self):
        query = "According to the Agreement by and between X and Y, what is the term length?"
        stripped = strip_contract_prefix(query)
        assert stripped == "what is the term length?"

    def test_no_prefix(self):
        query = "What is the governing law of this contract?"
        stripped = strip_contract_prefix(query)
        assert stripped == query

    def test_short_prefix_preserved(self):
        query = "Note: this is important"
        stripped = strip_contract_prefix(query)
        # Should not strip if remainder is too short or doesn't match contract pattern
        assert "this is important" in stripped or query == stripped


class TestTokenizeBM25:
    """Tests for BM25 tokenization and stopword filtering."""

    def test_basic_tokenization(self):
        text = "The Confidential Information shall remain the exclusive property of Disclosing Party."
        tokens = tokenize_bm25(text)
        assert "confidential" in tokens
        assert "information" in tokens
        assert "exclusive" in tokens
        assert "property" in tokens
        # Common English/legal stopwords should be filtered
        assert "the" not in tokens
        assert "shall" not in tokens
        assert "of" not in tokens

    def test_empty_string(self):
        assert tokenize_bm25("") == []

    def test_punctuation_stripping(self):
        text = "Section 14.2(b): Indemnification & Liability!"
        tokens = tokenize_bm25(text)
        assert "indemnification" in tokens
        assert "liability" in tokens
        assert "&" not in tokens


class TestExtractHeadings:
    """Tests for contractual heading extraction."""

    def test_article_heading(self):
        text = "PREAMBLE\n\nARTICLE I\nDEFINITIONS\n\nSection 1.1 Terms..."
        headings = extract_headings(text)
        assert len(headings) >= 1
        heading_texts = [h[2] for h in headings]
        assert any("ARTICLE" in h for h in heading_texts)

    def test_section_numbered_heading(self):
        text = "Some text.\n\n14. Governing Law and Jurisdiction\nThis agreement is governed by Delaware law."
        headings = extract_headings(text)
        heading_texts = [h[2] for h in headings]
        assert any("Governing Law" in h for h in heading_texts)

    def test_empty_document(self):
        assert extract_headings("") == []


class TestHeadingAwareChunking:
    """Tests for heading-aware chunking."""

    def test_heading_in_embedding_text_not_offsets(self):
        doc = (
            "AGREEMENT\n\n"
            "ARTICLE 5 - INDEMNIFICATION\n\n"
            "Each party shall defend, indemnify, and hold harmless the other party from claims."
        )
        cfg = ChunkingConfig(
            strategy="fixed",
            chunk_size=120,
            overlap=30,
            heading_aware=True,
        )
        chunks = chunk_document(doc, "test.txt", cfg)
        assert len(chunks) > 0
        for c in chunks:
            # Raw text sliced by start:end must match document verbatim
            assert doc[c.start:c.end] == c.text
            # If heading is attached, embedding_text should include it
            if c.heading:
                assert c.heading in c.embedding_text
                assert c.embedding_text.startswith(c.heading)


class TestMarginThresholdAbstention:
    """Tests for margin threshold abstention."""

    def test_margin_triggers_abstention(self):
        cfg = AbstentionConfig(
            enabled=True,
            score_threshold=0.3,
            margin_threshold=0.05,
        )
        # Scores are very close (0.60 vs 0.58) -> margin = 0.02 < 0.05
        assert should_abstain([0.60, 0.58], cfg) is True

    def test_margin_satisfied(self):
        cfg = AbstentionConfig(
            enabled=True,
            score_threshold=0.3,
            margin_threshold=0.05,
        )
        # Scores have clear separation (0.60 vs 0.50) -> margin = 0.10 >= 0.05
        assert should_abstain([0.60, 0.50], cfg) is False

    def test_single_result_margin(self):
        cfg = AbstentionConfig(
            enabled=True,
            score_threshold=0.3,
            margin_threshold=0.05,
        )
        # Single result above threshold should not be rejected by margin
        assert should_abstain([0.60], cfg) is False
