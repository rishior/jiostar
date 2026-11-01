"""
Tests for the document chunking module.

Verifies that chunks preserve character offsets, respect size constraints,
and correctly handle edge cases.
"""

import pytest
from src.indexing.chunker import chunk_document_fixed, chunk_document_sentence, chunk_document, Chunk
from src.config import ChunkingConfig


class TestFixedChunking:
    """Tests for fixed-size chunking."""

    def test_offset_preservation(self):
        """Verify that chunk offsets correctly map back to the original text."""
        text = "This is a test document with some content. " * 50
        chunks = chunk_document_fixed(text, "test.txt", chunk_size=100, overlap=20)

        for chunk in chunks:
            assert text[chunk.start:chunk.end] == chunk.text, (
                f"Chunk {chunk.chunk_id} text does not match offset range "
                f"[{chunk.start}:{chunk.end}]"
            )

    def test_chunk_size_respected(self):
        """Verify chunks do not exceed the specified size."""
        text = "Word " * 1000
        chunks = chunk_document_fixed(text, "test.txt", chunk_size=200, overlap=50)

        for chunk in chunks:
            assert chunk.length <= 210, (
                f"Chunk {chunk.chunk_id} exceeds max size: {chunk.length}"
            )

    def test_full_coverage(self):
        """Verify that chunks cover the entire document with overlap."""
        text = "A" * 1000
        chunks = chunk_document_fixed(text, "test.txt", chunk_size=200, overlap=50)

        covered = set()
        for chunk in chunks:
            for i in range(chunk.start, chunk.end):
                covered.add(i)

        for i in range(len(text)):
            assert i in covered, f"Character at position {i} not covered by any chunk"

    def test_empty_document(self):
        """Verify empty documents produce no chunks."""
        chunks = chunk_document_fixed("", "test.txt")
        assert len(chunks) == 0

    def test_small_document(self):
        """Verify documents smaller than chunk_size produce one chunk."""
        text = "Short document with enough chars to pass minimum."
        chunks = chunk_document_fixed(text, "test.txt", chunk_size=1000, min_chunk_size=5)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_file_path_preserved(self):
        """Verify file_path is preserved in all chunks."""
        text = "Content " * 100
        chunks = chunk_document_fixed(text, "contracts/agreement.txt", chunk_size=50, min_chunk_size=5)
        for chunk in chunks:
            assert chunk.file_path == "contracts/agreement.txt"

    def test_chunk_ids_sequential(self):
        """Verify chunk IDs are sequential starting from 0."""
        text = "Content " * 200
        chunks = chunk_document_fixed(text, "test.txt", chunk_size=100, min_chunk_size=5)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i


class TestSentenceChunking:
    """Tests for sentence-based chunking."""

    def test_offset_preservation(self):
        """Verify that sentence-based chunks preserve character offsets."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunk_document_sentence(text, "test.txt", chunk_size=50, min_chunk_size=5)

        for chunk in chunks:
            assert text[chunk.start:chunk.end] == chunk.text


class TestChunkDocumentConfig:
    """Tests for the config-driven chunk_document function."""

    def test_fixed_strategy(self):
        """Verify fixed strategy is selected correctly."""
        config = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=20, min_chunk_size=5)
        text = "Content " * 100
        chunks = chunk_document(text, "test.txt", config)
        assert len(chunks) > 0
        for chunk in chunks:
            assert text[chunk.start:chunk.end] == chunk.text

    def test_sentence_strategy(self):
        """Verify sentence strategy is selected correctly."""
        config = ChunkingConfig(strategy="sentence", chunk_size=100, overlap=20, min_chunk_size=5)
        text = "First sentence. Second sentence. Third sentence. " * 10
        chunks = chunk_document(text, "test.txt", config)
        assert len(chunks) > 0
        for chunk in chunks:
            assert text[chunk.start:chunk.end] == chunk.text

