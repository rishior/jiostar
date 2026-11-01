"""
Integration tests for the retrieval pipeline.

Tests the end-to-end flow from indexing through retrieval.
"""

import pytest
from pathlib import Path

from src.config import PipelineConfig
from src.retrieval.pipeline import RetrievalPipeline, RetrievalResult


class TestPipelineIntegration:
    """Integration tests for the retrieval pipeline."""

    @pytest.fixture
    def sample_documents(self, tmp_path):
        """Create sample documents for testing."""
        doc_dir = tmp_path / "corpus" / "cuad"
        doc_dir.mkdir(parents=True)

        doc1 = doc_dir / "contract1.txt"
        doc1.write_text(
            "This Agreement grants the Licensee a non-exclusive license to "
            "distribute the Content in the Territory. The Licensee may not "
            "sublicense the Content to any third party without prior written "
            "consent of the Licensor. The term of this Agreement shall be "
            "three (3) years from the Effective Date.",
            encoding="utf-8",
        )

        doc2 = doc_dir / "contract2.txt"
        doc2.write_text(
            "The parties agree that all intellectual property rights in any "
            "derivative works shall be owned exclusively by the Producer. "
            "The Distributor shall have no claim to ownership of any "
            "modifications, adaptations, or translations created under "
            "this Agreement.",
            encoding="utf-8",
        )

        return tmp_path

    def test_query_returns_results(self, sample_documents):
        """Pipeline should return results for a relevant query."""
        config = PipelineConfig(name="test")
        config.chunking.chunk_size = 200
        config.chunking.overlap = 50
        config.chunking.min_chunk_size = 10
        config.abstention.enabled = False

        pipeline = RetrievalPipeline(config)

        mock_queries = [
            {
                "query": "test",
                "snippets": [
                    {
                        "file_path": str(sample_documents / "corpus" / "cuad" / "contract1.txt"),
                        "start": 0, "end": 10, "answer": "",
                    }
                ],
            },
            {
                "query": "test2",
                "snippets": [
                    {
                        "file_path": str(sample_documents / "corpus" / "cuad" / "contract2.txt"),
                        "start": 0, "end": 10, "answer": "",
                    }
                ],
            },
        ]

        pipeline.build_index(mock_queries)
        results = pipeline.query("Who owns the intellectual property in derivative works?")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_have_offsets(self, sample_documents):
        """Each result should have valid character offsets."""
        config = PipelineConfig(name="test")
        config.chunking.chunk_size = 200
        config.chunking.overlap = 50
        config.chunking.min_chunk_size = 10
        config.abstention.enabled = False

        pipeline = RetrievalPipeline(config)

        mock_queries = [
            {
                "query": "test",
                "snippets": [
                    {
                        "file_path": str(sample_documents / "corpus" / "cuad" / "contract1.txt"),
                        "start": 0, "end": 10, "answer": "",
                    }
                ],
            },
        ]

        pipeline.build_index(mock_queries)
        results = pipeline.query("sublicense content to third party")

        for r in results:
            assert r.start >= 0
            assert r.end > r.start
            assert r.file_path is not None

