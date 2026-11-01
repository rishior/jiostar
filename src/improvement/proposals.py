"""
Change proposal generator for the self-improvement loop.

Analyzes diagnosis results and proposes concrete configuration changes
to improve the pipeline. Each proposal includes the rationale and the
specific parameter changes.
"""

from dataclasses import dataclass
from typing import List, Optional
from src.config import PipelineConfig, ChunkingConfig, EmbeddingConfig, RetrievalConfig, AbstentionConfig
from src.improvement.diagnosis import DiagnosisResult


@dataclass
class Proposal:
    """A proposed change to the pipeline configuration."""
    name: str
    rationale: str
    config: PipelineConfig
    priority: int = 0  # higher = try first


def generate_proposals(
    current_config: PipelineConfig,
    diagnosis: DiagnosisResult,
    iteration: int,
) -> List[Proposal]:
    """
    Generate a list of configuration change proposals based on the diagnosis.
    
    The proposals are ordered by priority. The generator uses the pattern of
    failures to decide what to try next.
    """
    proposals = []

    # Priority 1: Document routing
    # If document routing is not yet enabled and wrong_document errors exist
    if not current_config.retrieval.use_routing:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_document_routing"
        new_config.retrieval.use_routing = True
        proposals.append(Proposal(
            name="enable_document_routing",
            rationale=(
                f"Wrong document rate is {diagnosis.wrong_document}/{diagnosis.total_queries}. "
                "Enabling entity-aware document routing constrains search to the referenced contract, "
                "eliminating cross-contract false positive noise."
            ),
            config=new_config,
            priority=100,
        ))

    # Priority 2: Clause-aware chunking
    # If chunking strategy is still fixed, switch to clause-aware
    if current_config.chunking.strategy != "clause":
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_clause_chunking"
        new_config.chunking.strategy = "clause"
        new_config.chunking.chunk_size = 400
        new_config.chunking.overlap = 80
        proposals.append(Proposal(
            name="enable_clause_chunking",
            rationale=(
                f"Fixed chunking creates arbitrary cuts across sentences. "
                f"Switching to clause-aware boundary snapping (target 400 chars) matches "
                f"the median gold answer span (317 chars) and respects legal section titles."
            ),
            config=new_config,
            priority=90,
        ))

    # Priority 3: Cross-encoder reranking
    # If reranker is not enabled, add cross-encoder
    if not current_config.retrieval.use_reranker:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_cross_encoder_reranker"
        new_config.retrieval.use_reranker = True
        new_config.retrieval.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        new_config.retrieval.rerank_top_k = 30
        proposals.append(Proposal(
            name="enable_cross_encoder_reranker",
            rationale=(
                "Bi-encoder cosine similarity exhibits score compression on legal boilerplate. "
                "A cross-encoder computes full query-passage cross-attention to sharply separate "
                "relevant contractual terms from irrelevant boilerplate."
            ),
            config=new_config,
            priority=85,
        ))

    # Priority 4: BM25 hybrid retrieval
    if not current_config.retrieval.use_bm25:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_hybrid_bm25"
        new_config.retrieval.use_bm25 = True
        new_config.retrieval.bm25_weight = 0.3
        new_config.retrieval.dense_weight = 0.7
        proposals.append(Proposal(
            name="enable_bm25_hybrid",
            rationale=(
                "Enabling BM25 hybrid retrieval complements dense embeddings with exact term matching, "
                "which is effective for specific legal clause titles and section numbers."
            ),
            config=new_config,
            priority=80,
        ))

    # Priority 5: Increase chunk overlap to reduce partial overlap errors
    if current_config.chunking.overlap < 256:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_overlap_256"
        new_config.chunking.overlap = 256
        proposals.append(Proposal(
            name="increase_chunk_overlap",
            rationale=(
                f"Partial overlap count is {diagnosis.partial_overlap}/{diagnosis.total_queries}. "
                "Increasing chunk overlap from 128 to 256 characters (50% overlap) prevents clauses "
                "from being truncated at chunk boundaries."
            ),
            config=new_config,
            priority=70,
        ))

    # Priority 6: Fine-tuning abstention threshold on dense score
    if not current_config.abstention.use_reranker_score:
        for thresh in [0.35, 0.40]:
            if abs(current_config.abstention.score_threshold - thresh) > 0.01:
                new_config = PipelineConfig.from_dict(current_config.to_dict())
                new_config.name = f"iter_{iteration}_thresh_{thresh}"
                new_config.abstention.score_threshold = thresh
                proposals.append(Proposal(
                    name=f"tune_dense_threshold_{thresh}",
                    rationale=(
                        f"Tuning dense cosine abstention threshold from {current_config.abstention.score_threshold} "
                        f"to {thresh} to reject borderline unanswerable queries."
                    ),
                    config=new_config,
                    priority=65,
                ))

    # Priority 7: Calibrated reranker abstention
    if current_config.retrieval.use_reranker and not current_config.abstention.use_reranker_score:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_calibrated_abstention"
        new_config.abstention.use_reranker_score = True
        new_config.abstention.reranker_threshold = -1.5
        proposals.append(Proposal(
            name="calibrate_reranker_abstention",
            rationale=(
                f"Incorrect retrieval on unanswerable is {diagnosis.incorrect_retrieval}. "
                "Calibrating abstention using the cross-encoder logit threshold (-1.5) provides "
                "sharp separation between relevant answers and missing clauses."
            ),
            config=new_config,
            priority=75,
        ))

    # Priority 6: Fine-tuning reranker threshold
    if current_config.abstention.use_reranker_score:
        for thresh in [-2.0, -1.0, -0.5]:
            if abs(current_config.abstention.reranker_threshold - thresh) > 0.1:
                new_config = PipelineConfig.from_dict(current_config.to_dict())
                new_config.name = f"iter_{iteration}_thresh_{thresh}"
                new_config.abstention.reranker_threshold = thresh
                proposals.append(Proposal(
                    name=f"tune_reranker_threshold_{thresh}",
                    rationale=(
                        f"Tuning reranker abstention threshold to {thresh} to optimize the trade-off "
                        f"between false rejection of answerable queries and rejection of unanswerable queries."
                    ),
                    config=new_config,
                    priority=60,
                ))

    # Priority 7: Chunk size tuning
    if current_config.chunking.strategy == "clause" and current_config.chunking.chunk_size == 400:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_clause_300"
        new_config.chunking.chunk_size = 300
        new_config.chunking.overlap = 60
        proposals.append(Proposal(
            name="tune_clause_size_300",
            rationale="Tuning target clause chunk size to 300 chars to closely match the 317-char median gold span.",
            config=new_config,
            priority=50,
        ))

    # Sort by priority descending
    proposals.sort(key=lambda p: p.priority, reverse=True)
    return proposals
