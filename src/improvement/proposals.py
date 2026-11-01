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
    
    # Strategy 1: If many wrong_document errors, the embedding model may not
    # capture legal semantics well enough
    if diagnosis.wrong_document > diagnosis.total_queries * 0.2:
        if current_config.embedding.model_name == "all-MiniLM-L6-v2":
            new_config = PipelineConfig.from_dict(current_config.to_dict())
            new_config.name = f"iter_{iteration}_mpnet"
            new_config.embedding.model_name = "all-mpnet-base-v2"
            proposals.append(Proposal(
                name="upgrade_embedding_model",
                rationale=(
                    f"Wrong document rate is {diagnosis.wrong_document}/{diagnosis.total_queries}. "
                    "Upgrading from all-MiniLM-L6-v2 to all-mpnet-base-v2 for richer "
                    "semantic representations."
                ),
                config=new_config,
                priority=10,
            ))
    
    # Strategy 2: If many wrong_section errors, chunks may be too large or
    # too small to align with relevant passages
    if diagnosis.wrong_section > diagnosis.total_queries * 0.15:
        # Try smaller chunks for finer granularity
        if current_config.chunking.chunk_size >= 512:
            new_config = PipelineConfig.from_dict(current_config.to_dict())
            new_config.name = f"iter_{iteration}_smaller_chunks"
            new_config.chunking.chunk_size = max(256, current_config.chunking.chunk_size // 2)
            new_config.chunking.overlap = new_config.chunking.chunk_size // 4
            proposals.append(Proposal(
                name="reduce_chunk_size",
                rationale=(
                    f"Wrong section rate is {diagnosis.wrong_section}/{diagnosis.total_queries}. "
                    f"Reducing chunk size from {current_config.chunking.chunk_size} to "
                    f"{new_config.chunking.chunk_size} for finer granularity."
                ),
                config=new_config,
                priority=8,
            ))
    
    # Strategy 3: If partial overlaps are common, increase overlap between chunks
    if diagnosis.partial_overlap > diagnosis.total_queries * 0.1:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_more_overlap"
        new_config.chunking.overlap = min(
            current_config.chunking.chunk_size // 2,
            current_config.chunking.overlap * 2,
        )
        proposals.append(Proposal(
            name="increase_overlap",
            rationale=(
                f"Partial overlap count is {diagnosis.partial_overlap}/{diagnosis.total_queries}. "
                f"Increasing chunk overlap from {current_config.chunking.overlap} to "
                f"{new_config.chunking.overlap} to improve span coverage."
            ),
            config=new_config,
            priority=7,
        ))
    
    # Strategy 4: If incorrect abstentions are high, lower the threshold
    if diagnosis.incorrect_abstention > 2:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_lower_threshold"
        new_config.abstention.score_threshold = max(
            0.1, current_config.abstention.score_threshold - 0.05
        )
        proposals.append(Proposal(
            name="lower_abstention_threshold",
            rationale=(
                f"Incorrect abstention count is {diagnosis.incorrect_abstention}. "
                f"Lowering score threshold from {current_config.abstention.score_threshold} "
                f"to {new_config.abstention.score_threshold}."
            ),
            config=new_config,
            priority=9,
        ))
    
    # Strategy 5: If incorrect retrieval on unanswerable is high, raise threshold
    if diagnosis.incorrect_retrieval > diagnosis.total_queries * 0.3:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_higher_threshold"
        new_config.abstention.score_threshold = min(
            0.6, current_config.abstention.score_threshold + 0.05
        )
        proposals.append(Proposal(
            name="raise_abstention_threshold",
            rationale=(
                f"Incorrect retrieval on unanswerable is {diagnosis.incorrect_retrieval}. "
                f"Raising score threshold from {current_config.abstention.score_threshold} "
                f"to {new_config.abstention.score_threshold}."
            ),
            config=new_config,
            priority=6,
        ))
    
    # Strategy 6: Enable BM25 hybrid retrieval
    if not current_config.retrieval.use_bm25:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_hybrid_bm25"
        new_config.retrieval.use_bm25 = True
        new_config.retrieval.bm25_weight = 0.3
        new_config.retrieval.dense_weight = 0.7
        proposals.append(Proposal(
            name="enable_bm25_hybrid",
            rationale=(
                "Enabling BM25 hybrid retrieval to complement dense embeddings "
                "with sparse term matching, which is effective for legal text "
                "with specific terminology."
            ),
            config=new_config,
            priority=5,
        ))
    
    # Strategy 7: Try sentence-based chunking
    if current_config.chunking.strategy == "fixed":
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_sentence_chunking"
        new_config.chunking.strategy = "sentence"
        proposals.append(Proposal(
            name="sentence_chunking",
            rationale=(
                "Switching from fixed-size to sentence-based chunking to "
                "better preserve clause boundaries in legal documents."
            ),
            config=new_config,
            priority=4,
        ))
    
    # Strategy 8: Increase top_k
    if current_config.retrieval.top_k < 15:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_more_results"
        new_config.retrieval.top_k = current_config.retrieval.top_k + 5
        proposals.append(Proposal(
            name="increase_top_k",
            rationale=(
                f"Increasing top_k from {current_config.retrieval.top_k} to "
                f"{new_config.retrieval.top_k} to improve recall by considering "
                "more candidate passages."
            ),
            config=new_config,
            priority=3,
        ))
    
    # Sort by priority (highest first)
    proposals.sort(key=lambda p: p.priority, reverse=True)
    
    return proposals
