"""
Change proposal generator for the self-improvement loop.

Analyzes diagnosis results and proposes concrete configuration changes
to improve the pipeline. Each proposal includes the rationale, the
specific parameter changes, priority, and proposal origin.
"""

from dataclasses import dataclass
from typing import List, Optional
from src.config import PipelineConfig
from src.improvement.diagnosis import DiagnosisResult


@dataclass
class Proposal:
    """A proposed change to the pipeline configuration."""
    name: str
    rationale: str
    config: PipelineConfig
    priority: int = 0  # higher = try first
    proposed_by: str = "diagnosis"  # diagnosis, parameter_search, or rule


def generate_proposals(
    current_config: PipelineConfig,
    diagnosis: DiagnosisResult,
    iteration: int,
) -> List[Proposal]:
    """
    Generate a list of configuration change proposals based on diagnosis and current config.
    The proposals are ordered by priority.
    """
    proposals = []

    # Priority 1: Strip query prefix to combat preamble dominance
    if diagnosis.preamble_at_rank_1 > 15 and not current_config.retrieval.strip_query_prefix:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_strip_query_prefix"
        new_config.retrieval.strip_query_prefix = True
        proposals.append(Proposal(
            name="strip_query_prefix",
            rationale=(
                f"Diagnosis shows {diagnosis.preamble_at_rank_1} queries retrieve preamble offset 0 at rank 1. "
                "The benchmark query includes the agreement title, which falsely matches the contract header. "
                "Stripping the contract prefix isolates the substantive legal clause question."
            ),
            config=new_config,
            priority=100,
            proposed_by="diagnosis",
        ))

    # Priority 2: Heading aware chunking
    if diagnosis.wrong_section > 15 and not current_config.chunking.heading_aware:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_heading_aware_chunking"
        new_config.chunking.heading_aware = True
        proposals.append(Proposal(
            name="heading_aware_chunking",
            rationale=(
                f"Diagnosis shows {diagnosis.wrong_section} queries find the correct document but wrong section. "
                "Prepending active legal headers to chunk embedding representations provides structural context."
            ),
            config=new_config,
            priority=95,
            proposed_by="diagnosis",
        ))

    # Priority 3: BM25 with Reciprocal Rank Fusion
    if not current_config.retrieval.use_bm25:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_bm25_rrf_hybrid"
        new_config.retrieval.use_bm25 = True
        new_config.retrieval.use_rrf = True
        new_config.retrieval.rrf_k = 60
        proposals.append(Proposal(
            name="bm25_rrf_hybrid",
            rationale=(
                "Adding BM25 lexical retrieval fused via Reciprocal Rank Fusion (k=60) complements semantic search "
                "with exact keyword matching for specific contractual phrases and section titles."
            ),
            config=new_config,
            priority=90,
            proposed_by="parameter_search",
        ))

    # Priority 4: Preamble downweighting
    if diagnosis.preamble_at_rank_1 > 5 and not current_config.retrieval.downweight_preamble:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_preamble_downweight"
        new_config.retrieval.downweight_preamble = True
        new_config.retrieval.preamble_weight = 0.5
        proposals.append(Proposal(
            name="preamble_downweight",
            rationale=(
                f"Residual preamble dominance remains at {diagnosis.preamble_at_rank_1} queries. "
                "Applying a 0.5 position prior weight downweights offset 0 preamble chunks."
            ),
            config=new_config,
            priority=85,
            proposed_by="diagnosis",
        ))

    # Priority 5: Chunk size tuning
    if current_config.chunking.chunk_size == 512:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_chunk_size_384"
        new_config.chunking.chunk_size = 384
        new_config.chunking.overlap = 96
        proposals.append(Proposal(
            name="tune_chunk_size_384",
            rationale=(
                f"Partial overlap count is {diagnosis.partial_overlap}. "
                "Tuning chunk size to 384 characters closely targets the 317 character median gold span "
                "to improve character precision."
            ),
            config=new_config,
            priority=80,
            proposed_by="parameter_search",
        ))

    # Priority 6: Abstention threshold tuning for in corpus unanswerables
    if diagnosis.incorrect_retrieval_in_corpus > 5:
        # Candidate score thresholds to test
        for thresh in [0.42, 0.46, 0.50]:
            if abs(current_config.abstention.score_threshold - thresh) > 0.01:
                new_config = PipelineConfig.from_dict(current_config.to_dict())
                new_config.name = f"iter_{iteration}_score_threshold_{thresh}"
                new_config.abstention.score_threshold = thresh
                proposals.append(Proposal(
                    name=f"tune_score_threshold_{thresh}",
                    rationale=(
                        f"Diagnosis shows {diagnosis.incorrect_retrieval_in_corpus} in corpus unanswerable queries "
                        f"were incorrectly retrieved. Raising score threshold to {thresh} filters absent clauses."
                    ),
                    config=new_config,
                    priority=75,
                    proposed_by="diagnosis",
                ))
                break

    # Priority 7: Margin threshold for ambiguous near misses
    if current_config.abstention.margin_threshold == 0.0 and diagnosis.incorrect_retrieval_in_corpus > 5:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_margin_threshold_0.02"
        new_config.abstention.margin_threshold = 0.02
        proposals.append(Proposal(
            name="tune_margin_threshold_0.02",
            rationale=(
                "Enforcing a minimum score margin between rank 1 and rank 2 candidates rejects ambiguous "
                "in corpus queries where no single clause stands out with high confidence."
            ),
            config=new_config,
            priority=70,
            proposed_by="parameter_search",
        ))

    # Priority 8: Calibrated cross encoder reranker
    if not current_config.retrieval.use_reranker:
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_cross_encoder_reranker"
        new_config.retrieval.use_reranker = True
        new_config.retrieval.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        new_config.retrieval.rerank_top_k = 25
        proposals.append(Proposal(
            name="cross_encoder_reranker",
            rationale=(
                "Bi encoder cosine similarity exhibits score compression on legal text. "
                "A cross encoder computes full cross attention between query and candidate passages "
                "with sigmoid score calibration."
            ),
            config=new_config,
            priority=65,
            proposed_by="parameter_search",
        ))

    # Priority 9: Clause boundary chunking
    if current_config.chunking.strategy != "clause":
        new_config = PipelineConfig.from_dict(current_config.to_dict())
        new_config.name = f"iter_{iteration}_clause_chunking"
        new_config.chunking.strategy = "clause"
        new_config.chunking.chunk_size = 400
        new_config.chunking.overlap = 80
        proposals.append(Proposal(
            name="clause_chunking",
            rationale=(
                "Switching from fixed chunking to clause boundary snapped chunking aligns passage boundaries "
                "with legal paragraph and sentence delimiters."
            ),
            config=new_config,
            priority=60,
            proposed_by="diagnosis",
        ))

    # Sort by priority descending
    proposals.sort(key=lambda p: p.priority, reverse=True)
    return proposals
