"""
Pipeline configuration management.

All pipeline parameters are stored in YAML files so the self-improvement loop
can propose and swap configurations programmatically.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"          # "fixed", "sentence", "clause"
    chunk_size: int = 512            # characters
    overlap: int = 128               # characters of overlap between chunks
    min_chunk_size: int = 50         # discard chunks smaller than this
    heading_aware: bool = False      # whether to extract clause headings and prepend to embedding text


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True


@dataclass
class RetrievalConfig:
    top_k: int = 10                  # number of candidates to retrieve
    use_routing: bool = True         # whether to route queries to candidate contract
    strip_query_prefix: bool = False # whether to strip agreement title prefix from search query
    use_bm25: bool = False           # whether to include BM25 scores
    bm25_weight: float = 0.3         # weight for BM25 in hybrid scoring
    dense_weight: float = 0.7        # weight for dense embeddings
    use_rrf: bool = True             # whether to fuse dense and BM25 via Reciprocal Rank Fusion
    rrf_k: int = 60                  # RRF constant k
    use_reranker: bool = False       # whether to apply cross-encoder reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 30           # number of candidates passed to reranker
    downweight_preamble: bool = False # downweight chunks starting at offset 0
    preamble_weight: float = 0.8     # score multiplier for offset 0 chunks


@dataclass
class AbstentionConfig:
    enabled: bool = True
    score_threshold: float = 0.3     # minimum score to return a result (for dense/hybrid)
    min_gap: float = 0.0             # minimum gap between top score and threshold
    margin_threshold: float = 0.0    # minimum margin between top-1 and top-2 candidate scores
    use_reranker_score: bool = False # whether to threshold on cross-encoder score
    reranker_threshold: float = -1.0 # minimum cross-encoder logit score to return result


@dataclass
class PipelineConfig:
    name: str = "baseline"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            name=d.get("name", "unnamed"),
            chunking=ChunkingConfig(**d.get("chunking", {})),
            embedding=EmbeddingConfig(**d.get("embedding", {})),
            retrieval=RetrievalConfig(**d.get("retrieval", {})),
            abstention=AbstentionConfig(**d.get("abstention", {})),
        )

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
CORPUS_DIR = DATA_DIR / "corpus" / "cuad"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
INDEX_DIR = DATA_DIR / "index"

