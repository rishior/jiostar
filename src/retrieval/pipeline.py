"""
Retrieval pipeline.

Orchestrates the full retrieval flow: takes a query, embeds it, searches the
index, optionally combines with BM25 scores via RRF, routes to target documents,
reranks candidates with a cross-encoder, applies abstention logic, and
returns ranked results with character offsets.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import re

import numpy as np

from src.config import PipelineConfig, INDEX_DIR
from src.indexing.embedder import Embedder
from src.indexing.index import VectorIndex
from src.indexing.chunker import Chunk, chunk_document
from src.retrieval.abstention import should_abstain
from src.retrieval.router import DocumentRouter
from src.retrieval.ranker import CrossEncoderReranker
from src.data.loader import load_document, load_benchmark, get_referenced_files


COMMON_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
    # Recurring contract filler terms
    "agreement", "contract", "parties", "party", "hereto", "herein", "hereof", "thereof",
    "shall", "may", "will"
}


def strip_contract_prefix(query: str) -> str:
    """
    Strip contract-identifying prefix (e.g. 'Consider the X between A and B;') using query text only.
    Returns the substantive legal question.
    """
    # Semicolon separator (LegalBench-RAG standard pattern)
    if ";" in query:
        parts = query.split(";", 1)
        prefix = parts[0].strip().lower()
        if prefix.startswith("consider") or "between" in prefix or "agreement" in prefix or "contract" in prefix:
            remaining = parts[1].strip()
            if remaining:
                return remaining

    # Colon separator (e.g. "In the X Agreement: question")
    if ":" in query:
        parts = query.split(":", 1)
        prefix = parts[0].strip().lower()
        if "agreement" in prefix or "contract" in prefix or prefix.startswith("in the") or prefix.startswith("under the"):
            remaining = parts[1].strip()
            if remaining:
                return remaining

    # According to / Under / Pursuant to pattern
    m = re.match(r'^(?:according to|under|pursuant to)\s+the\s+.*?(?:agreement|contract).*?[,\-]\s*(.*)$', query, re.IGNORECASE)
    if m:
        remaining = m.group(1).strip()
        if remaining:
            return remaining

    return query


def tokenize_bm25(text: str) -> List[str]:
    """
    Extract meaningful word tokens using regex, filtering short tokens and stopwords.
    """
    tokens = re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())
    return [t for t in tokens if t not in COMMON_STOPWORDS]


@dataclass
class RetrievalResult:
    """A single retrieval result with its source location and score."""
    text: str
    file_path: str
    start: int
    end: int
    score: float
    chunk_id: int


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline.
    
    Given a configuration, builds or loads an index over the corpus, then
    answers queries by retrieving and ranking relevant passages.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.embedder = Embedder(config.embedding)
        self.index: Optional[VectorIndex] = None
        self.bm25 = None
        self.bm25_chunks: List[dict] = []
        self.router: Optional[DocumentRouter] = None
        self.reranker: Optional[CrossEncoderReranker] = None

    def _init_components(self):
        """Initialize document router and reranker based on configuration and indexed chunks."""
        if self.index is not None and self.config.retrieval.use_routing:
            doc_files = list(set(c["file_path"] for c in self.index.chunks))
            self.router = DocumentRouter(doc_files)
        else:
            self.router = None

        if self.config.retrieval.use_reranker:
            self.reranker = CrossEncoderReranker(self.config.retrieval.reranker_model)
        else:
            self.reranker = None

    def build_index(self, queries: Optional[List[dict]] = None):
        """
        Build the vector index over the corpus documents referenced by the queries.
        
        If queries is None, loads them from the benchmark file.
        Uses cached index on disk if available for the given embedding and chunking parameters.
        """
        heading_aware = getattr(self.config.chunking, "heading_aware", False)
        sig = f"{self.config.embedding.model_name}_{self.config.chunking.strategy}_{self.config.chunking.chunk_size}_{self.config.chunking.overlap}_{heading_aware}"
        safe_sig = re.sub(r'[^a-zA-Z0-9_\-]', '_', sig)
        cached_index_dir = INDEX_DIR / safe_sig

        # Check if cache exists
        if (cached_index_dir / "faiss.index").exists() and (cached_index_dir / "chunks.json").exists():
            print(f"Loading cached index from {cached_index_dir}...")
            self.load_index(cached_index_dir)
            return

        # Check default baseline index in INDEX_DIR
        if (INDEX_DIR / "faiss.index").exists() and (INDEX_DIR / "chunks.json").exists() and \
           self.config.embedding.model_name == "all-MiniLM-L6-v2" and self.config.chunking.strategy == "fixed" and \
           self.config.chunking.chunk_size == 512 and self.config.chunking.overlap == 128 and not heading_aware:
            print(f"Loading index from {INDEX_DIR}...")
            self.load_index(INDEX_DIR)
            cached_index_dir.mkdir(parents=True, exist_ok=True)
            self.save_index(cached_index_dir)
            return

        if queries is None:
            queries = load_benchmark()

        referenced_files = get_referenced_files(queries)
        print(f"Building index over {len(referenced_files)} documents...")

        all_chunks: List[Chunk] = []
        for file_path in referenced_files:
            try:
                text = load_document(file_path)
                chunks = chunk_document(text, file_path, self.config.chunking)
                all_chunks.extend(chunks)
            except FileNotFoundError:
                print(f"  Warning: document not found: {file_path}")

        print(f"Generated {len(all_chunks)} chunks")

        # Generate embeddings (use enriched embedding_text when available)
        chunk_texts = [
            c.embedding_text if getattr(c, "embedding_text", None) else c.text
            for c in all_chunks
        ]
        print("Generating embeddings...")
        embeddings = self.embedder.embed_texts(chunk_texts)

        # Build FAISS index
        self.index = VectorIndex(dimension=embeddings.shape[1])
        self.index.add(embeddings, all_chunks)

        # Save to cache
        cached_index_dir.mkdir(parents=True, exist_ok=True)
        self.save_index(cached_index_dir)

        # Build BM25 index if configured
        if self.config.retrieval.use_bm25:
            self._build_bm25(all_chunks)

        self._init_components()
        print(f"Index built with {self.index.size} vectors")

    def _build_bm25(self, chunks: List[Chunk]):
        """Build a BM25 index over the chunk texts using regex tokenization."""
        from rank_bm25 import BM25Okapi
        from dataclasses import asdict

        tokenized = [
            tokenize_bm25(c.embedding_text if getattr(c, "embedding_text", None) else c.text)
            for c in chunks
        ]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_chunks = [asdict(c) for c in chunks]

    def save_index(self, path: Optional[Path] = None):
        """Save the current index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        self.index.save(path)

    def load_index(self, path: Optional[Path] = None):
        """Load a previously saved index from disk."""
        self.index = VectorIndex.load(path)

        # Rebuild BM25 if configured
        if self.config.retrieval.use_bm25:
            from rank_bm25 import BM25Okapi

            self.bm25_chunks = self.index.chunks
            tokenized = [
                tokenize_bm25(c.get("embedding_text") or c["text"])
                for c in self.bm25_chunks
            ]
            self.bm25 = BM25Okapi(tokenized)

        self._init_components()

    def query(self, question: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Answer a question by retrieving relevant passages.
        
        Returns a list of RetrievalResult objects sorted by relevance score,
        or an empty list if the system decides to abstain (no relevant passage).
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")

        req_k = top_k if top_k is not None else self.config.retrieval.top_k

        # Document routing
        filter_file: Optional[str] = None
        if self.config.retrieval.use_routing and self.router is not None:
            target_file, conf, named = self.router.route(question)
            if named and target_file is None:
                # Question explicitly named an agreement not present in the indexed corpus
                return []
            if target_file is not None:
                filter_file = target_file

        # Strip contract prefix for search if configured
        if getattr(self.config.retrieval, "strip_query_prefix", False):
            search_query = strip_contract_prefix(question)
        else:
            search_query = question

        # Candidate retrieval size
        candidate_k = (
            self.config.retrieval.rerank_top_k
            if self.config.retrieval.use_reranker
            else max(req_k * 2, 20)
        )

        # Dense retrieval
        query_embedding = self.embedder.embed_query(search_query)
        dense_results = self.index.search(
            query_embedding,
            top_k=candidate_k,
            filter_file=filter_file,
        )

        # Combine with BM25 if configured
        if self.config.retrieval.use_bm25 and self.bm25 is not None:
            results = self._hybrid_search(search_query, dense_results, candidate_k, filter_file=filter_file)
        else:
            results = [
                RetrievalResult(
                    text=chunk["text"],
                    file_path=chunk["file_path"],
                    start=chunk["start"],
                    end=chunk["end"],
                    score=score,
                    chunk_id=chunk["chunk_id"],
                )
                for chunk, score in dense_results
            ]

        # Rerank with CrossEncoder if configured
        if self.config.retrieval.use_reranker and self.reranker is not None and results:
            results = self.reranker.rerank(search_query, results, top_k=req_k)
        else:
            results = results[:req_k]

        # Apply position prior downweighting preamble chunks (offset 0)
        if getattr(self.config.retrieval, "downweight_preamble", False):
            preamble_weight = getattr(self.config.retrieval, "preamble_weight", 0.8)
            for r in results:
                if r.start == 0 or r.chunk_id == 0:
                    r.score = r.score * preamble_weight
            results.sort(key=lambda x: x.score, reverse=True)

        # Apply abstention
        if self.config.abstention.enabled and results:
            if should_abstain(results, self.config.abstention):
                return []

        return results

    def _hybrid_search(
        self,
        question: str,
        dense_results: List[Tuple[dict, float]],
        top_k: int,
        filter_file: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Combine dense and BM25 retrieval scores using Reciprocal Rank Fusion (RRF)
        or normalized linear combination.
        """
        query_tokens = tokenize_bm25(question)
        if not query_tokens:
            query_tokens = question.lower().split()

        bm25_scores = self.bm25.get_scores(query_tokens)

        # Build mapping of candidates
        use_rrf = getattr(self.config.retrieval, "use_rrf", True)
        rrf_k = getattr(self.config.retrieval, "rrf_k", 60)
        dense_weight = self.config.retrieval.dense_weight
        bm25_weight = self.config.retrieval.bm25_weight

        if use_rrf:
            # Reciprocal Rank Fusion
            # 1. Dense ranks
            dense_ranks = {}
            for rank, (chunk, _) in enumerate(dense_results):
                key = (chunk["file_path"], chunk["start"], chunk["end"])
                dense_ranks[key] = (rank, chunk)

            # 2. Filter BM25 candidates to target file if routed
            bm25_ranked = []
            for i, score in enumerate(bm25_scores):
                if i < len(self.bm25_chunks):
                    c = self.bm25_chunks[i]
                    if filter_file is not None and c["file_path"] != filter_file:
                        continue
                    bm25_ranked.append((c, float(score)))

            bm25_ranked.sort(key=lambda x: x[1], reverse=True)
            bm25_ranks = {}
            for rank, (chunk, _) in enumerate(bm25_ranked[:top_k * 2]):
                key = (chunk["file_path"], chunk["start"], chunk["end"])
                bm25_ranks[key] = (rank, chunk)

            all_keys = set(dense_ranks.keys()) | set(bm25_ranks.keys())
            results = []
            for key in all_keys:
                chunk = dense_ranks[key][1] if key in dense_ranks else bm25_ranks[key][1]
                dense_r = dense_ranks[key][0] if key in dense_ranks else 200
                bm25_r = bm25_ranks[key][0] if key in bm25_ranks else 200

                rrf_score = (
                    dense_weight / (rrf_k + dense_r + 1) +
                    bm25_weight / (rrf_k + bm25_r + 1)
                )

                results.append(RetrievalResult(
                    text=chunk["text"],
                    file_path=chunk["file_path"],
                    start=chunk["start"],
                    end=chunk["end"],
                    score=float(rrf_score),
                    chunk_id=chunk["chunk_id"],
                ))

            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]

        else:
            # Linear normalized fusion
            bm25_max = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
            bm25_scores_norm = bm25_scores / bm25_max

            scored = {}
            for chunk, dense_score in dense_results:
                key = (chunk["file_path"], chunk["start"], chunk["end"])
                scored[key] = {
                    "chunk": chunk,
                    "dense_score": dense_score,
                    "bm25_score": 0.0,
                }

            for i, bm25_score_norm in enumerate(bm25_scores_norm):
                if i < len(self.bm25_chunks):
                    c = self.bm25_chunks[i]
                    if filter_file is not None and c["file_path"] != filter_file:
                        continue
                    key = (c["file_path"], c["start"], c["end"])
                    if key in scored:
                        scored[key]["bm25_score"] = float(bm25_score_norm)

            results = []
            for key, data in scored.items():
                combined = dense_weight * data["dense_score"] + bm25_weight * data["bm25_score"]
                results.append(RetrievalResult(
                    text=data["chunk"]["text"],
                    file_path=data["chunk"]["file_path"],
                    start=data["chunk"]["start"],
                    end=data["chunk"]["end"],
                    score=combined,
                    chunk_id=data["chunk"]["chunk_id"],
                ))

            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]
