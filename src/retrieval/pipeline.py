"""
Retrieval pipeline.

Orchestrates the full retrieval flow: takes a query, embeds it, searches the
index, optionally combines with BM25 scores, routes to target documents,
reranks candidates with a cross-encoder, applies abstention logic, and
returns ranked results with character offsets.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

from src.config import PipelineConfig, INDEX_DIR
from src.indexing.embedder import Embedder
from src.indexing.index import VectorIndex
from src.indexing.chunker import Chunk, chunk_document
from src.retrieval.abstention import should_abstain
from src.retrieval.router import DocumentRouter
from src.retrieval.ranker import CrossEncoderReranker
from src.data.loader import load_document, load_benchmark, get_referenced_files


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
        import re
        from tqdm import tqdm

        sig = f"{self.config.embedding.model_name}_{self.config.chunking.strategy}_{self.config.chunking.chunk_size}_{self.config.chunking.overlap}"
        safe_sig = re.sub(r'[^a-zA-Z0-9_\-]', '_', sig)
        cached_index_dir = INDEX_DIR / safe_sig

        # Check if cache exists
        if (cached_index_dir / "faiss.index").exists() and (cached_index_dir / "chunks.json").exists():
            print(f"Loading cached index from {cached_index_dir}...")
            self.load_index(cached_index_dir)
            return

        # Check default baseline index in INDEX_DIR
        if (INDEX_DIR / "faiss.index").exists() and (INDEX_DIR / "chunks.json").exists() and self.config.embedding.model_name == "all-MiniLM-L6-v2" and self.config.chunking.strategy == "fixed" and self.config.chunking.chunk_size == 512 and self.config.chunking.overlap == 128:
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
        for file_path in tqdm(referenced_files, desc="Chunking documents"):
            try:
                text = load_document(file_path)
                chunks = chunk_document(text, file_path, self.config.chunking)
                all_chunks.extend(chunks)
            except FileNotFoundError:
                print(f"  Warning: document not found: {file_path}")

        print(f"Generated {len(all_chunks)} chunks")

        # Generate embeddings
        chunk_texts = [c.text for c in all_chunks]
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
        """Build a BM25 index over the chunk texts."""
        from rank_bm25 import BM25Okapi

        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        from dataclasses import asdict
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
            tokenized = [c["text"].lower().split() for c in self.bm25_chunks]
            self.bm25 = BM25Okapi(tokenized)

        self._init_components()

    def query(self, question: str) -> List[RetrievalResult]:
        """
        Answer a question by retrieving relevant passages.
        
        Returns a list of RetrievalResult objects sorted by relevance score,
        or an empty list if the system decides to abstain (no relevant passage).
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")

        top_k = self.config.retrieval.top_k

        # Document routing
        filter_file: Optional[str] = None
        if self.config.retrieval.use_routing and self.router is not None:
            target_file, conf, named = self.router.route(question)
            if named and target_file is None:
                # Question explicitly named an agreement not present in the indexed corpus
                return []
            if target_file is not None:
                filter_file = target_file

        # Candidate retrieval size
        candidate_k = (
            self.config.retrieval.rerank_top_k
            if self.config.retrieval.use_reranker
            else max(top_k * 2, 20)
        )

        # Dense retrieval
        query_embedding = self.embedder.embed_query(question)
        dense_results = self.index.search(
            query_embedding,
            top_k=candidate_k,
            filter_file=filter_file,
        )

        # Combine with BM25 if configured
        if self.config.retrieval.use_bm25 and self.bm25 is not None:
            results = self._hybrid_search(question, dense_results, candidate_k, filter_file=filter_file)
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
            results = self.reranker.rerank(question, results, top_k=top_k)
        else:
            results = results[:top_k]

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
        Combine dense and BM25 retrieval scores.
        
        Uses a weighted combination of normalized dense and BM25 scores
        to produce the candidate ranking.
        """
        query_tokens = question.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_max = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        bm25_scores_norm = bm25_scores / bm25_max

        dense_weight = self.config.retrieval.dense_weight
        bm25_weight = self.config.retrieval.bm25_weight

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
                elif bm25_score_norm > 0.4:
                    scored[key] = {
                        "chunk": c,
                        "dense_score": 0.0,
                        "bm25_score": float(bm25_score_norm),
                    }

        results = []
        for key, data in scored.items():
            combined_score = (
                dense_weight * data["dense_score"]
                + bm25_weight * data["bm25_score"]
            )
            results.append(RetrievalResult(
                text=data["chunk"]["text"],
                file_path=data["chunk"]["file_path"],
                start=data["chunk"]["start"],
                end=data["chunk"]["end"],
                score=combined_score,
                chunk_id=data["chunk"]["chunk_id"],
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
