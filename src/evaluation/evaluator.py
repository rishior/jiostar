"""
Full evaluation harness.

Runs the retrieval pipeline against train or test queries and computes all
metrics. Produces structured results that can be logged and compared across
pipeline configurations.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from src.config import PipelineConfig, LOGS_DIR
from src.data.loader import load_benchmark
from src.data.splitter import split_queries, assign_split
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
)
from src.retrieval.pipeline import RetrievalPipeline

DEFAULT_K_VALUES = [1, 2, 3, 4, 5, 8, 10, 16, 32, 64]


@dataclass
class QueryResult:
    """Results for a single query evaluation."""
    query: str
    split: str
    gold_spans: List[Tuple[str, int, int]]
    retrieved_spans: List[Tuple[str, int, int]]
    scores: List[float]
    abstained: bool
    precision_at: Dict[int, float] = field(default_factory=dict)
    recall_at: Dict[int, float] = field(default_factory=dict)
    f1_at: Dict[int, float] = field(default_factory=dict)
    failure_category: str = "perfect_match"
    rank1_is_offset0: bool = False
    in_corpus_contract: bool = True

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "split": self.split,
            "gold_spans": self.gold_spans,
            "retrieved_spans": self.retrieved_spans,
            "scores": self.scores,
            "abstained": self.abstained,
            "precision_at": {str(k): v for k, v in self.precision_at.items()},
            "recall_at": {str(k): v for k, v in self.recall_at.items()},
            "f1_at": {str(k): v for k, v in self.f1_at.items()},
            "failure_category": self.failure_category,
            "rank1_is_offset0": self.rank1_is_offset0,
            "in_corpus_contract": self.in_corpus_contract,
        }


@dataclass
class EvaluationResults:
    """Aggregated evaluation results across all queries."""
    config_name: str
    split: str
    num_queries: int
    num_answerable: int
    num_unanswerable: int
    k_values: List[int]
    mean_precision_at: Dict[int, float] = field(default_factory=dict)
    mean_recall_at: Dict[int, float] = field(default_factory=dict)
    mean_f1_at: Dict[int, float] = field(default_factory=dict)
    abstention_rate_answerable: float = 0.0                      # false negatives
    abstention_rate_unanswerable: float = 0.0                    # true negatives overall
    abstention_rate_unanswerable_out_of_corpus: float = 0.0     # true negatives out of corpus
    abstention_rate_unanswerable_in_corpus: float = 0.0         # true negatives in corpus
    rank1_preamble_rate: float = 0.0                            # fraction where rank 1 starts at offset 0
    query_results: List[QueryResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "config_name": self.config_name,
            "split": self.split,
            "num_queries": self.num_queries,
            "num_answerable": self.num_answerable,
            "num_unanswerable": self.num_unanswerable,
            "k_values": self.k_values,
            "mean_precision_at": {str(k): v for k, v in self.mean_precision_at.items()},
            "mean_recall_at": {str(k): v for k, v in self.mean_recall_at.items()},
            "mean_f1_at": {str(k): v for k, v in self.mean_f1_at.items()},
            "abstention_rate_answerable": self.abstention_rate_answerable,
            "abstention_rate_unanswerable": self.abstention_rate_unanswerable,
            "abstention_rate_unanswerable_out_of_corpus": self.abstention_rate_unanswerable_out_of_corpus,
            "abstention_rate_unanswerable_in_corpus": self.abstention_rate_unanswerable_in_corpus,
            "rank1_preamble_rate": self.rank1_preamble_rate,
            "elapsed_seconds": self.elapsed_seconds,
            "query_results": [qr.to_dict() for qr in self.query_results],
        }
        return d

    def summary_table(self) -> str:
        """Format results as a readable table."""
        lines = []
        lines.append(f"Configuration: {self.config_name}")
        lines.append(f"Split: {self.split}")
        lines.append(f"Queries: {self.num_queries} ({self.num_answerable} answerable, {self.num_unanswerable} unanswerable)")
        lines.append(f"Time: {self.elapsed_seconds:.1f}s")
        lines.append("")
        lines.append(f"{'k':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        lines.append("-" * 42)
        for k in self.k_values:
            p = self.mean_precision_at.get(k, 0)
            r = self.mean_recall_at.get(k, 0)
            f = self.mean_f1_at.get(k, 0)
            lines.append(f"{k:<6} {p:<12.4f} {r:<12.4f} {f:<12.4f}")
        lines.append("")
        lines.append(f"Abstention on answerable (false neg): {self.abstention_rate_answerable:.4f}")
        lines.append(f"Abstention on unanswerable (total):    {self.abstention_rate_unanswerable:.4f}")
        lines.append(f"  - Out-of-corpus contracts:          {self.abstention_rate_unanswerable_out_of_corpus:.4f}")
        lines.append(f"  - In-corpus near-misses:            {self.abstention_rate_unanswerable_in_corpus:.4f}")
        lines.append(f"Rank-1 offset 0 (preamble) rate:      {self.rank1_preamble_rate:.4f}")
        return "\n".join(lines)


class Evaluator:
    """
    Evaluation harness that runs the pipeline on benchmark queries and computes
    character-level metrics.
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        k_values: Optional[List[int]] = None,
    ):
        self.pipeline = pipeline
        self.k_values = k_values or DEFAULT_K_VALUES

    def evaluate(
        self,
        queries: List[dict],
        unanswerable_queries: Optional[List[dict]] = None,
        split: str = "train",
        verbose: bool = True,
    ) -> EvaluationResults:
        """
        Evaluate the pipeline on a set of queries.
        """
        from tqdm import tqdm

        split_queries_list = [
            q for q in queries if assign_split(q["query"]) == split
        ]

        split_unanswerable = []
        if unanswerable_queries:
            split_unanswerable = [
                q for q in unanswerable_queries
                if assign_split(q["query"]) == split
            ]

        if verbose:
            print(f"Evaluating on {split} split: {len(split_queries_list)} answerable, {len(split_unanswerable)} unanswerable")

        start_time = time.time()
        max_k = max(self.k_values)

        query_results = []
        abstained_answerable = 0
        preamble_count = 0

        iterator = tqdm(split_queries_list, desc="Answerable") if verbose else split_queries_list
        for q in iterator:
            gold_spans = [
                (s["file_path"], s["start"], s["end"])
                for s in q["snippets"]
            ]

            results = self.pipeline.query(q["query"], top_k=max_k)
            abstained = len(results) == 0

            if abstained:
                abstained_answerable += 1

            retrieved_spans = [
                (r.file_path, r.start, r.end) for r in results
            ]
            scores = [r.score for r in results]

            rank1_is_offset0 = bool(results and results[0].start == 0)
            if rank1_is_offset0:
                preamble_count += 1

            qr = QueryResult(
                query=q["query"],
                split=split,
                gold_spans=gold_spans,
                retrieved_spans=retrieved_spans,
                scores=scores,
                abstained=abstained,
                rank1_is_offset0=rank1_is_offset0,
                in_corpus_contract=True,
            )

            for k in self.k_values:
                qr.precision_at[k] = precision_at_k(retrieved_spans, gold_spans, k)
                qr.recall_at[k] = recall_at_k(retrieved_spans, gold_spans, k)
                qr.f1_at[k] = f1_at_k(retrieved_spans, gold_spans, k)

            # Failure categorization
            if abstained:
                qr.failure_category = "false_abstention"
            else:
                gold_files = set(s[0] for s in gold_spans)
                ret_files = set(s[0] for s in retrieved_spans[:5])
                if not (gold_files & ret_files):
                    qr.failure_category = "wrong_document"
                else:
                    r5 = qr.recall_at.get(5, 0.0)
                    if r5 >= 0.8:
                        qr.failure_category = "perfect_match"
                    elif r5 > 0.0:
                        qr.failure_category = "partial_overlap"
                    else:
                        qr.failure_category = "wrong_section"

            query_results.append(qr)

        # Evaluate unanswerable queries
        abstained_unanswerable = 0
        abstained_out_of_corpus = 0
        total_out_of_corpus = 0
        abstained_in_corpus = 0
        total_in_corpus = 0
        unanswerable_results = []

        if split_unanswerable:
            iterator = tqdm(split_unanswerable, desc="Unanswerable") if verbose else split_unanswerable
            for q in iterator:
                is_in_corpus = q.get("in_corpus_contract", False)
                if not is_in_corpus:
                    total_out_of_corpus += 1
                else:
                    total_in_corpus += 1

                results = self.pipeline.query(q["query"], top_k=max_k)
                abstained = len(results) == 0
                if abstained:
                    abstained_unanswerable += 1
                    if is_in_corpus:
                        abstained_in_corpus += 1
                        cat = "true_abstention_in_corpus"
                    else:
                        abstained_out_of_corpus += 1
                        cat = "true_abstention_out_of_corpus"
                else:
                    cat = "false_retrieval_in_corpus" if is_in_corpus else "false_retrieval_out_of_corpus"

                qr = QueryResult(
                    query=q["query"],
                    split=split,
                    gold_spans=[],
                    retrieved_spans=[(r.file_path, r.start, r.end) for r in results],
                    scores=[r.score for r in results],
                    abstained=abstained,
                    failure_category=cat,
                    rank1_is_offset0=bool(results and results[0].start == 0),
                    in_corpus_contract=is_in_corpus,
                )
                unanswerable_results.append(qr)

        elapsed = time.time() - start_time

        # Aggregate metrics
        results = EvaluationResults(
            config_name=self.pipeline.config.name,
            split=split,
            num_queries=len(split_queries_list) + len(split_unanswerable),
            num_answerable=len(split_queries_list),
            num_unanswerable=len(split_unanswerable),
            k_values=self.k_values,
            elapsed_seconds=elapsed,
        )

        for k in self.k_values:
            if query_results:
                results.mean_precision_at[k] = sum(
                    qr.precision_at[k] for qr in query_results
                ) / len(query_results)
                results.mean_recall_at[k] = sum(
                    qr.recall_at[k] for qr in query_results
                ) / len(query_results)
                results.mean_f1_at[k] = sum(
                    qr.f1_at[k] for qr in query_results
                ) / len(query_results)

        if split_queries_list:
            results.abstention_rate_answerable = abstained_answerable / len(split_queries_list)
            results.rank1_preamble_rate = preamble_count / len(split_queries_list)

        if split_unanswerable:
            results.abstention_rate_unanswerable = abstained_unanswerable / len(split_unanswerable)
            if total_out_of_corpus > 0:
                results.abstention_rate_unanswerable_out_of_corpus = abstained_out_of_corpus / total_out_of_corpus
            if total_in_corpus > 0:
                results.abstention_rate_unanswerable_in_corpus = abstained_in_corpus / total_in_corpus

        results.query_results = query_results + unanswerable_results

        if verbose:
            print()
            print(results.summary_table())

        return results

    def save_results(self, results: EvaluationResults, path: Optional[Path] = None):
        """Save evaluation results to a JSON file and track test runs."""
        if path is None:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            path = LOGS_DIR / f"eval_{results.config_name}_{results.split}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2)

        # Log test runs to persistent verifiable test_runs.jsonl tracker
        if results.split == "test":
            test_tracker = LOGS_DIR / "test_runs.jsonl"
            run_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config_name": results.config_name,
                "split": results.split,
                "num_queries": results.num_queries,
                "recall_at_5": results.mean_recall_at.get(5, 0.0),
                "precision_at_5": results.mean_precision_at.get(5, 0.0),
                "f1_at_5": results.mean_f1_at.get(5, 0.0),
                "unanswerable_abstention": results.abstention_rate_unanswerable,
                "unanswerable_abstention_in_corpus": results.abstention_rate_unanswerable_in_corpus,
            }
            with open(test_tracker, "a", encoding="utf-8") as tf:
                tf.write(json.dumps(run_entry) + "\n")

        print(f"Results saved to {path}")
