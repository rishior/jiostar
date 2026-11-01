"""
Complete pipeline: build baseline, run improvement loop, evaluate on test.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig, CONFIGS_DIR, DATA_DIR, LOGS_DIR, INDEX_DIR
from src.data.loader import load_benchmark
from src.data.splitter import split_queries, split_summary, assign_split
from src.retrieval.pipeline import RetrievalPipeline
from src.evaluation.evaluator import Evaluator
from src.improvement.diagnosis import diagnose
from src.improvement.proposals import generate_proposals


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    experiment_log_path = LOGS_DIR / "experiment_log.jsonl"
    # Clear previous log
    if experiment_log_path.exists():
        experiment_log_path.unlink()

    queries = load_benchmark()
    print(f"Loaded {len(queries)} queries, split: {split_summary(queries)}")

    unanswerable_path = DATA_DIR / "unanswerable_queries.json"
    unanswerable = []
    if unanswerable_path.exists():
        with open(unanswerable_path, "r", encoding="utf-8") as f:
            unanswerable = json.load(f)
        print(f"Loaded {len(unanswerable)} unanswerable queries")

    # ========================================
    # BASELINE
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE")
    print("=" * 60)

    baseline_config = PipelineConfig(name="baseline")
    baseline_config.save(CONFIGS_DIR / "baseline.yaml")

    pipeline = RetrievalPipeline(baseline_config)
    pipeline.build_index(queries)
    pipeline.save_index(INDEX_DIR)

    evaluator = Evaluator(pipeline)
    baseline_results = evaluator.evaluate(queries, unanswerable, split="train")
    evaluator.save_results(baseline_results)

    best_config = baseline_config
    best_results = baseline_results
    best_recall5 = best_results.mean_recall_at.get(5, 0)
    best_precision5 = best_results.mean_precision_at.get(5, 0)

    def log_iteration(entry):
        with open(experiment_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # Log baseline
    log_iteration({
        "iteration": 0,
        "name": "baseline",
        "config": baseline_config.to_dict(),
        "recall_at_5": best_recall5,
        "precision_at_5": best_precision5,
        "f1_at_5": best_results.mean_f1_at.get(5, 0),
        "abstention_answerable": best_results.abstention_rate_answerable,
        "abstention_unanswerable": best_results.abstention_rate_unanswerable,
        "accepted": True,
        "reason": "baseline",
    })

    # ========================================
    # IMPROVEMENT ITERATIONS
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: IMPROVEMENT LOOP")
    print("Primary metric: Recall@5")
    print("Guardrails: Precision@5 drop < 0.05, answerable abstention < 0.15")
    print("=" * 60)

    # Define the configurations to try in order
    iterations = [
        {
            "name": "lower_threshold_0.25",
            "rationale": "Lower abstention threshold from 0.3 to 0.25 to reduce false negatives",
            "changes": {"abstention": {"score_threshold": 0.25}},
        },
        {
            "name": "smaller_chunks_256",
            "rationale": "Reduce chunk size from 512 to 256 for finer granularity in matching",
            "changes": {"chunking": {"chunk_size": 256, "overlap": 64}},
        },
        {
            "name": "larger_chunks_1024",
            "rationale": "Increase chunk size to 1024 to capture more context per clause",
            "changes": {"chunking": {"chunk_size": 1024, "overlap": 256}},
        },
        {
            "name": "sentence_chunking",
            "rationale": "Switch to sentence-based chunking to respect clause boundaries",
            "changes": {"chunking": {"strategy": "sentence", "chunk_size": 512, "overlap": 128}},
        },
        {
            "name": "hybrid_bm25",
            "rationale": "Add BM25 sparse retrieval to complement dense embeddings for legal terminology",
            "changes": {"retrieval": {"use_bm25": True, "bm25_weight": 0.3, "dense_weight": 0.7}},
        },
        {
            "name": "higher_overlap_50pct",
            "rationale": "Increase chunk overlap to 50% to improve span coverage across chunk boundaries",
            "changes": {"chunking": {"overlap": 256}},
        },
        {
            "name": "threshold_0.20",
            "rationale": "Further lower abstention threshold to 0.20",
            "changes": {"abstention": {"score_threshold": 0.20}},
        },
        {
            "name": "top_k_15",
            "rationale": "Increase top_k from 10 to 15 to improve recall",
            "changes": {"retrieval": {"top_k": 15}},
        },
    ]

    tried = set()

    for i, iteration in enumerate(iterations, 1):
        print(f"\n--- Iteration {i}: {iteration['name']} ---")
        print(f"Rationale: {iteration['rationale']}")

        # Build candidate config from current best
        candidate_dict = best_config.to_dict()
        candidate_dict["name"] = iteration["name"]
        for section, params in iteration["changes"].items():
            for key, value in params.items():
                candidate_dict[section][key] = value

        candidate_config = PipelineConfig.from_dict(candidate_dict)

        try:
            # Build new pipeline
            candidate_pipeline = RetrievalPipeline(candidate_config)
            candidate_pipeline.build_index(queries)

            candidate_evaluator = Evaluator(candidate_pipeline)
            candidate_results = candidate_evaluator.evaluate(queries, unanswerable, split="train")

            cand_recall5 = candidate_results.mean_recall_at.get(5, 0)
            cand_precision5 = candidate_results.mean_precision_at.get(5, 0)
            cand_f15 = candidate_results.mean_f1_at.get(5, 0)

            # Accept/reject logic
            precision_drop = best_precision5 - cand_precision5
            accepted = False
            reason = ""

            if candidate_results.abstention_rate_answerable > 0.15:
                reason = f"Answerable abstention too high: {candidate_results.abstention_rate_answerable:.4f}"
            elif precision_drop > 0.05:
                reason = f"Precision@5 dropped too much: {precision_drop:.4f}"
            elif cand_recall5 > best_recall5:
                accepted = True
                reason = f"Recall@5 improved: {best_recall5:.4f} -> {cand_recall5:.4f} (+{cand_recall5-best_recall5:.4f})"
            elif cand_recall5 == best_recall5 and cand_precision5 > best_precision5:
                accepted = True
                reason = f"Same recall, better precision: {best_precision5:.4f} -> {cand_precision5:.4f}"
            else:
                reason = f"No improvement: recall {best_recall5:.4f} -> {cand_recall5:.4f}, precision {best_precision5:.4f} -> {cand_precision5:.4f}"

            log_iteration({
                "iteration": i,
                "name": iteration["name"],
                "rationale": iteration["rationale"],
                "config": candidate_config.to_dict(),
                "recall_at_5": cand_recall5,
                "precision_at_5": cand_precision5,
                "f1_at_5": cand_f15,
                "abstention_answerable": candidate_results.abstention_rate_answerable,
                "abstention_unanswerable": candidate_results.abstention_rate_unanswerable,
                "accepted": accepted,
                "reason": reason,
            })

            if accepted:
                print(f"  ACCEPTED: {reason}")
                best_config = candidate_config
                best_results = candidate_results
                best_recall5 = cand_recall5
                best_precision5 = cand_precision5
            else:
                print(f"  REJECTED: {reason}")

        except Exception as e:
            print(f"  ERROR: {e}")
            log_iteration({
                "iteration": i,
                "name": iteration["name"],
                "rationale": iteration["rationale"],
                "config": candidate_config.to_dict(),
                "recall_at_5": 0,
                "precision_at_5": 0,
                "f1_at_5": 0,
                "accepted": False,
                "reason": f"Error: {str(e)}",
            })

    # Save final config
    best_config.save(CONFIGS_DIR / "final.yaml")
    print(f"\nFinal best configuration: {best_config.name}")
    print(f"Saved to {CONFIGS_DIR / 'final.yaml'}")

    # ========================================
    # FINAL TEST EVALUATION
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 3: TEST EVALUATION")
    print("=" * 60)

    final_pipeline = RetrievalPipeline(best_config)
    final_pipeline.build_index(queries)

    final_evaluator = Evaluator(final_pipeline)

    # Test split
    test_results = final_evaluator.evaluate(queries, unanswerable, split="test")
    final_evaluator.save_results(test_results)

    # Also save final train results
    train_results = final_evaluator.evaluate(queries, unanswerable, split="train")
    final_evaluator.save_results(train_results)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

