"""
Script to build index, run baseline evaluation, and execute the improvement loop.
Captures all output for the REPORT.md.
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


def main():
    print("=" * 60)
    print("CONTRACT INTELLIGENCE ASSISTANT")
    print("Baseline Evaluation")
    print("=" * 60)

    # Load data
    queries = load_benchmark()
    print(f"\nLoaded {len(queries)} queries")
    print(f"Split summary: {split_summary(queries)}")

    # Load unanswerable queries
    unanswerable_path = DATA_DIR / "unanswerable_queries.json"
    unanswerable = []
    if unanswerable_path.exists():
        with open(unanswerable_path, "r", encoding="utf-8") as f:
            unanswerable = json.load(f)
        print(f"Loaded {len(unanswerable)} unanswerable queries")

    # Build baseline pipeline
    config = PipelineConfig(name="baseline")
    config.save(CONFIGS_DIR / "baseline.yaml")
    print(f"\nBaseline config: chunk_size={config.chunking.chunk_size}, "
          f"overlap={config.chunking.overlap}, "
          f"model={config.embedding.model_name}, "
          f"threshold={config.abstention.score_threshold}")

    pipeline = RetrievalPipeline(config)

    print("\nBuilding index...")
    t0 = time.time()
    pipeline.build_index(queries)
    build_time = time.time() - t0
    print(f"Index built in {build_time:.1f}s")

    # Save index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.save_index(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")

    # Evaluate on train
    print("\n" + "=" * 60)
    print("TRAIN EVALUATION")
    print("=" * 60)

    evaluator = Evaluator(pipeline)
    train_results = evaluator.evaluate(queries, unanswerable, split="train")
    evaluator.save_results(train_results)

    print("\n" + train_results.summary_table())


if __name__ == "__main__":
    main()

