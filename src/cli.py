"""
Command-line interface for the Contract Intelligence Assistant.

Provides commands for querying contracts, running evaluation, and executing
the self-improvement loop.
"""

import sys
import json
import click
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig, CONFIGS_DIR, DATA_DIR
from src.data.loader import load_benchmark, download_dataset, validate_offsets
from src.data.splitter import split_queries, split_summary
from src.retrieval.pipeline import RetrievalPipeline
from src.evaluation.evaluator import Evaluator


@click.group()
def cli():
    """Contract Intelligence Assistant: clause search over legal contracts."""
    pass


@cli.command()
def download():
    """Download the LegalBench-RAG dataset."""
    download_dataset()
    queries = load_benchmark()
    print(f"\nSplit summary:")
    summary = split_summary(queries)
    for k, v in summary.items():
        print(f"  {k}: {v}")


@cli.command()
def validate():
    """Validate that character offsets match answer texts in the benchmark."""
    queries = load_benchmark()
    valid, total, errors = validate_offsets(queries)
    print(f"Validated {valid}/{total} snippets")
    if errors:
        print(f"\n{len(errors)} errors found:")
        for err in errors[:20]:
            print(f"  {err}")
    else:
        print("All offsets are valid.")


@cli.command()
@click.argument("question")
@click.option("--config", "-c", default=None, help="Path to pipeline config YAML.")
@click.option("--top-k", "-k", default=5, help="Number of results to return.")
def query(question, config, top_k):
    """Query the contracts with a natural language question."""
    if config:
        cfg = PipelineConfig.load(config)
    else:
        default_path = CONFIGS_DIR / "final.yaml"
        if default_path.exists():
            cfg = PipelineConfig.load(default_path)
        else:
            cfg = PipelineConfig()

    cfg.retrieval.top_k = top_k

    pipeline = RetrievalPipeline(cfg)
    pipeline.build_index()

    results = pipeline.query(question)

    if not results:
        print("\nNo relevant passages found. The contract corpus does not appear to contain a clause answering this question.")
        return

    print(f"\nFound {len(results)} relevant passage(s):\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score: {r.score:.4f}) ---")
        print(f"Document: {r.file_path}")
        print(f"Offsets:  [{r.start}:{r.end}]")
        print(f"Text:     {r.text[:500]}")
        if len(r.text) > 500:
            print(f"          ... ({len(r.text)} chars total)")
        print()


@cli.command()
@click.option("--config", "-c", default=None, help="Path to pipeline config YAML.")
@click.option("--split", "-s", default="train", help="Which split to evaluate (train or test).")
def evaluate(config, split):
    """Run evaluation on the benchmark queries."""
    if config:
        cfg = PipelineConfig.load(config)
    else:
        default_path = CONFIGS_DIR / "final.yaml"
        if default_path.exists():
            cfg = PipelineConfig.load(default_path)
        else:
            cfg = PipelineConfig()

    queries = load_benchmark()

    # Load unanswerable queries
    unanswerable_path = DATA_DIR / "unanswerable_queries.json"
    unanswerable = []
    if unanswerable_path.exists():
        with open(unanswerable_path, "r", encoding="utf-8") as f:
            unanswerable = json.load(f)

    pipeline = RetrievalPipeline(cfg)
    pipeline.build_index(queries)

    evaluator = Evaluator(pipeline)
    results = evaluator.evaluate(queries, unanswerable, split=split)
    evaluator.save_results(results)


@cli.command()
@click.option("--max-iterations", "-n", default=10, help="Maximum number of iterations.")
def improve(max_iterations):
    """Run the self-improvement loop."""
    from src.improvement.loop import ImprovementLoop

    queries = load_benchmark()

    unanswerable_path = DATA_DIR / "unanswerable_queries.json"
    unanswerable = []
    if unanswerable_path.exists():
        with open(unanswerable_path, "r", encoding="utf-8") as f:
            unanswerable = json.load(f)

    baseline_path = CONFIGS_DIR / "baseline.yaml"
    if baseline_path.exists():
        initial_config = PipelineConfig.load(baseline_path)
    else:
        initial_config = PipelineConfig(name="baseline")

    loop = ImprovementLoop(
        queries=queries,
        unanswerable_queries=unanswerable,
        max_iterations=max_iterations,
    )

    best_config = loop.run(initial_config)
    print(f"\nFinal best configuration: {best_config.name}")


@cli.command()
def build_index():
    """Build the search index over the corpus."""
    cfg_path = CONFIGS_DIR / "final.yaml"
    if cfg_path.exists():
        cfg = PipelineConfig.load(cfg_path)
    else:
        cfg = PipelineConfig()

    queries = load_benchmark()
    pipeline = RetrievalPipeline(cfg)
    pipeline.build_index(queries)
    pipeline.save_index()
    print("Index built and saved.")


@cli.command()
@click.argument("pytest_args", nargs=-1)
def test(pytest_args):
    """Run unit and integration tests."""
    import pytest
    args = list(pytest_args) if pytest_args else ["-v"]
    sys.exit(pytest.main(args))


if __name__ == "__main__":
    cli()
