# Contract Intelligence Assistant

## Overview

The Contract Intelligence Assistant is an automated legal clause retrieval and verification system built on the CUAD subset of the LegalBench RAG benchmark. Given a natural language query regarding contract terms, permissions, or covenants, the system retrieves and ranks relevant clauses with exact document paths and character offsets. When a query cannot be answered by the indexed corpus, the system abstains rather than returning irrelevant text.

## Setup

### Python Version

The project requires Python 3.10 or higher. Python 3.11, 3.12, 3.13, and 3.14 are supported.

### Installation

Clone the repository and install dependencies:

```bash
git clone <repository_url>
cd JioStar
pip install -r requirements.txt
```

### Dataset Acquisition

Download the LegalBench RAG CUAD benchmark and corpus files directly using the CLI command:

```bash
python run.py download
```

This downloads `benchmarks/cuad.json` and the corresponding documents under `corpus/cuad/` into the `data` directory, selecting the canonical 194 queries that constitute LegalBench RAG mini.

### LLM Configuration

The primary system relies entirely on dense sentence embeddings, FAISS vector indexing, entity aware document routing, and calibrated score based abstention thresholds. No external LLM API key or local generative model weights are required for default operation.

## How to Run

### Command Line Interface

Query the contract corpus:

```bash
python run.py query "Does this agreement allow sublicensing to third parties?"
```

Query with out of corpus entity (demonstrating abstention):

```bash
python run.py query "Consider the Master Services Agreement between Netflix Inc. and Sony Pictures; What is the dispute resolution procedure?"
```

Run evaluation on the train split:

```bash
python run.py evaluate --config configs/final.yaml --split train
```

Run evaluation on the test split:

```bash
python run.py evaluate --config configs/final.yaml --split test
```

Execute the self improvement loop:

```bash
python run.py improve --max-iterations 6
```

Validate document character offsets against ground truth snippets:

```bash
python run.py validate
```

Build or refresh the vector search index:

```bash
python run.py build-index
```

### Web Application

Start the interactive Streamlit user interface:

```bash
streamlit run src/app.py
```

The web application provides an intuitive search bar, adjustable top k passage count, document metadata views, and sample legal questions.

### Test Suite

Execute the test suite using either of the following commands:

```bash
python run.py test
```

or via the Python module runner:

```bash
python -m pytest
```

All 47 unit and integration tests covering chunking offset fidelity, character level metric computations, train test hash partitioning, abstention thresholding, document routing, and pipeline integration will execute.

## Architecture

The system is organized into modular components located within `src`:

1. `src/config.py`: Centralized configuration dataclasses and YAML serialization. Defines parameters for chunking, embedding, hybrid retrieval, routing, rerankers, and abstention thresholds.
2. `src/data/loader.py`: Dataset loading, snippet normalization, and document text extraction. Implements the benchmark authors mini selection function.
3. `src/data/splitter.py`: Deterministic SHA256 query partitioning assigning 72 percent of queries to train and 28 percent to test.
4. `src/indexing/chunker.py`: Character offset preserving text segmenters supporting fixed size sliding windows, sentence boundary chunking, and clause boundary chunking.
5. `src/indexing/embedder.py`: Dense sentence embedding generation with batching and vector normalization using `all-MiniLM-L6-v2`.
6. `src/indexing/index.py`: FAISS inner product vector index coupled with chunk metadata storage, supporting document scoped search.
7. `src/retrieval/router.py`: Entity aware document router matching queries against contract filenames and party profiles.
8. `src/retrieval/ranker.py`: Optional cross encoder reranker computing query passage cross attention.
9. `src/retrieval/abstention.py`: Calibrated score threshold and margin filters that trigger empty result sets when relevance criteria are unmet.
10. `src/retrieval/pipeline.py`: End to end retrieval coordinator executing document routing, vector search, optional reranking, and abstention filtering.
11. `src/evaluation/metrics.py`: Character level precision, recall, and F1 calculations per the LegalBench RAG specification.
12. `src/evaluation/evaluator.py`: Comprehensive test harness computing precision at k and recall at k across splits.
13. `src/improvement/proposals.py`: Systematic diagnostic proposal generator for the self improvement loop.
14. `src/improvement/loop.py`: Automated self improvement loop that diagnoses error modes, proposes configuration adjustments, and accepts or rejects modifications based on guardrail metrics.
15. `src/cli.py` and `src/app.py`: CLI command dispatcher and Streamlit web dashboard.

Data flows from documents through chunkers into FAISS indices. Natural language queries are routed to referenced agreements, searched against dense vector representations, filtered through calibrated abstention thresholds, and evaluated by computing exact character overlap with ground truth annotations.

## Reproducing Results

To reproduce reported benchmark metrics from a fresh environment:

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   Approximate time: 2 minutes.

2. Download benchmark corpus and queries:
   ```bash
   python run.py download
   ```
   Approximate time: 1 minute.

3. Verify annotation character offsets:
   ```bash
   python run.py validate
   ```
   Approximate time: 5 seconds. All 272 snippets validate without error.

4. Run unit test suite:
   ```bash
   python run.py test
   ```
   Approximate time: 5 seconds. All 47 tests pass.

5. Run evaluation on train partition:
   ```bash
   python run.py evaluate --config configs/final.yaml --split train
   ```
   Approximate time: 5 seconds (uses cached index).

6. Run evaluation on held out test partition:
   ```bash
   python run.py evaluate --config configs/final.yaml --split test
   ```
   Approximate time: 4 seconds (uses cached index).

## Assumptions

1. Character Encoding: Files are read as UTF8 in text mode without newline translation, preserving byte and character alignment with CUAD benchmark offsets.
2. Corpus Scope: Per the benchmark specification, indexing is restricted to the 29 unique documents referenced by the 194 queries in LegalBench RAG mini.
3. Ground Truth Scope: Where benchmark snippets provide multiple valid answer passages for a single query, character overlap across any gold span contributes to recall and precision.
4. Abstention Criterion: In production legal discovery, returning irrelevant text is often worse than acknowledging absence. Therefore, queries whose top retrieved score fails to meet the confidence threshold return an empty list with explicit abstention notices.
5. Evaluation Isolation: The test split is sequestered and evaluated strictly once on the final pipeline configuration. All exploratory parameter tuning and diagnostic loops operate exclusively on the train split.

## AI Tool Usage

AI coding assistants were utilized during development for:
1. Drafting boilerplate schema definitions and configuration dataclasses.
2. Generating unit test fixtures and synthetic edge cases for chunk boundary preservation.
3. Crafting thirty five plausible out of domain unanswerable contract queries.
4. Accelerating codebase scaffolding and formatting Markdown documentation.

All architectural decisions, algorithmic implementations, metric logic, diagnosis rules, and experimental validations were reviewed and verified independently.
