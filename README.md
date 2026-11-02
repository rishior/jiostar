# Contract Intelligence Assistant

## Overview

The Contract Intelligence Assistant is an automated legal clause retrieval and verification system engineered for high-precision contract discovery over the LegalBench-RAG benchmark (CUAD subset of 194 canonical queries across 29 commercial agreements). Given a natural language query regarding contract terms, permissions, intellectual property covenants, or liabilities, the system retrieves and ranks relevant clauses with exact document paths and character-level offsets. When a query cannot be answered by the contract corpus—either because it references an agreement outside the corpus or inquires about a clause absent from an indexed agreement—the system explicitly abstains rather than returning unverified text.

## Setup

### Python Version

The project requires Python 3.10 or higher. Tested and validated across Python 3.10, 3.11, 3.12, 3.13, and 3.14 on Linux, macOS, and Windows.

### Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone <repository_url>
cd JioStar
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Dataset Acquisition

Acquire the canonical LegalBench-RAG CUAD benchmark and corpus files with a single command:

```bash
python run.py download
```

This downloads `benchmarks/cuad.json` and the corresponding documents under `corpus/cuad/` into the `data` directory, deterministically selecting the canonical 194 queries that constitute LegalBench-RAG-mini using the benchmark authors' exact seed function.

### LLM Configuration

The primary retrieval and verification pipeline is completely local, deterministic, and self-contained. It relies on dense sentence transformers, FAISS vector indexing, entity-aware document routing, and calibrated score thresholding. No external LLM API key, network access, or external cloud endpoints are required. All inference runs on standard CPU hardware.

## How to Run

### Command Line Interface (CLI)

Search the contract corpus with a natural language query:

```bash
python run.py query "Does this agreement allow sublicensing to third parties?"
```

Query with a contract-specific query (demonstrating prefix stripping and document routing):

```bash
python run.py query "Consider the Cooperation Agreement between HPIL ENERGYTECH Inc. and GINARES GROUP AG; What is the governing law?"
```

Query referencing an unindexed contract (demonstrating out-of-corpus router abstention):

```bash
python run.py query "Consider the Master Services Agreement between Netflix Inc. and Sony Pictures; What is the dispute resolution procedure?"
```

Query an absent clause in an indexed contract (demonstrating in-corpus threshold abstention):

```bash
python run.py query "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett; What are the requirements for cryptocurrency payment of executive bonuses?"
```

Evaluate the pipeline on the train partition:

```bash
python run.py evaluate --config configs/final_v2.yaml --split train
```

Evaluate the pipeline on the test partition:

```bash
python run.py evaluate --config configs/final_v2.yaml --split test
```

Execute the automated self-improvement loop:

```bash
python run.py improve --max-iterations 6
```

Validate document character offsets against ground truth annotations:

```bash
python run.py validate
```

Build or refresh the vector search index:

```bash
python run.py build-index
```

### Web Application

Launch the Streamlit web dashboard:

```bash
python run.py app
```

Alternatively, invoke Streamlit directly:

```bash
python -m streamlit run src/app.py
```

The web application provides an enterprise legal intelligence interface featuring query input with automatic prefix stripping diagnostics, adjustable result cutoffs ($k \in [1, 20]$), formatted provision cards with exact character coordinates, and preconfigured benchmark test queries.

### Test Suite

Run the comprehensive unit and integration test suite:

```bash
python run.py test
```

or via pytest:

```bash
python -m pytest
```

All 62 tests execute in under 6 seconds, validating offset preservation, character-level metrics, split determinism, router entity extraction, heading-aware chunking, margin thresholding, and end-to-end pipeline execution.

## Architecture

The system follows a modular, decoupled architecture where components communicate through typed dataclasses and configuration objects:

```
src/
├── config.py                 # Centralized dataclasses and YAML serialization
├── cli.py                    # CLI command router and argument parsing
├── app.py                    # Streamlit enterprise web dashboard
├── data/
│   ├── loader.py             # Dataset loading and mini-benchmark extraction
│   ├── splitter.py           # Deterministic SHA256 train/test query partitioner
│   └── validator.py          # Character offset verification against source text
├── indexing/
│   ├── chunker.py            # Heading-aware and word-aligned sliding window chunkers
│   ├── embedder.py           # Dense sentence transformer wrapper (all-MiniLM-L6-v2)
│   └── index.py              # FAISS inner-product vector index with chunk metadata
├── retrieval/
│   ├── router.py             # Entity-aware document router
│   ├── ranker.py             # Cross-encoder candidate reranker
│   ├── abstention.py         # Calibrated score threshold and margin filters
│   └── pipeline.py           # End-to-end retrieval coordinator
├── evaluation/
│   ├── metrics.py            # Character-level Precision, Recall, and F1 calculations
│   └── evaluator.py          # Comprehensive evaluation harness across k cutoffs
└── improvement/
    ├── diagnosis.py          # Query failure classification and error analysis
    ├── proposals.py          # Systematic diagnostic and parameter search proposals
    └── loop.py               # Self-improvement loop with guardrail validation
```

### Data Flow

1. **Document Ingestion**: Contracts are parsed into heading-aware chunks (`extract_headings`). Each chunk stores verbatim character offsets (`start`, `end`) from raw text, while prepending the active contractual heading to `embedding_text`.
2. **Dense Indexing**: Chunk embedding texts are transformed into 384-dimensional vectors using `all-MiniLM-L6-v2`, L2-normalized, and indexed in FAISS (`IndexFlatIP`).
3. **Query Ingestion and Routing**: The user query is inspected by `DocumentRouter`. If the query identifies a specific agreement or contracting party, retrieval is scoped to that document partition. If the named agreement is absent from the corpus, the router abstains immediately.
4. **Query Prefix Optimization**: If routing succeeded, `strip_contract_prefix` removes the introductory contract title prefix (e.g., `"Consider the X Agreement between A and B;"`), isolating the substantive legal question for embedding.
5. **Vector Search & Candidate Scoring**: The dense query embedding is compared against indexed chunks within the target document scope via inner product (cosine similarity).
6. **Abstention Verification**: The top candidate score is evaluated against the calibrated threshold (0.30). If the score falls below the threshold or the candidate margin is insufficient, an empty result set is returned.
7. **Result Packaging**: Passing provisions are returned as ranked `RetrievalResult` objects containing verbatim character offsets, relevance scores, and source file paths.

## Reproducing Results

To reproduce all reported metrics from a clean environment:

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```
   *Execution time: ~2 minutes.*

2. **Data Acquisition & Verification**:
   ```bash
   python run.py download
   python run.py validate
   ```
   *Execution time: ~1 minute. All 272 ground truth spans validate with 100% character-level accuracy.*

3. **Execute Test Suite**:
   ```bash
   python run.py test
   ```
   *Execution time: ~5 seconds. All 62 test cases pass.*

4. **Reproduce Training Split Metrics (187 Queries)**:
   ```bash
   python run.py evaluate --config configs/final_v2.yaml --split train
   ```
   *Execution time: ~5 seconds (uses cached index). Matches reported metrics in REPORT.md.*

5. **Reproduce Held-Out Test Split Metrics (77 Queries)**:
   ```bash
   python run.py evaluate --config configs/final_v2.yaml --split test
   ```
   *Execution time: ~4 seconds. Matches reported metrics in REPORT.md.*

6. **Reproduce Self-Improvement Loop Execution**:
   ```bash
   python run.py improve --max-iterations 6
   ```
   *Execution time: ~45 seconds. Automatically diagnoses failure modes, applies proposals, validates guardrails, and logs results to `logs/experiment_log.jsonl`.*

## Assumptions

1. **Text Encoding & Offsets**: All contract files are ingested in standard Python UTF-8 text mode without newline alteration, preserving exact byte-to-character alignment with the LegalBench-RAG gold annotations.
2. **Benchmark Corpus Scope**: In accordance with the LegalBench-RAG-mini specification, indexing is bounded to the 29 commercial contracts referenced by the 194 CUAD benchmark queries.
3. **Multi-Span Overlap Scoring**: When a benchmark query possesses multiple valid ground truth spans across a contract, any retrieved chunk overlapping any valid span is awarded true positive character credit.
4. **Precision-First Abstention Policy**: In enterprise legal applications, returning incorrect text presents a severe operational liability. The pipeline enforces calibrated abstention when relevance confidence is inadequate.
5. **Split Integrity**: Train and test partitions are deterministically assigned via SHA-256 hash modulo 10 (72% train, 28% test). The improvement loop operates strictly on the train partition.

## AI Tool Usage

In accordance with the assignment guidelines, AI coding assistants (including Antigravity, Claude, and Gemini) were utilized as productivity accelerators during development:

1. **Codebase Scaffolding & Schemas**: Assisted in drafting boilerplate Pydantic/dataclass schema definitions in `src/config.py` and CLI argument parsers in `src/cli.py`.
2. **Test Suite Generation**: Accelerated creation of parameterized test cases in `tests/test_metrics.py` (character overlap edge cases, partial intervals, empty sets) and `tests/test_v2_features.py`.
3. **Unanswerable Query Drafting**: Assisted in brainstorming initial candidate topics for out-of-domain and in-corpus absent clause queries, which were subsequently audited and verified manually against contract texts.
4. **Documentation Formatting**: Formatted Markdown tables, ASCII architecture diagrams, and test run logs.

**Human Engineering & Independent Ownership**:
Every core algorithmic implementation—including the exact character-level metric calculations, word-aligned sliding window chunking, regex heading extraction with offset preservation, query prefix stripping logic, FAISS indexing integration, self-improvement loop diagnostic criteria, and guardrail enforcement—was conceived, debugged, and verified independently by the author. Every line of code in this repository can be explained, defended, and modified live during technical debriefs.
