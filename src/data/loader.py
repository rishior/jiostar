"""
Data loader for LegalBench-RAG dataset.

Downloads the CUAD subset from HuggingFace and provides access to the benchmark
queries and corpus documents. Implements the mini() function from the assignment
to select the 194-query subset.

The benchmark stores spans as [start, end] lists under the key "span".
File paths are stored as "cuad/FILENAME.txt" relative to the corpus root.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.config import DATA_DIR, CORPUS_DIR, BENCHMARKS_DIR


def mini(tests: List[dict], n: int = 194) -> List[dict]:
    """
    Select the LegalBench-RAG-mini subset of CUAD queries.

    This is the exact function provided in the assignment specification.
    It uses a deterministic selection based on the file_path of each query's
    first snippet.
    """
    if len(tests) <= n:
        return tests

    def key(t):
        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def parse_snippet(snippet: dict) -> dict:
    """
    Normalize a raw benchmark snippet into a consistent format.

    The benchmark stores character offsets as a two-element list under "span".
    This function extracts them into "start" and "end" fields for uniform access
    throughout the codebase.
    """
    span = snippet["span"]
    return {
        "file_path": snippet["file_path"],
        "start": int(span[0]),
        "end": int(span[1]),
        "answer": snippet["answer"],
    }


def load_benchmark(benchmark_path: Optional[Path] = None) -> List[dict]:
    """
    Load the CUAD benchmark file and return the mini subset of 194 queries.

    Each returned query dict contains:
        - query: the natural language question
        - snippets: list of ground-truth answer spans, each with:
            - file_path: path to the source document (e.g. "cuad/filename.txt")
            - start: character offset start
            - end: character offset end
            - answer: the answer text
    """
    if benchmark_path is None:
        benchmark_path = BENCHMARKS_DIR / "cuad.json"

    raw = benchmark_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    tests = data["tests"]
    mini_tests = mini(tests)

    # Normalize snippet format
    normalized = []
    for t in mini_tests:
        normalized.append({
            "query": t["query"],
            "snippets": [parse_snippet(s) for s in t["snippets"]],
        })

    return normalized


def load_document(file_path: str) -> str:
    """
    Load a contract document by its file_path (as referenced in the benchmark).

    The file_path in the benchmark looks like "cuad/FILENAME.txt".
    Documents live at data/corpus/cuad/FILENAME.txt.
    They are read in text mode with UTF-8 encoding to match the character
    offsets in the benchmark annotations.
    """
    # Try the path under corpus/ (e.g. corpus/cuad/FILENAME.txt)
    full_path = DATA_DIR / "corpus" / file_path
    if full_path.exists():
        return full_path.read_text(encoding="utf-8")

    # Try direct under DATA_DIR
    alt_path = DATA_DIR / file_path
    if alt_path.exists():
        return alt_path.read_text(encoding="utf-8")

    # Try just the filename in CORPUS_DIR
    name_only = Path(file_path).name
    alt_path2 = CORPUS_DIR / name_only
    if alt_path2.exists():
        return alt_path2.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Document not found: {file_path}")


def get_referenced_files(queries: List[dict]) -> List[str]:
    """
    Extract the unique set of document file paths referenced by the queries.
    Only these documents should be indexed, per the assignment specification.
    """
    files = set()
    for q in queries:
        for snippet in q["snippets"]:
            files.add(snippet["file_path"])
    return sorted(files)


def validate_offsets(queries: List[dict]) -> Tuple[int, int, List[str]]:
    """
    Validate that document[start:end] matches each snippet's answer text.

    Returns:
        (num_valid, num_total, list_of_error_messages)
    """
    valid = 0
    total = 0
    errors = []

    for q in queries:
        for snippet in q["snippets"]:
            total += 1
            try:
                doc_text = load_document(snippet["file_path"])
                start = snippet["start"]
                end = snippet["end"]
                extracted = doc_text[start:end]
                expected = snippet["answer"]
                if extracted == expected:
                    valid += 1
                else:
                    errors.append(
                        f"Offset mismatch in {snippet['file_path']} "
                        f"[{start}:{end}]: "
                        f"expected {repr(expected[:80])}... "
                        f"got {repr(extracted[:80])}..."
                    )
            except FileNotFoundError as e:
                errors.append(str(e))

    return valid, total, errors


def download_dataset():
    """
    Download the LegalBench-RAG dataset from HuggingFace.

    Downloads only the files needed: benchmarks/cuad.json and corpus/cuad/.
    Uses huggingface_hub for reliable downloading.
    """
    from huggingface_hub import snapshot_download

    print("Downloading LegalBench-RAG dataset from HuggingFace...")
    snapshot_download(
        repo_id="awinml/legalbench-rag",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=["benchmarks/cuad.json", "corpus/cuad/*"],
    )
    print(f"Dataset downloaded to {DATA_DIR}")

    benchmark_path = BENCHMARKS_DIR / "cuad.json"
    if not benchmark_path.exists():
        raise RuntimeError(
            f"Expected benchmark file not found at {benchmark_path}. "
            "Check the download completed successfully."
        )

    queries = load_benchmark(benchmark_path)
    print(f"Loaded {len(queries)} queries from LegalBench-RAG-mini (CUAD)")

    files = get_referenced_files(queries)
    print(f"Queries reference {len(files)} unique documents")

    return queries


if __name__ == "__main__":
    download_dataset()
