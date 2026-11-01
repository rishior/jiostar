"""
Error diagnosis engine for the self-improvement loop.

Analyzes evaluation results to categorize failures and identify systematic
patterns that can guide the next improvement iteration.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter


@dataclass
class DiagnosisResult:
    """Diagnosis of evaluation failures."""
    total_queries: int = 0
    wrong_document: int = 0        # retrieved doc does not contain gold span
    wrong_section: int = 0         # right doc but wrong section
    partial_overlap: int = 0       # right section but span only partly covered
    incorrect_abstention: int = 0  # system abstained on answerable query
    incorrect_retrieval: int = 0   # system returned results for unanswerable
    perfect_match: int = 0         # full overlap with gold span
    
    examples: Dict[str, List[dict]] = field(default_factory=lambda: {
        "wrong_document": [],
        "wrong_section": [],
        "partial_overlap": [],
        "incorrect_abstention": [],
        "incorrect_retrieval": [],
    })

    def summary(self) -> str:
        lines = []
        lines.append(f"Total queries analyzed: {self.total_queries}")
        lines.append(f"Perfect match: {self.perfect_match}")
        lines.append(f"Wrong document: {self.wrong_document}")
        lines.append(f"Wrong section: {self.wrong_section}")
        lines.append(f"Partial overlap: {self.partial_overlap}")
        lines.append(f"Incorrect abstention: {self.incorrect_abstention}")
        lines.append(f"Incorrect retrieval (unanswerable): {self.incorrect_retrieval}")
        return "\n".join(lines)


def diagnose(query_results: List, unanswerable_results: Optional[List] = None) -> DiagnosisResult:
    """
    Analyze evaluation results to categorize failures.
    
    For each answerable query, determines whether the failure was due to:
    1. Wrong document: no retrieved chunk comes from a document containing gold spans
    2. Wrong section: right document but retrieved chunk does not overlap with gold
    3. Partial overlap: some overlap but incomplete coverage
    4. Incorrect abstention: system returned nothing for an answerable query
    
    For unanswerable queries, checks if the system incorrectly returned results.
    """
    diagnosis = DiagnosisResult()
    
    for qr in query_results:
        diagnosis.total_queries += 1
        
        if qr.abstained and qr.gold_spans:
            diagnosis.incorrect_abstention += 1
            diagnosis.examples["incorrect_abstention"].append({
                "query": qr.query,
                "gold_files": list(set(s[0] for s in qr.gold_spans)),
            })
            continue
        
        if not qr.retrieved_spans or not qr.gold_spans:
            continue
        
        gold_files = set(s[0] for s in qr.gold_spans)
        retrieved_files = set(s[0] for s in qr.retrieved_spans)
        
        # Check if any retrieved document matches a gold document
        if not gold_files & retrieved_files:
            diagnosis.wrong_document += 1
            diagnosis.examples["wrong_document"].append({
                "query": qr.query,
                "gold_files": list(gold_files),
                "retrieved_files": list(retrieved_files)[:3],
            })
            continue
        
        # Check overlap quality
        gold_chars = set()
        for fp, s, e in qr.gold_spans:
            for pos in range(s, e):
                gold_chars.add((fp, pos))
        
        retrieved_chars = set()
        for fp, s, e in qr.retrieved_spans:
            for pos in range(s, e):
                retrieved_chars.add((fp, pos))
        
        overlap = gold_chars & retrieved_chars
        
        if not overlap:
            diagnosis.wrong_section += 1
            diagnosis.examples["wrong_section"].append({
                "query": qr.query,
                "gold_spans": [(s[0], s[1], s[2]) for s in qr.gold_spans[:2]],
                "retrieved_spans": [(s[0], s[1], s[2]) for s in qr.retrieved_spans[:2]],
            })
        elif len(overlap) / len(gold_chars) < 0.8:
            diagnosis.partial_overlap += 1
            diagnosis.examples["partial_overlap"].append({
                "query": qr.query,
                "recall": len(overlap) / len(gold_chars),
            })
        else:
            diagnosis.perfect_match += 1
    
    # Analyze unanswerable queries
    if unanswerable_results:
        for qr in unanswerable_results:
            diagnosis.total_queries += 1
            if not qr.abstained:
                diagnosis.incorrect_retrieval += 1
                diagnosis.examples["incorrect_retrieval"].append({
                    "query": qr.query,
                    "top_score": qr.scores[0] if qr.scores else 0,
                })
    
    return diagnosis
