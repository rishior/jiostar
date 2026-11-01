"""
Document routing module.

In legal benchmarks and real contracts, queries often reference the specific
agreement or parties involved. The DocumentRouter identifies the target
document from the query so retrieval can focus within the correct contract.
If the query refers to a contract not present in the corpus, the router
flags it so the system can abstain cleanly.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from src.data.loader import load_document, get_referenced_files


class DocumentRouter:
    """
    Routes queries to their target document in the corpus based on contract
    names, party entities, and title mentions.
    """

    STOPWORDS = {
        "consider", "the", "between", "and", "for", "this", "agreement",
        "contract", "what", "does", "are", "any", "is", "there", "under",
        "with", "which", "shall", "from", "each", "other", "into", "party",
        "parties", "dated", "effective", "hereof", "hereto", "herein",
        "service", "services", "license", "licensing", "amendment", "inc",
        "ltd", "corp", "llc", "plc", "co", "corporation", "company",
        "position", "officer", "chief", "executive", "general", "group",
        "its", "partner", "consulting", "promotion", "endorsement", "agency",
        "sponsorship", "marketing", "transportation", "strategic", "alliance",
        "york", "street", "avenue", "suite", "floor", "box",
    }

    def __init__(self, corpus_files: List[str]):
        self.corpus_files = corpus_files
        self.doc_profiles: Dict[str, Set[str]] = {}
        self._build_profiles()

    def _build_profiles(self):
        """Build term profiles for each document using filename and header text."""
        for f in self.corpus_files:
            try:
                text = load_document(f)
            except Exception:
                text = ""

            # Clean filename and header text
            fname_clean = re.sub(r'[^a-zA-Z0-9]', ' ', f).lower()
            header_clean = re.sub(r'[^a-zA-Z0-9]', ' ', text[:3000]).lower()

            words = set(
                w for w in (fname_clean + " " + header_clean).split()
                if len(w) >= 3 and w not in self.STOPWORDS
            )
            self.doc_profiles[f] = words

    def route(self, query: str) -> Tuple[Optional[str], float, bool]:
        """
        Identify the target document for a given query.

        Returns:
            (target_file_path or None, confidence_score, contract_named_in_query)
        """
        # Extract the contract naming prefix (before the semicolon if present)
        prefix = query.split(";")[0] if ";" in query else query
        named_contract = ("between" in prefix.lower() or "consider" in prefix.lower())

        q_words = [
            w.lower() for w in re.findall(r'[a-zA-Z0-9]+', prefix)
            if len(w) >= 3 and w.lower() not in self.STOPWORDS
        ]

        if not q_words:
            return None, 0.0, False

        scores: Dict[str, int] = {}
        for f, doc_words in self.doc_profiles.items():
            matches = sum(1 for w in q_words if w in doc_words)
            scores[f] = matches

        if not scores:
            return None, 0.0, False

        best_file, best_score = max(scores.items(), key=lambda x: x[1])

        # If a contract is explicitly named:
        # Benchmark queries match >= 3 distinctive tokens.
        # Queries for non-corpus agreements only match generic tokens (<= 2).
        if named_contract:
            if best_score >= 3:
                return best_file, float(best_score), True
            else:
                # Named a contract, but none in corpus matched convincingly
                return None, float(best_score), True

        # Generic question without a contract named
        return None, float(best_score), False
