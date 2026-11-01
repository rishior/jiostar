"""
Tests for document routing.
"""

import pytest
from src.retrieval.router import DocumentRouter


def test_document_router_basic(monkeypatch):
    mock_docs = {
        "cuad/Apple_Inc_Services_Agreement.txt": "Services Agreement between Apple Inc. and Supplier",
        "cuad/BicycleTX_Ltd_Consulting_Agreement.txt": "Consulting Agreement between BicycleTX Ltd and Nigel Crockett for Chief Business Officer position",
        "cuad/Holiday_RV_Superstores_Endorsement.txt": "Endorsement Agreement between Holiday RV Superstores and Athlete",
    }
    monkeypatch.setattr("src.retrieval.router.load_document", lambda path: mock_docs.get(path, ""))

    corpus = list(mock_docs.keys())
    router = DocumentRouter(corpus)

    # Direct match on BicycleTX Ltd and Nigel Crockett
    q = "Consider the Service Agreement between BicycleTX Ltd and Nigel Crockett; What is the governing law?"
    target, conf, named = router.route(q)
    assert named is True
    assert target == "cuad/BicycleTX_Ltd_Consulting_Agreement.txt"
    assert conf >= 3.0


def test_document_router_unindexed_named_contract(monkeypatch):
    mock_docs = {
        "cuad/BicycleTX_Ltd_Consulting_Agreement.txt": "Consulting Agreement between BicycleTX Ltd and Nigel Crockett",
    }
    monkeypatch.setattr("src.retrieval.router.load_document", lambda path: mock_docs.get(path, ""))

    corpus = list(mock_docs.keys())
    router = DocumentRouter(corpus)

    # Named contract that is not in corpus
    q = "Consider the Distribution Agreement between Netflix Entertainment and Marvel Studios; Can it be assigned?"
    target, conf, named = router.route(q)
    assert named is True
    assert target is None  # Should detect out-of-corpus named contract


def test_document_router_generic_query(monkeypatch):
    mock_docs = {
        "cuad/BicycleTX_Ltd_Consulting_Agreement.txt": "Consulting Agreement between BicycleTX Ltd and Nigel Crockett",
    }
    monkeypatch.setattr("src.retrieval.router.load_document", lambda path: mock_docs.get(path, ""))

    corpus = list(mock_docs.keys())
    router = DocumentRouter(corpus)

    # Generic query without contract named
    q = "What is the policy for telepathic communication with extraterrestrial species?"
    target, conf, named = router.route(q)
    assert named is False
    assert target is None
