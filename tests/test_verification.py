"""
Unit tests for Module 7: S6 Evidence Fusion & S7 Fact Verification Agent.
"""
from __future__ import annotations

import pytest

from hmafact.schemas.claims import Claim
from hmafact.schemas.evidence import EvidencePassage
from hmafact.schemas.graph import GraphSnapshot
from hmafact.graph.builder import EvidenceGraph
from hmafact.fusion.fusion import fuse_graph_evidence
from hmafact.verification.verifier import FactVerifier


@pytest.fixture
def sample_graph():
    builder = EvidenceGraph(query_id="q1", question="What is Paris and when was Eiffel Tower built?")
    # Add claims
    claim1 = Claim(claim_id="q1:c0", text="Paris is the capital of France.", claim_type="entity")
    claim2 = Claim(claim_id="q1:c1", text="The Eiffel Tower was built in 1950.", claim_type="temporal")

    builder.add_claim(claim1)
    builder.add_claim(claim2)

    # Add search queries
    builder.add_search_queries("q1:c0", ["Paris capital"])
    builder.add_search_queries("q1:c1", ["Eiffel Tower built year"])

    # Add passage evidence
    p1 = EvidencePassage(
        passage_id="p1",
        doc_id="d1",
        title="Paris",
        text="Paris is the capital and largest city of France.",
        retrieval_score=0.9,
    )
    p2 = EvidencePassage(
        passage_id="p2",
        doc_id="d2",
        title="Eiffel Tower",
        text="The Eiffel Tower was constructed from 1887 to 1889.",
        retrieval_score=0.85,
    )

    builder.add_passage_evidence("q1:c0", "q1:c0:sq0", p1, tau_e=0.5, tau_c=0.5)
    builder.add_passage_evidence("q1:c1", "q1:c1:sq0", p2, tau_e=0.5, tau_c=0.5)

    return builder.to_snapshot()


def test_fuse_graph_evidence(sample_graph: GraphSnapshot):
    fusions = fuse_graph_evidence(sample_graph)
    assert len(fusions) == 2

    fusion_map = {f.claim_id: f for f in fusions}

    # Claim 1 should be SUPPORT
    f1 = fusion_map["q1:c0"]
    assert f1.consensus == "SUPPORT"
    assert f1.total_support_score > 0
    assert f1.top_support_citation == "Paris"

    # Claim 2 should be CONTRADICT
    f2 = fusion_map["q1:c1"]
    assert f2.consensus == "CONTRADICT"
    assert f2.total_contradict_score > 0
    assert f2.top_contradict_citation == "Eiffel Tower"


def test_fact_verifier_algorithmic(sample_graph: GraphSnapshot):
    verifier = FactVerifier(llm_client=None)

    claims = [
        Claim(claim_id="q1:c0", text="Paris is the capital of France."),
        Claim(claim_id="q1:c1", text="The Eiffel Tower was built in 1950."),
    ]

    response = verifier.verify_claims_graph(
        query_id="q1",
        question="What is Paris and when was Eiffel Tower built?",
        claims=claims,
        graph=sample_graph,
        use_llm=False
    )

    assert response.query_id == "q1"
    assert len(response.results) == 2
    assert response.supported_count == 1
    assert response.contradicted_count == 1
    assert response.overall_verdict == "CONTRADICTED"

    res1 = response.results[0]
    assert res1.verdict == "SUPPORTED"

    res2 = response.results[1]
    assert res2.verdict == "CONTRADICTED"


def test_fact_verifier_conflict():
    builder = EvidenceGraph(query_id="q2", question="Is coffee healthy?")
    claim = Claim(claim_id="q2:c0", text="Coffee is healthy.")
    builder.add_claim(claim)
    builder.add_search_queries("q2:c0", ["coffee health benefits"])

    p1 = EvidencePassage(passage_id="p1", doc_id="d1", title="Study A", text="Coffee reduces heart disease risk.", retrieval_score=0.9)
    p2 = EvidencePassage(passage_id="p2", doc_id="d2", title="Study B", text="Coffee increases heart failure risk dramatically.", retrieval_score=0.9)

    builder.add_passage_evidence("q2:c0", "q2:c0:sq0", p1)
    builder.add_passage_evidence("q2:c0", "q2:c0:sq0", p2)

    # Manually add contradictory/supporting edge scores if needed or check fusion
    # In builder.add_passage_evidence, NLI scores p1/p2 against claim
    fusions = fuse_graph_evidence(builder.graph)
    assert len(fusions) == 1

    verifier = FactVerifier(llm_client=None)
    response = verifier.verify_claims_graph("q2", "Is coffee healthy?", [claim], builder.graph, use_llm=False)
    assert len(response.results) == 1
