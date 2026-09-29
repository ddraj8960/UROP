"""
tests/test_graph.py — Unit tests for Module 6 (S4 Dynamic Evidence Retrieval Graph & S4a NLI Scorer).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.graph.builder import EvidenceGraph, build_evidence_graph
from hmafact.graph.nli import score_nli
from hmafact.schemas.claims import Claim
from hmafact.schemas.evidence import EvidencePassage
from hmafact.schemas.graph import BuildGraphRequest, GraphSnapshot, NLIScores


class TestNLIScorer:
    def test_score_nli_entailment(self):
        premise = "Albert Einstein was a German-born theoretical physicist."
        hypothesis = "Albert Einstein was a physicist."
        scores = score_nli(premise, hypothesis)

        assert isinstance(scores, NLIScores)
        assert scores.entail > scores.contradict
        assert round(scores.entail + scores.neutral + scores.contradict, 2) == 1.0

    def test_score_nli_contradiction(self):
        premise = "The American Medical Association advises against smoking cigarettes."
        hypothesis = "The American Medical Association recommends smoking cigarettes."
        scores = score_nli(premise, hypothesis)

        assert scores.contradict > scores.entail


class TestEvidenceGraphBuilder:
    def test_graph_node_edge_construction(self):
        eg = EvidenceGraph(query_id="q1", question="What is the capital of France?")
        c1 = Claim(
            claim_id="q1:c0",
            text="Paris is the capital of France.",
            entities=["Paris", "France"],
            claim_type="entity",
        )
        eg.add_claim(c1)
        eg.add_search_queries("q1:c0", ["Paris capital of France", "Capital city of France"])

        passage = EvidencePassage(
            passage_id="wiki_api:100:0",
            source="wiki_api",
            doc_id="100",
            title="Paris",
            text="Paris is the capital and most populous city of France.",
            url="https://en.wikipedia.org/wiki/Paris",
            retrieval_score=0.95,
            rank=1,
        )
        eg.add_passage_evidence(claim_id="q1:c0", sq_id="q1:c0:sq0", passage=passage)

        snap = eg.to_snapshot()
        assert isinstance(snap, GraphSnapshot)
        assert snap.node_count >= 5  # Query, Claim, 2 SearchQueries, Passage, Entities
        assert snap.edge_count >= 4  # HAS_CLAIM, 2 GENERATED_QUERY, RETRIEVED, SUPPORTS/RELATED_TO, MENTIONS

    @pytest.mark.live
    def test_build_evidence_graph_integration(self):
        c1 = Claim(
            claim_id="tqa:1:c0",
            text="Doctors do not recommend smoking cigarettes.",
            entities=["Doctors"],
            claim_type="relational",
        )
        req = BuildGraphRequest(
            query_id="tqa:1",
            question="What brand of cigarettes do doctors recommend?",
            claims=[c1.model_dump()],
            k_per_query=2,
        )
        res = build_evidence_graph(req)

        assert res.query_id == "tqa:1"
        assert res.claim_count == 1
        assert res.passage_count > 0
        assert res.graph_snapshot.node_count > 0
