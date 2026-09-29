"""
tests/test_retrieval.py — Unit tests for Module 3 (S3 Retrieval Service).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.schemas.evidence import EvidencePassage, SearchRequest, SearchResponse
from hmafact.retrieval.hybrid import reciprocal_rank_fusion, search_evidence
from hmafact.retrieval.wiki_api import search_wikipedia_api


class TestEvidenceSchemas:
    def test_evidence_passage_validation(self):
        passage = EvidencePassage(
            passage_id="wiki_api:123:0",
            source="wiki_api",
            doc_id="123",
            title="Python (programming language)",
            text="Python is a high-level programming language.",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            retrieval_score=0.95,
            rank=1,
        )
        assert passage.passage_id == "wiki_api:123:0"
        assert passage.title == "Python (programming language)"
        assert passage.rank == 1

    def test_search_request_response(self):
        req = SearchRequest(queries=["Python programming"], k=3)
        assert req.k == 3
        resp = SearchResponse(passages={})
        assert isinstance(resp.passages, dict)


class TestRRF:
    def test_rrf_fusion_logic(self):
        p1 = EvidencePassage(passage_id="p1", doc_id="1", title="Doc 1", text="Text 1")
        p2 = EvidencePassage(passage_id="p2", doc_id="2", title="Doc 2", text="Text 2")
        p3 = EvidencePassage(passage_id="p3", doc_id="3", title="Doc 3", text="Text 3")

        list1 = [p1, p2]
        list2 = [p2, p3]

        fused = reciprocal_rank_fusion([list1, list2], top_k=3)
        assert len(fused) == 3
        # p2 appears in both lists, so it should have the highest RRF score and rank 1
        assert fused[0].passage_id == "p2"
        assert fused[0].rank == 1
        assert fused[0].retrieval_score == 1.0


class TestWikipediaRetrieval:
    @pytest.mark.live
    def test_wikipedia_api_search(self):
        passages = search_wikipedia_api(query="Albert Einstein", k=3)
        assert len(passages) > 0
        assert "Einstein" in passages[0].title or "Einstein" in passages[0].text
        assert passages[0].url.startswith("https://en.wikipedia.org/")

    @pytest.mark.live
    def test_search_evidence_interface(self):
        passages = search_evidence(query="Solar System", k=2)
        assert len(passages) == 2
        assert passages[0].rank == 1
        assert passages[1].rank == 2
