"""
tests/test_claims.py — Unit tests for Module 5 (S5 Claim Extraction Agent).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.claims.extractor import (
    _clean_json_array,
    extract_claims,
    extract_entities,
    generate_claim_queries,
)
from hmafact.schemas.claims import (
    Claim,
    ExtractClaimsRequest,
    ExtractClaimsResponse,
    GenerateQueriesRequest,
)


class TestClaimSchemas:
    def test_claim_model(self):
        c = Claim(
            claim_id="q1:c0",
            text="Elvis Presley was born in Tupelo, Mississippi.",
            entities=["Elvis Presley", "Tupelo", "Mississippi"],
            claim_type="entity",
        )
        assert c.claim_id == "q1:c0"
        assert len(c.entities) == 3
        assert c.claim_type == "entity"


class TestEntityExtraction:
    def test_extract_entities_fallback(self):
        text = "Albert Einstein won the Nobel Prize in Physics in Sweden."
        ents = extract_entities(text)
        assert "Albert Einstein" in ents or "Einstein" in ents or "Nobel Prize" in ents


class TestFeverPassThrough:
    def test_fever_mode_claims(self):
        req = ExtractClaimsRequest(
            query_id="fever:100",
            question="The Walking Dead is a comic book series.",
            long_answer="",
            mode="claim",
        )
        resp = extract_claims(req)
        assert resp.n_claims == 1
        assert resp.claims[0].text == "The Walking Dead is a comic book series."
        assert resp.claims[0].claim_id == "fever:100:c0"


class TestQueryGen:
    def test_query_generation(self):
        claim = Claim(
            claim_id="c1",
            text="Boston Patriots were founded in 1960.",
            entities=["Boston Patriots"],
            claim_type="temporal",
        )
        resp = generate_claim_queries(GenerateQueriesRequest(claims=[claim]))
        queries = resp.queries.get("c1", [])
        assert len(queries) >= 1
        assert "Boston Patriots" in queries[0] or "Boston Patriots" in queries[1]


class TestLiveClaimExtraction:
    @pytest.mark.live
    def test_live_extraction(self):
        req = ExtractClaimsRequest(
            query_id="tqa:1",
            question="What brand of cigarettes do doctors recommend?",
            long_answer="Doctors do not recommend any specific brand of cigarettes because all smoking causes cancer.",
            mode="qa",
        )
        resp = extract_claims(req)
        assert isinstance(resp, ExtractClaimsResponse)
        assert resp.n_claims >= 1
        assert len(resp.claims[0].text) > 0
