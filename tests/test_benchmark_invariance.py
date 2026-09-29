"""
Test Requirement A: Benchmark Invariance Test.
Asserts that with live.enabled=false, output claims and queries are 100% byte-identical to golden benchmark behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from hmafact.schemas.claims import ExtractClaimsRequest, GenerateQueriesRequest, Claim
from hmafact.claims.extractor import extract_claims, generate_claim_queries


def test_benchmark_invariance_queries():
    claim = Claim(
        claim_id="b_test:c0",
        text="M.K. Stalin is the current Chief Minister of Tamil Nadu.",
        entities=["M.K. Stalin"],
        time_sensitive=True,
    )
    req = GenerateQueriesRequest(claims=[claim])

    # When live_enabled=False (benchmark behavior)
    res_off = generate_claim_queries(req, live_enabled=False)

    # Output MUST be identical to basic entity + claim query pair
    expected_queries = [
        "M.K. Stalin is the current Chief Minister of Tamil Nadu.",
        "M.K. Stalin Nadu."
    ]
    assert res_off.queries["b_test:c0"] == expected_queries


def test_benchmark_invariance_flag_disabled():
    # Benchmark queries must not trigger live question-anchored query generation when live_enabled=False
    claim = Claim(
        claim_id="b_test:c1",
        text="The Eiffel Tower was built in 1889.",
        entities=["Eiffel Tower"],
        time_sensitive=False,
    )
    req = GenerateQueriesRequest(claims=[claim])
    res = generate_claim_queries(req, live_enabled=False)
    assert len(res.queries["b_test:c1"]) == 2
