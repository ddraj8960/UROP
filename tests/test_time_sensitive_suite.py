"""
Test Requirement B: Time-Sensitive Suite (Offline).
Tests 5 fixture questions (4 time-sensitive + 1 negative control).
"""
from __future__ import annotations

import pytest
from hmafact.claims.time_sensitive import detect_time_sensitivity
from hmafact.schemas.claims import Claim, GenerateQueriesRequest
from hmafact.claims.extractor import generate_claim_queries


FIXTURE_QUESTIONS = [
    {
        "id": "ts_1",
        "question": "Who is the current Chief Minister of Tamil Nadu?",
        "expected_ts": True,
        "expected_scope": "current",
    },
    {
        "id": "ts_2",
        "question": "Who is currently the CEO of Apple?",
        "expected_ts": True,
        "expected_scope": "current",
    },
    {
        "id": "ts_3",
        "question": "What is the current population of Tokyo?",
        "expected_ts": True,
        "expected_scope": "current",
    },
    {
        "id": "ts_4",
        "question": "Who is the latest president of France?",
        "expected_ts": True,
        "expected_scope": "current",
    },
    {
        "id": "neg_1",
        "question": "Where is the Eiffel Tower located?",
        "expected_ts": False,
        "expected_scope": "unspecified",
    },
]


@pytest.mark.parametrize("item", FIXTURE_QUESTIONS, ids=[x["id"] for x in FIXTURE_QUESTIONS])
def test_time_sensitive_detection_suite(item):
    q = item["question"]
    is_ts, scope = detect_time_sensitivity(q)
    assert is_ts == item["expected_ts"]
    assert scope == item["expected_scope"]


def test_time_sensitive_neutral_query_generation():
    q = "Who is the current Chief Minister of Tamil Nadu?"
    claim = Claim(
        claim_id="ts1:c0",
        text="M.K. Stalin is the current Chief Minister of Tamil Nadu.",
        entities=["M.K. Stalin"],
        time_sensitive=True,
    )
    req = GenerateQueriesRequest(claims=[claim])
    res = generate_claim_queries(req, question=q, as_of_year="2026", live_enabled=True)

    queries = res.queries["ts1:c0"]
    assert len(queries) > 0
    # First query should be question-anchored and NOT contain old entity 'M.K. Stalin'
    assert "M.K. Stalin" not in queries[0]
    assert "Chief Minister of Tamil Nadu" in queries[0]


def test_negative_control_unmodified():
    q = "Where is the Eiffel Tower located?"
    claim = Claim(
        claim_id="neg1:c0",
        text="The Eiffel Tower is located in Paris, France.",
        entities=["Eiffel Tower"],
        time_sensitive=False,
    )
    req = GenerateQueriesRequest(claims=[claim])
    res = generate_claim_queries(req, question=q, live_enabled=True)

    # Negative control query list must not inject time-sensitive neutral queries
    assert "as of" not in res.queries["neg1:c0"][0]
