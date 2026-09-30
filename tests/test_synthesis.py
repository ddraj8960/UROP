"""
Unit tests for Module 8: S8 Logic Validation, S9 Confidence Estimator, S10 Response Synthesis.
"""
from __future__ import annotations

import pytest

from hmafact.schemas.claims import Claim
from hmafact.schemas.verification import ClaimVerificationResult, VerifyClaimsResponse
from hmafact.schemas.logic import LogicValidationResult
from hmafact.logic.validator import validate_claims_logic
from hmafact.confidence.estimator import estimate_confidence
from hmafact.synthesis.synthesizer import ResponseSynthesizer


def test_logic_validation_consistent():
    claims = [
        Claim(claim_id="q1:c0", text="Albert Einstein was born in 1879.", entities=["Albert Einstein"]),
        Claim(claim_id="q1:c1", text="Albert Einstein published relativity papers in 1905.", entities=["Albert Einstein"]),
    ]

    res = validate_claims_logic("q1", claims)
    assert res.is_consistent is True
    assert res.logic_score == 1.0
    assert len(res.contradictions) == 0


def test_logic_validation_contradiction():
    claims = [
        Claim(claim_id="q2:c0", text="Author X died in 1900.", entities=["Author X"]),
        Claim(claim_id="q2:c1", text="Author X published a novel in 1925.", entities=["Author X"]),
    ]

    res = validate_claims_logic("q2", claims)
    assert res.is_consistent is False
    assert res.logic_score < 1.0
    assert len(res.contradictions) == 1
    assert "activity after reported death" in res.contradictions[0].reason


def test_confidence_estimator():
    claim_results = [
        ClaimVerificationResult(
            claim_id="q1:c0",
            claim_text="Paris is capital of France",
            verdict="SUPPORTED",
            confidence=0.95,
            citations=["Paris"],
        ),
        ClaimVerificationResult(
            claim_id="q1:c1",
            claim_text="Eiffel Tower built in 1950",
            verdict="CONTRADICTED",
            confidence=0.90,
            citations=["Eiffel Tower"],
        ),
    ]

    metrics = estimate_confidence(claim_results)
    assert metrics.evidence_coverage == 1.0
    assert metrics.verification_ratio == 0.5
    assert metrics.overall_confidence > 0.5


def test_response_synthesizer_fallback():
    synthesizer = ResponseSynthesizer(llm_client=None)

    question = "Where is Eiffel Tower?"
    original_answer = "The Eiffel Tower is located in Rome, Italy."
    claims = [Claim(claim_id="q1:c0", text="The Eiffel Tower is located in Rome, Italy.")]

    ver_resp = VerifyClaimsResponse(
        query_id="q1",
        overall_verdict="CONTRADICTED",
        supported_count=0,
        contradicted_count=1,
        insufficient_count=0,
        results=[
            ClaimVerificationResult(
                claim_id="q1:c0",
                claim_text="The Eiffel Tower is located in Rome, Italy.",
                verdict="CONTRADICTED",
                confidence=0.95,
                citations=["Paris"],
                corrected_text="The Eiffel Tower is located in Paris, France.",
            )
        ]
    )

    synth = synthesizer.synthesize(
        query_id="q1",
        question=question,
        original_answer=original_answer,
        claims=claims,
        verification_response=ver_resp,
        use_llm=False
    )

    assert synth.query_id == "q1"
    assert synth.overall_verdict == "CONTRADICTED"
    assert "Paris" in synth.final_answer
    assert "The Eiffel Tower is located in Paris, France" in synth.final_answer
    assert len(synth.corrections_made) == 1


def test_response_synthesizer_rephrased_claim_fallback():
    synthesizer = ResponseSynthesizer(llm_client=None)

    question = "When was Eiffel Tower built?"
    original_answer = "The Eiffel Tower was built in 1950."
    claims = [Claim(claim_id="q1:c0", text="Eiffel Tower construction year was 1950")]

    ver_resp = VerifyClaimsResponse(
        query_id="q1",
        overall_verdict="CONTRADICTED",
        supported_count=0,
        contradicted_count=1,
        insufficient_count=0,
        results=[
            ClaimVerificationResult(
                claim_id="q1:c0",
                claim_text="Eiffel Tower construction year was 1950",
                verdict="CONTRADICTED",
                confidence=0.95,
                citations=["Eiffel Tower History"],
                corrected_text="The Eiffel Tower was constructed in 1889.",
            )
        ]
    )

    synth = synthesizer.synthesize(
        query_id="q1",
        question=question,
        original_answer=original_answer,
        claims=claims,
        verification_response=ver_resp,
        use_llm=False
    )

    assert synth.query_id == "q1"
    assert "1889" in synth.final_answer
    assert "1950" not in synth.final_answer
    assert len(synth.corrections_made) == 1


def test_confidence_estimator_clamped_logic():
    claim_results = [
        ClaimVerificationResult(
            claim_id="q1:c0",
            claim_text="Paris is capital of France",
            verdict="SUPPORTED",
            confidence=0.95,
            citations=["Paris"],
        ),
    ]

    # Test logic score out of bounds (<0.0)
    invalid_logic = LogicValidationResult(query_id="q1", logic_score=-0.5, is_consistent=False)
    metrics = estimate_confidence(claim_results, logic_result=invalid_logic)
    assert metrics.logic_consistency == 0.0
    assert 0.0 <= metrics.overall_confidence <= 1.0

