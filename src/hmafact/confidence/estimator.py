"""
Service S9: Heuristic Confidence Estimator v0.
Computes multi-dimensional confidence metrics over evidence graphs, verification verdicts, and logic validation.
"""
from __future__ import annotations

from typing import Any
from hmafact.schemas.verification import ClaimVerificationResult
from hmafact.schemas.logic import LogicValidationResult
from hmafact.schemas.confidence import ConfidenceMetrics


def estimate_confidence(
    claim_results: list[ClaimVerificationResult],
    logic_result: LogicValidationResult | None = None,
    live_enabled: bool = False,
) -> ConfidenceMetrics:
    """
    Computes overall system confidence score and multi-dimensional metrics.

    Args:
        claim_results: Verification results per claim.
        logic_result: Optional logic validation result.
        live_enabled: Whether live time-sensitivity penalty features are enabled.

    Returns:
        ConfidenceMetrics object.
    """
    if not claim_results:
        return ConfidenceMetrics(
            overall_confidence=0.0,
            evidence_coverage=0.0,
            consensus_score=0.0,
            verification_ratio=0.0,
            logic_consistency=1.0 if not logic_result else logic_result.logic_score,
        )

    n_claims = len(claim_results)

    # 1. Evidence coverage: claims with non-empty citations or non-zero fusion scores
    with_evidence = sum(
        1 for r in claim_results
        if r.citations or (r.fusion and (r.fusion.total_support_score > 0 or r.fusion.total_contradict_score > 0))
    )
    evidence_coverage = round(with_evidence / n_claims, 3)

    # 2. Verification ratio: claims rated SUPPORTED
    supported_cnt = sum(1 for r in claim_results if r.verdict == "SUPPORTED")
    partially_cnt = sum(1 for r in claim_results if r.verdict == "PARTIALLY_SUPPORTED")
    verification_ratio = round((supported_cnt + 0.5 * partially_cnt) / n_claims, 3)

    # 3. Consensus score: average claim confidence
    consensus_score = round(sum(r.confidence for r in claim_results) / n_claims, 3)

    # 4. Logic consistency score
    logic_consistency = (
        max(0.0, min(1.0, float(logic_result.logic_score)))
        if logic_result is not None
        else 1.0
    )

    # 5. WP6 Time-sensitivity penalty check
    stale_ts_penalty = 0.0
    if live_enabled:
        stale_ts_claims = sum(
            1 for r in claim_results
            if r.fusion and r.fusion.meta.get("time_sensitive") and r.fusion.meta.get("undated_evidence")
        )
        if stale_ts_claims > 0:
            stale_ts_penalty = 0.15 * (stale_ts_claims / n_claims)

    # 6. Overall weighted confidence calculation
    overall = (
        0.35 * verification_ratio +
        0.35 * consensus_score +
        0.15 * evidence_coverage +
        0.15 * logic_consistency -
        stale_ts_penalty
    )
    overall_confidence = round(max(0.0, min(1.0, overall)), 3)

    return ConfidenceMetrics(
        overall_confidence=overall_confidence,
        evidence_coverage=evidence_coverage,
        consensus_score=consensus_score,
        verification_ratio=verification_ratio,
        logic_consistency=logic_consistency,
        meta={
            "n_claims": n_claims,
            "supported_count": supported_cnt,
            "partially_supported_count": partially_cnt,
            "stale_ts_penalty": round(stale_ts_penalty, 3),
        }
    )
