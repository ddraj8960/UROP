"""
Service S10: Response Synthesis Agent.
Synthesizes a fact-checked, corrected final response with inline citations and transparent confidence breakdown.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from hmafact.llm.openrouter import OpenRouterClient
from hmafact.schemas.claims import Claim
from hmafact.schemas.verification import VerifyClaimsResponse, ClaimVerificationResult
from hmafact.schemas.logic import LogicValidationResult
from hmafact.schemas.confidence import ConfidenceMetrics
from hmafact.schemas.synthesis import SynthesizedResponse
from hmafact.confidence.estimator import estimate_confidence
from hmafact.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_PROMPT

logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    """Response Synthesis Agent in HMA-Fact framework."""

    def __init__(self, llm_client: OpenRouterClient | None = None):
        self.llm_client = llm_client

    def synthesize(
        self,
        query_id: str,
        question: str,
        original_answer: str,
        claims: list[Claim],
        verification_response: VerifyClaimsResponse,
        logic_result: LogicValidationResult | None = None,
        use_llm: bool = True,
        as_of_date: date | str | None = None,
        live_enabled: bool = False,
        hedge_when_undated: bool = True,
    ) -> SynthesizedResponse:
        """
        Synthesizes a corrected final answer from claim verification results.

        Args:
            query_id: Unique query identifier.
            question: Original question.
            original_answer: Raw generated response.
            claims: Extracted claims.
            verification_response: Verification results.
            logic_result: Logic validation results.
            use_llm: Whether to invoke OpenRouter for answer rewriting.
            as_of_date: Reference anchor date for time-sensitive facts.
            live_enabled: Whether live time-sensitivity synthesis is active.
            hedge_when_undated: Whether to hedge when evidence is undated/stale.

        Returns:
            SynthesizedResponse object.
        """
        as_of_str = str(as_of_date) if as_of_date else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        claim_results = verification_response.results
        confidence_metrics = estimate_confidence(claim_results, logic_result, live_enabled=live_enabled)

        # Collect corrections, unverified claims, & citations
        corrections: list[str] = []
        unverified: list[str] = []
        citations_set: set[str] = set()

        for res in claim_results:
            for cite in res.citations:
                if cite:
                    citations_set.add(cite)

            if res.corrected_text and res.verdict in ("CONTRADICTED", "PARTIALLY_SUPPORTED"):
                corr_text = res.corrected_text
                if live_enabled and res.fusion and res.fusion.meta.get("time_sensitive") and "as of" not in corr_text.lower():
                    corr_text = f"{corr_text} (as of {as_of_str})"
                corrections.append(corr_text)
            elif res.verdict in ("INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"):
                if live_enabled and res.fusion and res.fusion.meta.get("time_sensitive") and hedge_when_undated:
                    unverified.append(f"{res.claim_text} (as of {as_of_str}, but this may have changed)")
                else:
                    unverified.append(res.claim_text)

        citations_used = sorted(list(citations_set))

        # Algorithmic fallback answer synthesis
        fallback_final_answer = self._synthesize_fallback(
            original_answer=original_answer,
            claim_results=claim_results,
            corrections=corrections,
            citations=citations_used,
            live_enabled=live_enabled,
            as_of_str=as_of_str,
        )

        final_answer = fallback_final_answer

        if use_llm and self.llm_client is not None and (corrections or citations_used or unverified):
            ver_lines = [
                f"- Claim: '{r.claim_text}' -> Verdict: {r.verdict} (Citations: {r.citations})"
                for r in claim_results
            ]
            corr_lines = [f"- {c}" for c in corrections] if corrections else ["- None"]
            unver_lines = [f"- {u}" for u in unverified] if unverified else ["- None"]

            user_prompt = SYNTHESIS_USER_PROMPT.format(
                question=question,
                original_answer=original_answer,
                verification_summary="\n".join(ver_lines),
                corrections_summary="\n".join(corr_lines),
                unverified_summary="\n".join(unver_lines),
            )

            try:
                llm_response = self.llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=500,
                )
                if llm_response and len(llm_response.strip()) > 10:
                    final_answer = llm_response.strip()
            except Exception as e:
                logger.warning("LLM synthesis failed (%s); using fallback synthesis.", e)

        return SynthesizedResponse(
            query_id=query_id,
            question=question,
            final_answer=final_answer,
            overall_verdict=verification_response.overall_verdict,
            confidence=confidence_metrics,
            claim_results=claim_results,
            corrections_made=corrections,
            citations_used=citations_used,
        )

    @staticmethod
    def _is_claim_match(claim_text: str, sentence: str) -> bool:
        """Checks substring match or significant token overlap between claim and sentence."""
        import re
        c_lower = claim_text.lower().strip()
        s_lower = sentence.lower().strip()

        if c_lower in s_lower or s_lower in c_lower:
            return True

        c_tokens = set(w for w in re.findall(r"\w+", c_lower) if len(w) > 2)
        s_tokens = set(w for w in re.findall(r"\w+", s_lower) if len(w) > 2)

        if not c_tokens or not s_tokens:
            return False

        intersection = c_tokens.intersection(s_tokens)
        overlap = len(intersection) / float(min(len(c_tokens), len(s_tokens)))
        return overlap >= 0.4

    @classmethod
    def _synthesize_fallback(
        cls,
        original_answer: str,
        claim_results: list[ClaimVerificationResult],
        corrections: list[str],
        citations: list[str],
        live_enabled: bool = False,
        as_of_str: str = "",
    ) -> str:
        """Constructs an algorithmic template synthesis answer with robust matching."""
        sentences = [s.strip() for s in original_answer.split(".") if s.strip()]

        reconstructed: list[str] = []
        applied_corrections: set[str] = set()

        for s in sentences:
            matched_correction = None
            is_unverified = False
            for r in claim_results:
                if cls._is_claim_match(r.claim_text, s):
                    if r.verdict in ("CONTRADICTED", "PARTIALLY_SUPPORTED") and r.corrected_text:
                        matched_correction = r.corrected_text
                        applied_corrections.add(r.corrected_text)
                        break
                    elif r.verdict in ("INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"):
                        is_unverified = True

            if matched_correction:
                reconstructed.append(matched_correction)
            elif is_unverified:
                if live_enabled:
                    reconstructed.append(f"[Unverified as of {as_of_str}: {s}]")
                else:
                    reconstructed.append(f"[Unverified: {s}]")
            else:
                reconstructed.append(s)

        # Append any unapplied corrections so no factual corrections are dropped
        for corr in corrections:
            if not any(corr.lower() in r_str.lower() for r_str in reconstructed):
                reconstructed.append(corr)

        res_text = ". ".join(reconstructed)
        if not res_text.endswith("."):
            res_text += "."

        if citations:
            cite_str = ", ".join(f"[{c}]" for c in citations[:3])
            res_text += f" {cite_str}"

        return res_text
