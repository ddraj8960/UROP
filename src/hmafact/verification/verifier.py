"""
Service S7: Fact Verification Agent.
Evaluates EvidenceGraph & Evidence Fusion outputs to produce 5-way verdicts,
explanations, evidence citations, and corrected factual claims.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any
import networkx as nx

from hmafact.llm.openrouter import OpenRouterClient
from hmafact.schemas.claims import Claim
from hmafact.schemas.graph import GraphSnapshot
from hmafact.schemas.fusion import ClaimEvidenceFusion
from hmafact.schemas.verification import (
    ClaimVerificationResult,
    ClaimVerdictType,
    VerifyClaimsRequest,
    VerifyClaimsResponse,
)
from hmafact.fusion.fusion import fuse_graph_evidence
from hmafact.verification.prompts import (
    VERIFICATION_SYSTEM_PROMPT,
    VERIFICATION_USER_PROMPT,
    VERIFICATION_SYSTEM_PROMPT_V2,
    VERIFICATION_USER_PROMPT_V2,
)

logger = logging.getLogger(__name__)


class FactVerifier:
    """Fact verification engine operating over claims and evidence fusion graphs."""

    def __init__(self, llm_client: OpenRouterClient | None = None):
        self.llm_client = llm_client

    def verify_claims_graph(
        self,
        query_id: str,
        question: str,
        claims: list[Claim],
        graph: nx.MultiDiGraph | GraphSnapshot | dict[str, Any],
        use_llm: bool = True,
        as_of_date: date | str | None = None,
        live_enabled: bool = False,
    ) -> VerifyClaimsResponse:
        """
        Fuses graph evidence and runs 5-way claim verification for each claim.

        Args:
            query_id: Unique query identifier.
            question: Original question/context.
            claims: Atomic claims to verify.
            graph: EvidenceGraph / GraphSnapshot / MultiDiGraph.
            use_llm: Whether to invoke OpenRouter LLM for explanations/corrections.
            as_of_date: Anchor date for verification.
            live_enabled: Whether live date-aware prompts & recency are enabled.

        Returns:
            VerifyClaimsResponse with per-claim results and overall aggregate verdict.
        """
        # Step 1: Fuse graph evidence with optional recency weighting
        fusions = fuse_graph_evidence(
            graph=graph,
            as_of_date=as_of_date,
            live_enabled=live_enabled,
        )
        fusion_map = {f.claim_id: f for f in fusions}

        # Extract passage texts from graph for LLM context if available
        nx_graph: nx.MultiDiGraph
        if isinstance(graph, GraphSnapshot):
            nx_graph = nx.node_link_graph(graph.node_link)
        elif isinstance(graph, dict):
            nx_graph = nx.node_link_graph(graph.get("node_link", graph))
        elif isinstance(graph, nx.MultiDiGraph):
            nx_graph = graph
        else:
            nx_graph = getattr(graph, "graph", nx.MultiDiGraph())

        results: list[ClaimVerificationResult] = []
        supported_cnt = 0
        contradicted_cnt = 0
        insufficient_cnt = 0

        for claim in claims:
            fusion = fusion_map.get(claim.claim_id) or ClaimEvidenceFusion(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                consensus="INSUFFICIENT"
            )

            # Gather passage texts related to this claim
            passages_text_list: list[str] = []
            citations: list[str] = []
            for p_id in (fusion.support_passages + fusion.contradict_passages):
                if p_id in nx_graph.nodes:
                    p_node = nx_graph.nodes[p_id]
                    p_title = p_node.get("title", p_id)
                    p_text = p_node.get("text", "")
                    p_doc_date = p_node.get("doc_date")
                    date_str = f" (Date: {p_doc_date})" if p_doc_date else ""

                    if p_title not in citations:
                        citations.append(p_title)
                    passages_text_list.append(f"[{p_title}]{date_str}: {p_text}")

            result = self.verify_single_claim(
                question=question,
                claim=claim,
                fusion=fusion,
                passages_text="\n".join(passages_text_list),
                citations=citations,
                use_llm=use_llm,
                as_of_date=as_of_date,
                live_enabled=live_enabled,
            )

            results.append(result)

            if result.verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                supported_cnt += 1
            elif result.verdict == "CONTRADICTED":
                contradicted_cnt += 1
            else:
                insufficient_cnt += 1

        # Compute overall verdict proportionally
        overall_verdict: ClaimVerdictType
        if contradicted_cnt > 0 and contradicted_cnt >= supported_cnt:
            overall_verdict = "CONTRADICTED"
        elif supported_cnt > 0 and contradicted_cnt > 0:
            overall_verdict = "PARTIALLY_SUPPORTED"
        elif supported_cnt > 0 and insufficient_cnt == 0:
            overall_verdict = "SUPPORTED"
        elif supported_cnt > 0:
            overall_verdict = "PARTIALLY_SUPPORTED"
        else:
            overall_verdict = "INSUFFICIENT_EVIDENCE"

        return VerifyClaimsResponse(
            query_id=query_id,
            results=results,
            overall_verdict=overall_verdict,
            supported_count=supported_cnt,
            contradicted_count=contradicted_cnt,
            insufficient_count=insufficient_cnt,
        )

    def verify_single_claim(
        self,
        question: str,
        claim: Claim,
        fusion: ClaimEvidenceFusion,
        passages_text: str = "",
        citations: list[str] | None = None,
        use_llm: bool = True,
        as_of_date: date | str | None = None,
        live_enabled: bool = False,
    ) -> ClaimVerificationResult:
        """Verifies a single claim using LLM or algorithmic fallback."""
        citations = citations or []

        # Resolve as_of_date string
        as_of_str = str(as_of_date) if as_of_date else datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Default fallback heuristic based on fusion consensus
        fallback_verdict: ClaimVerdictType
        fallback_conf: float
        if fusion.consensus == "SUPPORT":
            fallback_verdict = "SUPPORTED"
            fallback_conf = min(0.95, 0.6 + fusion.total_support_score * 0.3)
        elif fusion.consensus == "CONTRADICT":
            fallback_verdict = "CONTRADICTED"
            fallback_conf = min(0.95, 0.6 + fusion.total_contradict_score * 0.3)
        elif fusion.consensus == "CONFLICT":
            fallback_verdict = "CONFLICTING_EVIDENCE"
            fallback_conf = 0.70
        elif fusion.consensus == "NEUTRAL":
            fallback_verdict = "PARTIALLY_SUPPORTED"
            fallback_conf = 0.60
        else:
            fallback_verdict = "INSUFFICIENT_EVIDENCE"
            fallback_conf = 0.40

        if not use_llm or self.llm_client is None or not passages_text.strip():
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                verdict=fallback_verdict,
                confidence=round(fallback_conf, 2),
                explanation=f"Algorithmic verification based on fusion consensus '{fusion.consensus}'.",
                citations=citations or ([fusion.top_support_citation] if fusion.top_support_citation else []),
                corrected_text=None,
                fusion=fusion,
            )

        # Select prompt version
        if live_enabled:
            sys_prompt = VERIFICATION_SYSTEM_PROMPT_V2.format(as_of_date=as_of_str)
            user_prompt = VERIFICATION_USER_PROMPT_V2.format(
                as_of_date=as_of_str,
                question=question,
                claim_text=claim.text,
                time_sensitive=claim.time_sensitive,
                evidence_text=passages_text if passages_text.strip() else "(No passages retrieved)",
                consensus=fusion.consensus,
                support_score=fusion.total_support_score,
                contradict_score=fusion.total_contradict_score,
            )
        else:
            sys_prompt = VERIFICATION_SYSTEM_PROMPT
            user_prompt = VERIFICATION_USER_PROMPT.format(
                question=question,
                claim_text=claim.text,
                evidence_text=passages_text if passages_text.strip() else "(No passages retrieved)",
                consensus=fusion.consensus,
                support_score=fusion.total_support_score,
                contradict_score=fusion.total_contradict_score,
            )

        try:
            raw_response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                temperature=0.0,
                max_tokens=400,
            )

            parsed = self._parse_llm_json(raw_response)
            verdict_str = parsed.get("verdict", fallback_verdict).upper()

            valid_verdicts = {
                "SUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED",
                "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"
            }
            if verdict_str not in valid_verdicts:
                verdict_str = fallback_verdict

            verdict: ClaimVerdictType = verdict_str  # type: ignore

            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                verdict=verdict,
                confidence=float(parsed.get("confidence", fallback_conf)),
                explanation=parsed.get("explanation", f"Verified claim against {len(citations)} sources."),
                citations=parsed.get("citations", citations),
                corrected_text=parsed.get("corrected_text"),
                fusion=fusion,
            )
        except Exception as e:
            logger.warning("LLM verification failed (%s); using fallback verification.", e)
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                verdict=fallback_verdict,
                confidence=round(fallback_conf, 2),
                explanation=f"Fallback verification based on evidence consensus '{fusion.consensus}'.",
                citations=citations,
                corrected_text=None,
                fusion=fusion,
            )

    @staticmethod
    def _parse_llm_json(response_text: str) -> dict[str, Any]:
        """Parses JSON output from LLM response string safely."""
        text = response_text.strip()
        if text.startswith("```json"):
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        return json.loads(text)
