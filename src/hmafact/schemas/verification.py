"""
Verification schemas for Service S7 (Fact Verification Agent).
"""
from __future__ import annotations

from typing import Literal, Any
from pydantic import BaseModel, Field
from hmafact.schemas.claims import Claim
from hmafact.schemas.graph import GraphSnapshot
from hmafact.schemas.fusion import ClaimEvidenceFusion

ClaimVerdictType = Literal[
    "SUPPORTED",
    "CONTRADICTED",
    "PARTIALLY_SUPPORTED",
    "INSUFFICIENT_EVIDENCE",
    "CONFLICTING_EVIDENCE"
]


class ClaimVerificationResult(BaseModel):
    claim_id: str = Field(..., description="Unique claim identifier")
    claim_text: str = Field(..., description="Original atomic claim statement")
    verdict: ClaimVerdictType = Field(..., description="5-way verdict classification")
    confidence: float = Field(default=0.0, description="Verification confidence score [0.0, 1.0]")
    explanation: str = Field(default="", description="Detailed rationale explaining the verdict")
    citations: list[str] = Field(default_factory=list, description="Passage IDs or titles cited as evidence")
    corrected_text: str | None = Field(default=None, description="Corrected statement if claim is contradicted/partially supported")
    fusion: ClaimEvidenceFusion | None = Field(default=None, description="Associated evidence fusion metrics")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class VerifyClaimsRequest(BaseModel):
    query_id: str = Field(..., description="Query identifier")
    question: str = Field(..., description="Original user question")
    claims: list[Claim] = Field(..., description="Atomic claims to verify")
    graph_snapshot: GraphSnapshot | None = Field(default=None, description="Graph snapshot if available")
    use_llm: bool = Field(default=True, description="Whether to use OpenRouter LLM for verification explanation/verdict")


class VerifyClaimsResponse(BaseModel):
    query_id: str = Field(..., description="Matches request query_id")
    results: list[ClaimVerificationResult] = Field(default_factory=list, description="Verification result per claim")
    overall_verdict: ClaimVerdictType = Field(default="INSUFFICIENT_EVIDENCE", description="Aggregated overall verdict")
    supported_count: int = Field(default=0, description="Count of supported claims")
    contradicted_count: int = Field(default=0, description="Count of contradicted claims")
    insufficient_count: int = Field(default=0, description="Count of insufficient evidence claims")
