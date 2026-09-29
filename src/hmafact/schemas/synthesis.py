"""
Synthesis schemas for Service S10 (Response Synthesis Agent).
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from hmafact.schemas.verification import ClaimVerificationResult, ClaimVerdictType
from hmafact.schemas.confidence import ConfidenceMetrics


class SynthesizedResponse(BaseModel):
    query_id: str = Field(..., description="Query ID")
    question: str = Field(..., description="Original user question")
    final_answer: str = Field(..., description="Fact-corrected answer text with inline citations")
    overall_verdict: ClaimVerdictType = Field(..., description="Overall verification verdict")
    confidence: ConfidenceMetrics = Field(..., description="Detailed system confidence breakdown")
    claim_results: list[ClaimVerificationResult] = Field(default_factory=list, description="Per-claim verification breakdown")
    corrections_made: list[str] = Field(default_factory=list, description="List of corrected factual statements")
    citations_used: list[str] = Field(default_factory=list, description="List of cited evidence titles/IDs")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")
