"""
Confidence schemas for Service S9 (Confidence Estimator v0).
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ConfidenceMetrics(BaseModel):
    overall_confidence: float = Field(..., description="Aggregated verification confidence [0.0, 1.0]")
    evidence_coverage: float = Field(default=0.0, description="Ratio of claims with retrieved evidence [0.0, 1.0]")
    consensus_score: float = Field(default=0.0, description="Average evidence consensus score [0.0, 1.0]")
    verification_ratio: float = Field(default=0.0, description="Ratio of claims verified as SUPPORTED [0.0, 1.0]")
    logic_consistency: float = Field(default=1.0, description="Inter-claim logic consistency [0.0, 1.0]")
    meta: dict[str, Any] = Field(default_factory=dict, description="Detailed score breakdown")
