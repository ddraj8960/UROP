"""
Fusion schemas for Service S6 (Evidence Fusion Engine).
"""
from __future__ import annotations

from typing import Literal, Any
from pydantic import BaseModel, Field
from hmafact.schemas.graph import GraphSnapshot

ConsensusType = Literal["SUPPORT", "CONTRADICT", "CONFLICT", "NEUTRAL", "INSUFFICIENT"]


class ClaimEvidenceFusion(BaseModel):
    claim_id: str = Field(..., description="Unique claim identifier")
    claim_text: str = Field(default="", description="Text of the atomic claim")
    support_passages: list[str] = Field(default_factory=list, description="IDs of supporting passages")
    contradict_passages: list[str] = Field(default_factory=list, description="IDs of contradicting passages")
    total_support_score: float = Field(default=0.0, description="Aggregated support score")
    total_contradict_score: float = Field(default=0.0, description="Aggregated contradiction score")
    conflict_detected: bool = Field(default=False, description="True if evidence contains strong conflict")
    top_support_citation: str | None = Field(default=None, description="Title/ID of highest scoring support passage")
    top_contradict_citation: str | None = Field(default=None, description="Title/ID of highest scoring contradiction passage")
    consensus: ConsensusType = Field(default="INSUFFICIENT", description="Fused evidence consensus verdict")
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional fusion metadata")


class FuseGraphEvidenceRequest(BaseModel):
    query_id: str = Field(..., description="Query ID")
    graph_snapshot: GraphSnapshot = Field(..., description="Serialized evidence graph snapshot")


class FuseGraphEvidenceResponse(BaseModel):
    query_id: str = Field(..., description="Matches request query_id")
    fusions: list[ClaimEvidenceFusion] = Field(default_factory=list, description="Fused evidence per claim")
