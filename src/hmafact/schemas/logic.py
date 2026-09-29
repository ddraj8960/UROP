"""
Logic schemas for Service S8 (Cross-Claim Logic Validation Agent).
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class LogicContradiction(BaseModel):
    claim_id_1: str = Field(..., description="First claim ID")
    claim_id_2: str = Field(..., description="Second claim ID")
    reason: str = Field(..., description="Explanation of logical inconsistency")


class LogicValidationResult(BaseModel):
    query_id: str = Field(..., description="Query ID")
    contradictions: list[LogicContradiction] = Field(default_factory=list, description="Detected inter-claim contradictions")
    logic_score: float = Field(default=1.0, description="Logical consistency score [0.0, 1.0]")
    is_consistent: bool = Field(default=True, description="True if no severe logical contradictions found")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")
