"""
Claim schemas for Service S5 (Claim Extraction Agent).
Defines Claim, ExtractClaimsRequest, ExtractClaimsResponse, and QueryGen models.
"""
from __future__ import annotations

from typing import Literal, Any
from pydantic import BaseModel, Field

ClaimType = Literal["entity", "temporal", "numeric", "relational", "other"]
TemporalScopeType = Literal["current", "past", "unspecified"]


class Claim(BaseModel):
    claim_id: str = Field(..., description="Unique claim identifier: '{query_id}:c{index}'")
    text: str = Field(..., description="Decontextualized, standalone atomic claim statement")
    source_span: str | None = Field(default=None, description="Original text span from long_answer")
    entities: list[str] = Field(default_factory=list, description="Extracted named entities")
    claim_type: ClaimType = Field(default="other", description="Classification of claim type")
    time_sensitive: bool = Field(default=False, description="True if claim is time-sensitive (e.g. current office holder)")
    temporal_scope: TemporalScopeType = Field(default="unspecified", description="Temporal scope of claim")
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExtractClaimsRequest(BaseModel):
    query_id: str = Field(..., description="Query ID associated with the response")
    question: str = Field(..., description="Original user question or claim text")
    long_answer: str = Field(..., description="LLM generated long answer to extract claims from")
    mode: Literal["qa", "claim"] = Field(default="qa", description="FEVER 'claim' mode passes through single claim")


class ExtractClaimsResponse(BaseModel):
    query_id: str = Field(..., description="Matches request query_id")
    claims: list[Claim] = Field(default_factory=list, description="Extracted atomic claims")
    n_claims: int = Field(default=0, description="Total number of claims extracted")


class GenerateQueriesRequest(BaseModel):
    claims: list[Claim] = Field(..., description="List of atomic claims to generate queries for")


class GenerateQueriesResponse(BaseModel):
    queries: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map from claim_id to list of search queries"
    )
