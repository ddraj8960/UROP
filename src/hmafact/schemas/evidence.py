"""
Evidence schemas for S3 Retrieval Service.
Defines EvidencePassage, SearchRequest, and SearchResponse contracts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Any
from pydantic import BaseModel, Field

SourceName = Literal["local_wiki", "wiki_api", "wikidata"]


class EvidencePassage(BaseModel):
    passage_id: str = Field(..., description="Stable unique id: '{source}:{doc_id}:{chunk_idx}'")
    source: SourceName = Field(default="wiki_api", description="Knowledge source origin")
    doc_id: str = Field(..., description="Source document or page ID")
    title: str = Field(..., description="Document / Wikipedia article title")
    text: str = Field(..., description="Passage text snippet")
    url: str | None = Field(default=None, description="Source URL if available")
    retrieval_score: float = Field(default=0.0, description="Normalized retrieval score 0.0 - 1.0")
    rank: int = Field(default=1, description="Rank within retrieved results")
    search_query: str = Field(default="", description="Query string used to retrieve this passage")
    retrieved_at: datetime | None = Field(default=None, description="Timestamp when passage was retrieved")
    doc_date: datetime | None = Field(default=None, description="Document revision or publication date")
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional passage metadata")


class SearchRequest(BaseModel):
    queries: list[str] = Field(..., description="List of search query strings")
    sources: list[SourceName] = Field(default_factory=lambda: ["wiki_api"], description="Sources to query")
    k: int = Field(default=5, description="Number of passages to return per query")


class SearchResponse(BaseModel):
    passages: dict[str, list[EvidencePassage]] = Field(
        default_factory=dict,
        description="Map from search query to list of top-k EvidencePassages"
    )
