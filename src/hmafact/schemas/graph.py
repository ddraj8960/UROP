"""
Graph and NLI schemas for Service S4 (Dynamic Evidence Retrieval Graph) & S4a (NLI Scorer).
"""
from __future__ import annotations

from typing import Literal, Any
from pydantic import BaseModel, Field

NodeType = Literal["QUERY", "CLAIM", "SEARCH_QUERY", "PASSAGE", "DOCUMENT", "SOURCE", "ENTITY"]
EdgeType = Literal[
    "HAS_CLAIM", "GENERATED_QUERY", "RETRIEVED", "PART_OF",
    "FROM_SOURCE", "MENTIONS", "SUPPORTS", "CONTRADICTS",
    "RELATED_TO", "CONFLICTS_WITH", "DERIVED_FROM"
]


class NLIScores(BaseModel):
    entail: float = Field(..., description="Entailment probability [0.0, 1.0]")
    neutral: float = Field(..., description="Neutral probability [0.0, 1.0]")
    contradict: float = Field(..., description="Contradiction probability [0.0, 1.0]")


class GraphSnapshot(BaseModel):
    node_link: dict[str, Any] = Field(..., description="Serialized NetworkX graph (node_link_data format)")
    iteration: int = Field(default=0, description="Graph expansion iteration count")
    node_count: int = Field(default=0, description="Total node count in graph")
    edge_count: int = Field(default=0, description="Total edge count in graph")


class BuildGraphRequest(BaseModel):
    query_id: str = Field(..., description="Unique query identifier")
    question: str = Field(..., description="Original user question or claim text")
    claims: list[dict[str, Any]] = Field(..., description="List of atomic claim dicts or Claim models")
    k_per_query: int = Field(default=3, description="Passages to retrieve per search query")


class BuildGraphResponse(BaseModel):
    query_id: str = Field(..., description="Matches request query_id")
    graph_snapshot: GraphSnapshot = Field(..., description="Serialized evidence graph snapshot")
    passage_count: int = Field(default=0, description="Total unique evidence passages in graph")
    claim_count: int = Field(default=0, description="Total claim nodes in graph")
