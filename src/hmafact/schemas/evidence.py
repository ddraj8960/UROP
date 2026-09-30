"""
Evidence schemas for S3 Retrieval Service (M3).
Defines EvidencePassage, PassageRecord, SearchRequest, SearchResponse, CorpusManifest,
BuildIndexRequest, BuildIndexResponse, EvalRetrievalRequest, and EvalRetrievalResponse.
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


class PassageRecord(BaseModel):
    passage_id: str = Field(..., description="Unique passage ID '{source}:{doc_id}:{chunk_idx}'")
    doc_id: str = Field(..., description="Source document ID")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Chunk text content")
    chunk_idx: int = Field(..., description="Zero-based chunk index within document")
    word_count: int = Field(..., description="Word count of chunk")
    dataset_origin: str = Field(default="wikipedia", description="Origin dataset (fever, hotpotqa, truthfulqa)")


class SearchRequest(BaseModel):
    queries: dict[str, list[str]] | list[str] = Field(..., description="Search query string(s) or claim_id -> query list mapping")
    sources: list[SourceName] = Field(default_factory=lambda: ["local_wiki", "wiki_api"], description="Sources to query")
    k: int = Field(default=5, description="Number of passages to return per query")


class SearchResponse(BaseModel):
    passages: dict[str, Any] = Field(
        default_factory=dict,
        description="Map from query string (or claim_id) to list of top-k EvidencePassages or single passage map"
    )


class CorpusManifest(BaseModel):
    total_passages: int = Field(..., description="Total number of passages indexed")
    total_docs: int = Field(..., description="Total unique documents indexed")
    dataset_counts: dict[str, int] = Field(default_factory=dict, description="Passage counts per origin dataset")
    seed: int = Field(default=42, description="Random seed used for corpus sampling")
    checksums: dict[str, str] = Field(default_factory=dict, description="File checksums for index artifacts")
    created_at: str = Field(..., description="ISO timestamp of corpus creation")


class BuildIndexRequest(BaseModel):
    data_dir: str = Field(default="data/raw", description="Directory containing raw benchmark datasets")
    output_dir: str = Field(default="data/indices", description="Output directory for built indexes and manifest")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    sample_ratio: float = Field(default=1.0, description="Ratio of corpus to sample for indexing")


class BuildIndexResponse(BaseModel):
    passage_count: int = Field(..., description="Number of passages indexed")
    dense_index_path: str = Field(..., description="Path to FAISS / vector index")
    sparse_index_path: str = Field(..., description="Path to BM25 index")
    manifest_path: str = Field(..., description="Path to manifest.json")
    gpu_time_seconds: float = Field(default=0.0, description="Elapsed GPU/CPU build time in seconds")


class EvalRetrievalRequest(BaseModel):
    split: str = Field(default="dev", description="Data split to evaluate on")
    dataset: str = Field(default="fever", description="Benchmark dataset to evaluate (fever, hotpotqa, all)")
    mode: str = Field(default="hybrid", description="Retrieval mode (hybrid, dense, sparse)")
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20], description="k values for Recall@k evaluation")
    max_samples: int = Field(default=100, description="Maximum samples to evaluate")


class EvalRetrievalResponse(BaseModel):
    metrics: dict[str, float] = Field(..., description="Computed retrieval metrics (Recall@k, MRR@k)")
    split: str = Field(..., description="Evaluated split")
    mode: str = Field(..., description="Evaluated retrieval mode")
    sample_count: int = Field(..., description="Number of samples evaluated")
