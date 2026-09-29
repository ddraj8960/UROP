"""
Generator schemas (Service S2).
Defines contracts for Vanilla Baseline and Initial Answer Generator.
"""
from __future__ import annotations

from typing import Literal, Any
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    query_id: str = Field(..., description="Unique query identifier")
    input_text: str = Field(..., description="Question or claim text under test")
    mode: Literal["qa", "claim"] = Field(default="qa", description="qa = Question Answering, claim = FEVER claim statement")
    dataset: str | None = Field(default=None, description="Dataset name if applicable")
    context_passages: list[str] = Field(default_factory=list, description="Optional context passages (for RAG baseline)")


class GenerateResponse(BaseModel):
    query_id: str = Field(..., description="Matches request query_id")
    input_text: str = Field(..., description="Input query text")
    short_answer: str = Field(..., description="Concise direct answer or verdict")
    long_answer: str = Field(..., description="Detailed explanation/generated response")
    raw_text: str = Field(..., description="Raw output text from LLM")
    model: str = Field(..., description="Model identifier used for generation")
    latency_sec: float = Field(..., description="Execution latency in seconds")
    prompt_tokens: int = Field(default=0, description="Prompt token count")
    completion_tokens: int = Field(default=0, description="Completion token count")
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
