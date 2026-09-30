"""
schemas/evaluation.py — M2 evaluation contract models.

Defines the typed API surface for the judge, batch runner,
scorers, and validate_judge. All shared across Vanilla, RAG, HMA-Fact.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

SystemName = Literal["vanilla", "rag", "hmafact"]
JudgeRole = Literal["judge_primary", "judge_secondary"]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class JudgeRequest(BaseModel):
    sample_id: str
    dataset: Literal["truthfulqa", "fever", "hotpotqa"]
    input_text: str
    answer_text: str
    gold_answer: str | None = None
    gold_correct_answers: list[str] = Field(default_factory=list)
    gold_incorrect_answers: list[str] = Field(default_factory=list)
    role: JudgeRole = "judge_primary"


class JudgeVerdict(BaseModel):
    correct: bool
    informative: bool
    rationale: str
    judge_model: str


class ValidateJudgeRequest(BaseModel):
    """Parameters for bake-off κ validation."""
    role: JudgeRole = "judge_primary"
    items: list["BakeoffItem"] = Field(default_factory=list)


class BakeoffItem(BaseModel):
    """One row from j_labels_consensus.csv (100 human-labelled items)."""
    sample_id: str
    dataset: Literal["truthfulqa", "hotpotqa"]
    input_text: str
    answer_text: str
    gold_answer: str | None = None
    gold_correct_answers: list[str] = Field(default_factory=list)
    gold_incorrect_answers: list[str] = Field(default_factory=list)
    consensus_correct: bool


class ValidateJudgeResponse(BaseModel):
    kappa: float = Field(..., description="Cohen's κ between judge and human labels")
    agreement: float = Field(..., description="Raw agreement fraction")
    n: int
    role: JudgeRole


# ---------------------------------------------------------------------------
# Prediction rows and batch runner
# ---------------------------------------------------------------------------

class PredictionRow(BaseModel):
    sample_id: str
    system: SystemName
    short_answer: Optional[str] = None
    answer_text: str = ""
    abstained: bool = False
    fever_label: Optional[str] = None
    confidence: Optional[float] = None
    judge: Optional[JudgeVerdict] = None
    latency_ms: float = 0.0
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace_id: Optional[str] = None
    error: Optional[str] = None


class RunSystemRequest(BaseModel):
    system: SystemName
    dataset: Literal["truthfulqa", "fever", "hotpotqa"]
    split: Literal["dev", "test"] = "dev"
    final: bool = False
    limit: Optional[int] = None
    run_id: Optional[str] = None  # auto-generated if None


class RunSystemResponse(BaseModel):
    run_id: str
    n_done: int
    n_failed: int
    predictions_path: str


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricsResponse(BaseModel):
    run_id: str
    dataset: str
    system: SystemName
    n_scored: int
    n_failed: int
    metrics: dict[str, float]
    meta: dict[str, Any] = Field(default_factory=dict)


class ComputeMetricsRequest(BaseModel):
    run_id: str
    dataset: Literal["truthfulqa", "fever", "hotpotqa"]
    system: SystemName
    predictions_path: str
    output_path: Optional[str] = None  # defaults to runs/<run_id>/metrics.json
