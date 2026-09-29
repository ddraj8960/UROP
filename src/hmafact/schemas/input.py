"""
BenchmarkSample and UserQuery schemas.
Owned by M0; M1 imports from here, never modifies.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    query_id: str
    text: str
    mode: Literal["qa", "claim"] = "qa"   # "claim" = FEVER: skip generator/extractor
    dataset: str | None = None             # "truthfulqa" | "fever" | "hotpotqa" | None (live)
    as_of: date | None = Field(default=None, description="Anchor date for time-sensitive queries")


class BenchmarkSample(BaseModel):
    sample_id: str = Field(..., description="Stable unique id: '{dataset}:{hash_or_native_id}'")
    dataset: Literal["truthfulqa", "fever", "hotpotqa"]
    split: Literal["dev", "test"]
    input_text: str = Field(..., description="Question (QA) or standalone claim (FEVER)")

    # Answer fields
    gold_answer: str | None = Field(default=None, description="HotpotQA answer / TruthfulQA best_answer")
    gold_correct_answers: list[str] = Field(default_factory=list, description="TruthfulQA correct answers")
    gold_incorrect_answers: list[str] = Field(default_factory=list, description="TruthfulQA incorrect answers")

    # FEVER fields
    gold_label: Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"] | None = Field(
        default=None, description="FEVER gold label"
    )
    gold_evidence: list[tuple[str, int]] = Field(
        default_factory=list,
        description="(wiki_title, sentence_idx) pairs. Raw underscore form for FEVER.",
    )

    # Evaluation subset flag (M1 §4.6: frozen 150-item test ablation subset)
    ablation_subset: bool = Field(
        default=False,
        description="True if this sample is part of the 150-item frozen ablation test subset.",
    )

    # Flexible metadata; keys are dataset-specific (see data/README.md)
    meta: dict[str, Any] = Field(default_factory=dict)
