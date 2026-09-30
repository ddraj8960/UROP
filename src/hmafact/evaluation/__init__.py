"""hmafact.evaluation — M2 judge, metrics, and batch runner."""
from hmafact.evaluation.judge import judge_answer, validate_judge, JUDGE_PROMPTS
from hmafact.evaluation.metrics import (
    score_fever,
    score_hotpotqa,
    score_truthfulqa,
    compute_metrics,
    SCORERS,
)
from hmafact.evaluation.runner import VanillaRunner, run_system, RUNNERS

__all__ = [
    "judge_answer",
    "validate_judge",
    "JUDGE_PROMPTS",
    "score_fever",
    "score_hotpotqa",
    "score_truthfulqa",
    "compute_metrics",
    "SCORERS",
    "VanillaRunner",
    "run_system",
    "RUNNERS",
]
