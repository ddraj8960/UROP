"""
evaluation/metrics.py — M2 scorers and compute_metrics (S13).

Hallucination rate is defined ONCE here and never redefined:
    Among NON-ABSTAINED answers, the fraction judged incorrect.

Always report abstention_rate beside hallucination_rate — a system can
look flawless by refusing to answer, and the M14 ablation depends on
that pairing being visible.

Metrics are computed from STORED verdicts/predictions, never from fresh
model calls, so re-scoring is free and reproducible.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hmafact.schemas.evaluation import (
    ComputeMetricsRequest,
    MetricsResponse,
    PredictionRow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-level helpers for HotpotQA EM + F1
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower, strip punctuation and articles (SQuAD-style normalization)."""
    import re, string

    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def _em(pred: str | None, gold: str | None) -> float:
    if not pred or not gold:
        return 0.0
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _token_f1(pred: str | None, gold: str | None) -> float:
    if not pred or not gold:
        return 0.0
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# score_fever
# ---------------------------------------------------------------------------

def score_fever(
    rows: list[PredictionRow],
    gold: dict[str, Any],
) -> dict[str, float]:
    """
    FEVER: accuracy and macro-F1 over SUPPORTS / REFUTES / NOT ENOUGH INFO.

    The generator uses SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE — these
    are normalized here to the gold label space before scoring (Issue 4 fix).

    Predicting NOT ENOUGH INFO (or abstaining) counts as abstention for
    the hallucination rate.
    """
    from sklearn.metrics import f1_score  # type: ignore[import]

    # Canonical mapping: generator output → FEVER gold label space
    _NORMALIZE_FEVER: dict[str, str] = {
        # Gold space (already correct)
        "SUPPORTS": "SUPPORTS",
        "REFUTES": "REFUTES",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO",
        # Generator output → gold space
        "SUPPORTED": "SUPPORTS",
        "CONTRADICTED": "REFUTES",
        "INSUFFICIENT_EVIDENCE": "NOT ENOUGH INFO",
        # lower-case variants from short_answer
        "supports": "SUPPORTS",
        "refutes": "REFUTES",
        "not enough info": "NOT ENOUGH INFO",
        "supported": "SUPPORTS",
        "contradicted": "REFUTES",
        "insufficient_evidence": "NOT ENOUGH INFO",
        "insufficient evidence": "NOT ENOUGH INFO",
    }

    truth: list[str] = []
    predicted: list[str] = []

    for row in rows:
        g = gold.get(row.sample_id)
        if g is None:
            continue
        truth.append(g.gold_label or "NOT ENOUGH INFO")
        raw_pred = row.fever_label or row.short_answer or ""
        predicted.append(_NORMALIZE_FEVER.get(raw_pred.strip(), "NOT ENOUGH INFO"))

    if not truth:
        return {"accuracy": 0.0, "macro_f1": 0.0, "abstention_rate": 1.0, "hallucination_rate": 0.0}

    answered = [
        (t, p) for t, p in zip(truth, predicted)
        if p != "NOT ENOUGH INFO"
    ]
    n = len(truth)

    return {
        "accuracy": sum(t == p for t, p in zip(truth, predicted)) / n,
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "abstention_rate": 1 - len(answered) / n,
        "hallucination_rate": (
            sum(t != p for t, p in answered) / len(answered)
            if answered else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# score_hotpotqa
# ---------------------------------------------------------------------------

def score_hotpotqa(
    rows: list[PredictionRow],
    gold: dict[str, Any],
) -> dict[str, float]:
    """HotpotQA: judge-based accuracy + EM + token-F1."""
    answered = [row for row in rows if not row.abstained and not row.error]
    correct = [row for row in answered if row.judge and row.judge.correct]
    n = len(rows)
    if n == 0:
        return {"accuracy": 0.0, "exact_match": 0.0, "token_f1": 0.0,
                "abstention_rate": 1.0, "hallucination_rate": 0.0}

    return {
        "accuracy": len(correct) / n,
        "exact_match": sum(
            _em(row.short_answer, gold[row.sample_id].gold_answer)
            for row in rows
            if row.sample_id in gold
        ) / n,
        "token_f1": sum(
            _token_f1(row.short_answer, gold[row.sample_id].gold_answer)
            for row in rows
            if row.sample_id in gold
        ) / n,
        "abstention_rate": 1 - len(answered) / n,
        "hallucination_rate": (
            1 - len(correct) / len(answered) if answered else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# score_truthfulqa
# ---------------------------------------------------------------------------

def score_truthfulqa(
    rows: list[PredictionRow],
    gold: dict[str, Any],
) -> dict[str, float]:
    """TruthfulQA: judge-based truthfulness + informative rate."""
    answered = [row for row in rows if not row.abstained and not row.error]
    truthful = [row for row in answered if row.judge and row.judge.correct]
    informative = [
        row for row in answered
        if row.judge and row.judge.correct and row.judge.informative
    ]
    n = len(rows)
    if n == 0:
        return {"truthful_rate": 0.0, "truthful_and_informative": 0.0,
                "abstention_rate": 1.0, "hallucination_rate": 0.0}

    return {
        "truthful_rate": len(truthful) / n,
        "truthful_and_informative": len(informative) / n,
        "abstention_rate": 1 - len(answered) / n,
        "hallucination_rate": (
            1 - len(truthful) / len(answered) if answered else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCORERS: dict[str, Any] = {
    "fever": score_fever,
    "hotpotqa": score_hotpotqa,
    "truthfulqa": score_truthfulqa,
}


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    request: ComputeMetricsRequest,
    cfg: Any,
) -> MetricsResponse:
    """
    Load predictions from disk and compute all metrics.

    Metrics are derived from STORED verdicts — no model calls made here.
    Failed rows (error != None) are excluded from metric computation
    but counted in n_failed.

    Args:
        request: ComputeMetricsRequest with run_id, dataset, predictions_path.
        cfg: Config object (used for samples_dir).

    Returns:
        MetricsResponse written to runs/<run_id>/metrics.json.
    """
    from hmafact.data.loader import load_samples

    pred_path = Path(request.predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}\n"
            "Run `run_system` first to generate predictions."
        )

    # Load predictions
    rows_raw: list[dict] = []
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows_raw.append(json.loads(line))

    all_rows = [PredictionRow.model_validate(r) for r in rows_raw]
    good_rows = [r for r in all_rows if not r.error]
    n_failed = len(all_rows) - len(good_rows)

    # Load gold labels as a lookup dict
    try:
        samples_dir_override = None
        try:
            samples_dir_override = Path(cfg.data.samples_dir)  # type: ignore[attr-defined]
        except AttributeError:
            pass
        gold_list = load_samples(
            dataset=request.dataset,
            split="dev",
            data_dir=samples_dir_override,
        )
        gold = {s.sample_id: s for s in gold_list}
    except Exception as exc:
        logger.warning("Could not load gold labels: %s — metrics may be incomplete.", exc)
        gold = {}

    scorer = SCORERS.get(request.dataset)
    if scorer is None:
        raise ValueError(
            f"No scorer for dataset '{request.dataset}'. Supported: {list(SCORERS)}"
        )

    metrics = scorer(good_rows, gold)

    # Latency and token stats
    if good_rows:
        latencies = [r.latency_ms for r in good_rows]
        metrics["latency_ms_mean"] = sum(latencies) / len(latencies)
        metrics["latency_ms_p95"] = sorted(latencies)[int(len(latencies) * 0.95)]
        metrics["prompt_tokens_mean"] = sum(r.prompt_tokens for r in good_rows) / len(good_rows)
        metrics["completion_tokens_mean"] = sum(r.completion_tokens for r in good_rows) / len(good_rows)

    result = MetricsResponse(
        run_id=request.run_id,
        dataset=request.dataset,
        system=request.system,
        n_scored=len(good_rows),
        n_failed=n_failed,
        metrics=metrics,
    )

    # Write metrics.json
    output_path = (
        Path(request.output_path)
        if request.output_path
        else Path("runs") / request.run_id / "metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(), indent=2), encoding="utf-8"
    )
    logger.info(
        "Metrics written to %s — %s %s: %s",
        output_path, request.dataset, request.system,
        {k: round(v, 4) for k, v in metrics.items()},
    )

    return result
