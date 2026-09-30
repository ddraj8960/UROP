"""
evaluation/runner.py — M2 SystemRunner protocol, VanillaRunner, run_system, RUNNERS.

Design principles (per M2 spec):
  - Strategy via SystemRunner Protocol: one interface, multiple implementations.
  - Idempotent / resumable: run_system skips already-completed sample_ids.
  - Degrade, never abort: failures write an error row and continue.
  - Open with "a" mode (append), not "w" (overwrite) — Issue 6 fix.
  - Temperature 0, deterministic, cache on.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from hmafact.generator.generator import generate_answer
from hmafact.schemas.evaluation import (
    PredictionRow,
    RunSystemRequest,
    RunSystemResponse,
    SystemName,
)
from hmafact.schemas.generator import GenerateRequest
from hmafact.schemas.input import BenchmarkSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FEVER label normalization
# (same table as metrics.py — kept in sync; generator outputs are normalised here
#  before writing the prediction row so downstream scorers see gold-space labels)
# ---------------------------------------------------------------------------
_FEVER_FROM_SHORT: dict[str, str] = {
    "supports": "SUPPORTS",
    "refutes": "REFUTES",
    "not enough info": "NOT ENOUGH INFO",
    "supported": "SUPPORTS",
    "contradicted": "REFUTES",
    "insufficient_evidence": "NOT ENOUGH INFO",
    "insufficient evidence": "NOT ENOUGH INFO",
    # already-correct gold labels pass through
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
}


# ---------------------------------------------------------------------------
# SystemRunner Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SystemRunner(Protocol):
    """
    Every system implements exactly this.
    Vanilla now, RAG in M4, HMA-Fact in M9.
    """
    name: SystemName

    def run(self, sample: BenchmarkSample, cfg: Any) -> PredictionRow:
        """Run one sample and return a PredictionRow (never raises)."""
        ...


# ---------------------------------------------------------------------------
# VanillaRunner — Baseline 1
# ---------------------------------------------------------------------------

class VanillaRunner:
    """
    Baseline 1: generator alone, no retrieval, no verification, no abstention.
    Implements SystemRunner.
    """
    name: SystemName = "vanilla"

    def run(self, sample: BenchmarkSample, cfg: Any) -> PredictionRow:
        """
        Run one sample through the generator.

        Args:
            sample: BenchmarkSample from the dev split.
            cfg: Config object with model routing and run settings.

        Returns:
            PredictionRow (error field populated on failure, never raises).
        """
        start_ms = time.perf_counter() * 1000

        try:
            mode = "claim" if sample.dataset == "fever" else "qa"
            req = GenerateRequest(
                query_id=sample.sample_id,
                input_text=sample.input_text,
                mode=mode,
                dataset=sample.dataset,
            )

            # Resolve generator model from role routing
            model = _resolve_role("generator", cfg)
            result = generate_answer(req, model=model, temperature=0.0)

            latency_ms = time.perf_counter() * 1000 - start_ms

            # Normalise FEVER label to gold label space (Issue 4 fix)
            fever_label: str | None = None
            if mode == "claim":
                raw = (result.short_answer or "").strip()
                fever_label = _FEVER_FROM_SHORT.get(raw) or _FEVER_FROM_SHORT.get(raw.lower())

            return PredictionRow(
                sample_id=sample.sample_id,
                system=self.name,
                short_answer=result.short_answer,
                answer_text=f"{result.short_answer}. {result.long_answer}".strip(),
                abstained=False,
                fever_label=fever_label,
                confidence=None,
                latency_ms=latency_ms,
                n_llm_calls=1,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )

        except Exception as exc:
            latency_ms = time.perf_counter() * 1000 - start_ms
            logger.warning(
                "VanillaRunner failed for %s: %s", sample.sample_id, exc
            )
            return PredictionRow(
                sample_id=sample.sample_id,
                system=self.name,
                short_answer=None,
                answer_text="",
                latency_ms=latency_ms,
                n_llm_calls=0,
                prompt_tokens=0,
                completion_tokens=0,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )


# ---------------------------------------------------------------------------
# RUNNERS registry
# ---------------------------------------------------------------------------

from hmafact.baselines.rag import RagRunner

RUNNERS: dict[str, type[SystemRunner]] = {
    "vanilla": VanillaRunner,
    "rag": RagRunner,
}


# ---------------------------------------------------------------------------
# run_system — resumable batch runner
# ---------------------------------------------------------------------------

def _read_done_ids(path: Path) -> set[str]:
    """
    Read sample_ids of already-completed (non-error) rows from predictions.jsonl.
    Used for resume logic.
    """
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not obj.get("error"):
                    done.add(obj["sample_id"])
            except Exception:
                pass
    return done


def run_system(request: RunSystemRequest, cfg: Any) -> RunSystemResponse:
    """
    Resumable batch runner.

    - Skips sample_ids already present in predictions.jsonl without error.
    - Opens predictions.jsonl in APPEND mode — never overwrites (Issue 6 fix).
    - Degrades per item: failures write an error row and continue (never aborts).
    - One PredictionRow per sample.

    Args:
        request: RunSystemRequest.
        cfg: Config object.

    Returns:
        RunSystemResponse with run_id, n_done, n_failed, predictions_path.
    """
    if request.system not in RUNNERS:
        raise ValueError(
            f"Unknown system '{request.system}'. Registered: {list(RUNNERS)}"
        )

    # Load samples
    from hmafact.data.loader import load_samples

    try:
        samples_dir = Path(cfg.data.samples_dir)  # type: ignore[attr-defined]
    except AttributeError:
        samples_dir = None

    samples = load_samples(
        dataset=request.dataset,
        split=request.split,
        final=request.final,
        limit=request.limit,
        data_dir=samples_dir,
    )

    # Resolve run_id and output path
    run_id = request.run_id or _make_run_id(request)
    try:
        runs_dir = Path(cfg.runs_dir)  # type: ignore[attr-defined]
    except AttributeError:
        runs_dir = Path("runs")

    pred_path = runs_dir / run_id / "predictions.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-done sample_ids
    done_ids = _read_done_ids(pred_path)
    if done_ids:
        logger.info(
            "run_system resuming: %d/%d already done for run_id=%s",
            len(done_ids), len(samples), run_id,
        )

    runner = RUNNERS[request.system]()
    failed = 0
    n_new = 0

    # APPEND mode — never overwrites existing predictions (Issue 6 fix)
    with pred_path.open("a", encoding="utf-8") as out_f:
        for sample in samples:
            if sample.sample_id in done_ids:
                continue

            row = runner.run(sample, cfg)

            if row.error:
                failed += 1
            else:
                n_new += 1

            out_f.write(row.model_dump_json() + "\n")
            out_f.flush()  # ensure crash-safety

    total_done = len(done_ids) + n_new
    logger.info(
        "run_system complete: run_id=%s  total_done=%d  new=%d  failed=%d  path=%s",
        run_id, total_done, n_new, failed, pred_path,
    )

    return RunSystemResponse(
        run_id=run_id,
        n_done=total_done,
        n_failed=failed,
        predictions_path=str(pred_path),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_id(request: RunSystemRequest) -> str:
    short_uuid = uuid.uuid4().hex[:8]
    return f"{request.system}_{request.dataset}_{request.split}_{short_uuid}"


def _resolve_role(role: str, cfg: Any) -> str | None:
    """Look up model_id for a role from cfg.llm.roles. Falls back to None."""
    try:
        return cfg.llm.roles.get(role)  # type: ignore[attr-defined]
    except AttributeError:
        try:
            return cfg.get("llm", {}).get("roles", {}).get(role)  # type: ignore[union-attr]
        except Exception:
            return None
