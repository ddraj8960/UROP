"""
Module 4 — Standard RAG Baseline (Retrieval-Augmented Generation).

Retrieves top-k passages from M3 Retrieval Engine and conditions M2 Generator on context.
Implements SystemRunner.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from hmafact.generator.generator import generate_answer
from hmafact.retrieval.hybrid import search
from hmafact.schemas.evaluation import PredictionRow, SystemName
from hmafact.schemas.evidence import EvidencePassage, SearchRequest
from hmafact.schemas.generator import GenerateRequest
from hmafact.schemas.input import BenchmarkSample

logger = logging.getLogger(__name__)

class RagRunner:
    """
    Standard RAG Baseline.
    Retrieves top-k passages using M3, then generates an answer using M2.
    Implements SystemRunner.
    """
    name: SystemName = "rag"

    def run(self, sample: BenchmarkSample, cfg: Any) -> PredictionRow:
        """Run one sample through the RAG baseline."""
        from hmafact.evaluation.runner import _resolve_role, _FEVER_FROM_SHORT
        start_ms = time.perf_counter() * 1000
        
        try:
            # 1. Retrieval using M3 batch handler
            # Retrieve k passages from configured sources
            k = getattr(cfg.retrieval, "top_k", 5) if hasattr(cfg, "retrieval") else 5
            sources = getattr(cfg.retrieval, "sources", ["local_wiki", "wiki_api"]) if hasattr(cfg, "retrieval") else ["local_wiki", "wiki_api"]
            
            search_req = SearchRequest(
                queries=[sample.input_text],
                sources=sources,
                k=k
            )
            search_res = search(search_req, cfg)
            
            # The search response contains all passages in a dict keyed by passage_id
            # Sort them by rank/score
            passages = list(search_res.passages.values())
            passages.sort(key=lambda p: (-p.retrieval_score, p.passage_id))
            passages = passages[:k]

            # 2. Format Context
            context_texts = [
                f"[{p.passage_id}] Source: {p.title}\n{p.text}"
                for p in passages
            ]

            # 3. Generation using M2
            mode = "claim" if sample.dataset == "fever" else "qa"
            
            # For FEVER, we must provide the original claim text
            generate_req = GenerateRequest(
                query_id=sample.sample_id,
                input_text=sample.input_text,
                mode=mode,
                dataset=sample.dataset,
                context_passages=context_texts,
            )

            # Route generator model
            model = _resolve_role("generator", cfg)
            
            # Execute Generation (must use RAG prompt)
            result = generate_answer(generate_req, model=model, temperature=0.0)

            latency_ms = time.perf_counter() * 1000 - start_ms

            # 4. Pipeline Trace Storage
            self._store_trace(sample.sample_id, passages, result, cfg)

            # Normalise FEVER label
            fever_label = None
            if mode == "claim":
                raw = (result.short_answer or "").strip()
                fever_label = _FEVER_FROM_SHORT.get(raw) or _FEVER_FROM_SHORT.get(raw.lower())

            # Abstention detection (if the model indicates insufficient evidence)
            abstained = False
            if result.short_answer and "INSUFFICIENT" in result.short_answer.upper():
                abstained = True
                
            return PredictionRow(
                sample_id=sample.sample_id,
                system=self.name,
                short_answer=result.short_answer,
                answer_text=f"{result.short_answer}. {result.long_answer}".strip(),
                abstained=abstained,
                fever_label=fever_label,
                confidence=None,
                latency_ms=latency_ms,
                n_llm_calls=1,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )

        except Exception as exc:
            latency_ms = time.perf_counter() * 1000 - start_ms
            logger.warning("RagRunner failed for %s: %s", sample.sample_id, exc)
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
            
    def _store_trace(self, sample_id: str, passages: list[EvidencePassage], result: Any, cfg: Any):
        """Write PipelineTrace to S12 trace storage."""
        try:
            runs_dir = Path(getattr(cfg, "runs_dir", "runs"))
            trace_dir = runs_dir / "traces" / self.name
            trace_dir.mkdir(parents=True, exist_ok=True)
            
            trace_file = trace_dir / f"{sample_id}.json"
            
            trace_data = {
                "sample_id": sample_id,
                "system": self.name,
                "state": {
                    "passages": [p.model_dump(mode="json") for p in passages],
                    "generator_info": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "short_answer": result.short_answer,
                        "long_answer": result.long_answer,
                    }
                }
            }
            
            trace_file.write_text(json.dumps(trace_data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to store trace for {sample_id}: {e}")
