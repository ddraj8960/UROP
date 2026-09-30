#!/usr/bin/env python3
"""
scripts/run_rag_baseline.py — CLI Runner for Module 4 Standard RAG Baseline.

Usage:
    python scripts/run_rag_baseline.py --dataset truthfulqa --split dev --limit 5
    python scripts/run_rag_baseline.py --dataset fever --split dev --limit 5
    python scripts/run_rag_baseline.py --dataset hotpotqa --split dev --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.baselines.rag import run_rag_baseline
from hmafact.data.loader import load_samples
from hmafact.schemas.generator import GenerateRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_rag_eval(
    dataset: str = "truthfulqa",
    split: str = "dev",
    limit: int | None = None,
    top_k: int = 5,
    output_dir: Path = Path("runs"),
) -> None:
    logger.info("Loading %s_%s benchmark samples for Standard RAG...", dataset, split)
    samples = load_samples(dataset=dataset, split=split)

    if limit is not None:
        samples = samples[:limit]
        logger.info("Running RAG baseline on subset of %d items...", len(samples))

    results = []
    start_total = time.time()

    print("\n" + "=" * 75)
    print(f" MODULE 4 — STANDARD RAG BASELINE ({dataset.upper()} - {split.upper()})")
    print("=" * 75 + "\n")

    for i, s in enumerate(samples, 1):
        mode = "claim" if s.dataset == "fever" else "qa"
        req = GenerateRequest(
            query_id=s.sample_id,
            input_text=s.input_text,
            mode=mode,
            dataset=s.dataset,
        )

        print(f"[{i}/{len(samples)}] Sample ID: {s.sample_id}")
        print(f"Input ({mode}): {s.input_text}")

        res, passages = run_rag_baseline(req, top_k=top_k)

        print(f"Retrieved {len(passages)} passages:")
        for p in passages:
            print(f"  - [{p.rank}] {p.title} (score: {p.retrieval_score})")

        print(f"RAG Short Answer: {res.short_answer}")
        print(f"RAG Long Answer:  {res.long_answer}")
        print(f"Latency:          {res.latency_sec}s | Model: {res.model}\n" + "-" * 75)

        record = {
            "sample_id": s.sample_id,
            "dataset": s.dataset,
            "split": s.split,
            "input_text": s.input_text,
            "gold_answer": s.gold_answer,
            "gold_label": s.gold_label,
            "short_answer": res.short_answer,
            "long_answer": res.long_answer,
            "raw_text": res.raw_text,
            "model": res.model,
            "latency_sec": res.latency_sec,
            "retrieved_passages": [p.model_dump() for p in passages],
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
        }
        results.append(record)

    total_time = round(time.time() - start_total, 2)
    output_path = output_dir / f"rag_{dataset}_{split}_predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode — never overwrites existing predictions (Issue 6 fix)
    with open(output_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("M4 RAG baseline complete. Wrote %d predictions to %s (Total time: %ss)",
                len(results), output_path, total_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 4 Standard RAG Baseline Evaluation")
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "fever", "hotpotqa"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("-k", type=int, default=5, help="Number of context passages to retrieve")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    run_rag_eval(dataset=args.dataset, split=args.split, limit=args.limit, top_k=args.k, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
