#!/usr/bin/env python3
"""
scripts/run_generator_baseline.py — CLI Runner for S2 Vanilla Baseline.

Usage:
    python scripts/run_generator_baseline.py --dataset truthfulqa --split dev --limit 5
    python scripts/run_generator_baseline.py --dataset fever --split dev --limit 5
    python scripts/run_generator_baseline.py --dataset hotpotqa --split dev --limit 5
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

from hmafact.data.loader import load_samples
from hmafact.generator.generator import generate_answer
from hmafact.schemas.generator import GenerateRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_generator_eval(
    dataset: str = "truthfulqa",
    split: str = "dev",
    limit: int | None = None,
    output_dir: Path = Path("runs"),
) -> None:
    logger.info("Loading %s_%s benchmark samples...", dataset, split)
    samples = load_samples(dataset=dataset, split=split)

    if limit is not None:
        samples = samples[:limit]
        logger.info("Running generator baseline on subset of %d items...", len(samples))

    results = []
    start_total = time.time()

    print("\n" + "=" * 75)
    print(f" MODULE 2 — S2 GENERATOR VANILLA BASELINE ({dataset.upper()} - {split.upper()})")
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

        res = generate_answer(req)

        print(f"Short Answer: {res.short_answer}")
        print(f"Long Answer:  {res.long_answer}")
        print(f"Latency:      {res.latency_sec}s  | Model: {res.model}\n" + "-" * 75)

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
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
        }
        results.append(record)

    total_time = round(time.time() - start_total, 2)
    output_path = output_dir / f"vanilla_{dataset}_{split}_predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("M2 Generator baseline complete. Wrote %d predictions to %s (Total time: %ss)",
                len(results), output_path, total_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 2 S2 Generator Vanilla Baseline")
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "fever", "hotpotqa"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    run_generator_eval(dataset=args.dataset, split=args.split, limit=args.limit, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
