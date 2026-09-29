#!/usr/bin/env python3
"""
scripts/test_openrouter.py — Run live OpenRouter baseline evaluation test on benchmark samples.

Usage:
    python scripts/test_openrouter.py --dataset truthfulqa --limit 3
    python scripts/test_openrouter.py --dataset fever --limit 3
    python scripts/test_openrouter.py --dataset hotpotqa --limit 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.data.loader import load_samples
from hmafact.llm.openrouter import generate_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_baseline_test(dataset: str = "truthfulqa", split: str = "dev", limit: int = 3) -> None:
    logger.info("Loading %s_%s samples...", dataset, split)
    samples = load_samples(dataset=dataset, split=split)
    sample_subset = samples[:limit]
    logger.info("Fetched %d total samples. Running OpenRouter on first %d items...\n", len(samples), len(sample_subset))

    results = []
    print("=" * 70)
    print(f" HMA-FACT OPENROUTER BASELINE TEST ({dataset.upper()} - {split.upper()})")
    print("=" * 70 + "\n")

    for i, s in enumerate(sample_subset, 1):
        print(f"[{i}/{limit}] Sample ID: {s.sample_id}")
        print(f"Input ({s.dataset}): {s.input_text}")
        if s.gold_answer:
            print(f"Gold Answer: {s.gold_answer}")
        if s.gold_label:
            print(f"Gold Label:  {s.gold_label}")

        # Send prompt to OpenRouter
        try:
            res = generate_response(prompt=s.input_text)
            print(f"Model Output ({res['model']}, {res['latency_sec']}s):")
            print(f"  -> \"{res['text']}\"\n")

            results.append({
                "sample_id": s.sample_id,
                "dataset": s.dataset,
                "split": s.split,
                "input_text": s.input_text,
                "prediction": res["text"],
                "gold_answer": s.gold_answer,
                "gold_label": s.gold_label,
                "gold_evidence": s.gold_evidence,
                "model": res["model"],
                "latency_sec": res["latency_sec"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
            })
        except Exception as e:
            print(f"[ERROR] Querying OpenRouter: {e}\n")

        print("-" * 70)

    # Save outputs to runs/
    output_path = Path("runs") / f"openrouter_{dataset}_{split}_predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("Saved %d predictions to %s", len(results), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenRouter integration with benchmark samples.")
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "fever", "hotpotqa"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to evaluate")
    args = parser.parse_args()

    run_baseline_test(dataset=args.dataset, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
