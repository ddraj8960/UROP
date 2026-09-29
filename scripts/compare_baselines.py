#!/usr/bin/env python3
"""
scripts/compare_baselines.py — Compare Vanilla (M2) vs Standard RAG (M4) side-by-side.

Usage:
    python scripts/compare_baselines.py --dataset truthfulqa --limit 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.baselines.rag import run_rag_baseline
from hmafact.data.loader import load_samples
from hmafact.generator.generator import generate_answer
from hmafact.schemas.generator import GenerateRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compare_baselines(dataset: str = "truthfulqa", split: str = "dev", limit: int = 3) -> None:
    samples = load_samples(dataset=dataset, split=split)[:limit]

    print("\n" + "=" * 80)
    print(f" SIDE-BY-SIDE COMPARISON: VANILLA (M2) vs STANDARD RAG (M4) [{dataset.upper()}]")
    print("=" * 80 + "\n")

    for i, s in enumerate(samples, 1):
        mode = "claim" if s.dataset == "fever" else "qa"
        req = GenerateRequest(
            query_id=s.sample_id,
            input_text=s.input_text,
            mode=mode,
            dataset=s.dataset,
        )

        # 1. Run Vanilla Baseline
        vanilla_res = generate_answer(req)

        # 2. Run Standard RAG Baseline
        rag_res, passages = run_rag_baseline(req, top_k=3)

        print(f"[{i}/{len(samples)}] Question / Claim: {s.input_text}")
        if s.gold_answer:
            print(f"  Gold Answer: {s.gold_answer}")
        if s.gold_label:
            print(f"  Gold Label:  {s.gold_label}")

        print(f"\n  [M2 Vanilla Baseline - No RAG]:")
        print(f"     Short Answer: {vanilla_res.short_answer}")
        print(f"     Long Answer:  {vanilla_res.long_answer[:120]}...")

        print(f"\n  [M4 Standard RAG Baseline - With Wikipedia RAG]:")
        print(f"     Retrieved:    [{', '.join(p.title for p in passages[:2])}]")
        print(f"     Short Answer: {rag_res.short_answer}")
        print(f"     Long Answer:  {rag_res.long_answer[:120]}...")

        print("\n" + "=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Vanilla vs RAG baselines side-by-side")
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "fever", "hotpotqa"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    compare_baselines(dataset=args.dataset, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
