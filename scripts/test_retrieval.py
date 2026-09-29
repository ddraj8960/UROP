#!/usr/bin/env python3
"""
scripts/test_retrieval.py — Test CLI Runner for S3 Retrieval Service (Module 3).

Usage:
    python scripts/test_retrieval.py --query "What brand of cigarettes do doctors recommend?"
    python scripts/test_retrieval.py --dataset fever --limit 3
    python scripts/test_retrieval.py --dataset hotpotqa --limit 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.data.loader import load_samples
from hmafact.retrieval.hybrid import search_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_custom_query(query: str, k: int = 5) -> None:
    print(f"\nSearching Wikipedia evidence for query: '{query}' (k={k})...\n")
    passages = search_evidence(query=query, k=k)

    if not passages:
        print("No evidence passages found.")
        return

    for p in passages:
        print(f"[{p.rank}] {p.title} (Score: {p.retrieval_score})")
        print(f"    URL: {p.url}")
        print(f"    Snippet: {p.text}\n" + "-" * 70)


def test_dataset_samples(dataset: str = "truthfulqa", split: str = "dev", limit: int = 3, k: int = 3) -> None:
    samples = load_samples(dataset=dataset, split=split)
    sample_subset = samples[:limit]
    print(f"\n======================================================================")
    print(f" MODULE 3 — RETRIEVAL SERVICE TEST ({dataset.upper()} - {split.upper()})")
    print(f"======================================================================\n")

    for i, s in enumerate(sample_subset, 1):
        print(f"[{i}/{limit}] Sample ID: {s.sample_id}")
        print(f"Input: {s.input_text}")

        passages = search_evidence(query=s.input_text, k=k)
        print(f"Retrieved {len(passages)} passages:")
        for p in passages:
            print(f"  [{p.rank}] {p.title} (score: {p.retrieval_score}) -> \"{p.text[:100]}...\"")
        print("\n" + "-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Module 3 Retrieval Service")
    parser.add_argument("--query", type=str, default=None, help="Custom search query text")
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "fever", "hotpotqa"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=3, help="Number of benchmark samples to test")
    parser.add_argument("-k", type=int, default=3, help="Top k passages per query")
    args = parser.parse_args()

    if args.query:
        test_custom_query(query=args.query, k=args.k)
    else:
        test_dataset_samples(dataset=args.dataset, split=args.split, limit=args.limit, k=args.k)


if __name__ == "__main__":
    main()
