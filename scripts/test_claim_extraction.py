#!/usr/bin/env python3
"""
scripts/test_claim_extraction.py — CLI Test Runner for Service S5 Claim Extraction Agent (Module 5).

Usage:
    python scripts/test_claim_extraction.py --dataset truthfulqa --limit 3
    python scripts/test_claim_extraction.py --dataset fever --limit 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.claims.extractor import extract_claims, generate_claim_queries
from hmafact.data.loader import load_samples
from hmafact.generator.generator import generate_answer
from hmafact.schemas.claims import ExtractClaimsRequest, GenerateQueriesRequest
from hmafact.schemas.generator import GenerateRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_claim_extraction(dataset: str = "truthfulqa", split: str = "dev", limit: int = 3) -> None:
    samples = load_samples(dataset=dataset, split=split)[:limit]

    print("\n" + "=" * 75)
    print(f" MODULE 5 — CLAIM EXTRACTION AGENT TEST ({dataset.upper()} - {split.upper()})")
    print("=" * 75 + "\n")

    for i, s in enumerate(samples, 1):
        mode = "claim" if s.dataset == "fever" else "qa"
        gen_req = GenerateRequest(
            query_id=s.sample_id,
            input_text=s.input_text,
            mode=mode,
            dataset=s.dataset,
        )

        # 1. Generate answer (M2)
        gen_res = generate_answer(gen_req)

        # 2. Extract atomic claims (M5)
        ext_req = ExtractClaimsRequest(
            query_id=s.sample_id,
            question=s.input_text,
            long_answer=gen_res.long_answer,
            mode=mode,
        )
        ext_res = extract_claims(ext_req)

        # 3. Generate search queries (M5)
        queries_res = generate_claim_queries(GenerateQueriesRequest(claims=ext_res.claims))

        print(f"[{i}/{len(samples)}] Question / Claim: {s.input_text}")
        print(f"Generated Answer: {gen_res.long_answer}")
        print(f"\nExtracted {ext_res.n_claims} Atomic Claims:")

        for c in ext_res.claims:
            print(f"  - Claim [{c.claim_id}] (Type: {c.claim_type}):")
            print(f"    Text:     \"{c.text}\"")
            print(f"    Entities: {c.entities}")
            print(f"    Queries:  {queries_res.queries.get(c.claim_id, [])}")

        print("\n" + "-" * 75 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Module 5 Claim Extraction Agent")
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "fever", "hotpotqa"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to evaluate")
    args = parser.parse_args()

    test_claim_extraction(dataset=args.dataset, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
