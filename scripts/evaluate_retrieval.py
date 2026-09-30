#!/usr/bin/env python3
"""
M3 Script: Evaluate retrieval performance.
"""
import argparse
import sys

from hmafact.schemas.evidence import EvalRetrievalRequest
from hmafact.retrieval.eval import evaluate_retrieval

def main():
    parser = argparse.ArgumentParser(description="Evaluate HMA-Fact retrieval.")
    parser.add_argument("--split", default="dev", help="Dataset split")
    parser.add_argument("--dataset", default="fever", choices=["fever", "hotpotqa", "all"])
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    print(f"Evaluating {args.mode} retrieval on {args.dataset} {args.split} (max {args.max_samples} samples)...")
    req = EvalRetrievalRequest(
        split=args.split,
        dataset=args.dataset,
        mode=args.mode,
        max_samples=args.max_samples
    )
    
    try:
        resp = evaluate_retrieval(req)
        print("Evaluation complete.")
        for metric, val in resp.metrics.items():
            print(f"  {metric}: {val:.4f}")
    except Exception as e:
        print(f"Error evaluating retrieval: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
