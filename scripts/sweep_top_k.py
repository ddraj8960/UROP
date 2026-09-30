#!/usr/bin/env python3
"""
M4 Script: Sweep top-k values for RAG baseline.
Executes RAG evaluation for k in [3, 5, 10] across dev splits.
"""
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
import logging

from hmafact.schemas.evaluation import RunSystemRequest, ComputeMetricsRequest
from hmafact.evaluation.runner import run_system
from hmafact.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockConfig:
    def __init__(self, top_k=5):
        self.runs_dir = "runs"
        class RetrievalCfg:
            def __init__(self, top_k):
                self.top_k = top_k
                self.sources = ["local_wiki", "wiki_api"]
                self.index_dir = "data/indices"
        self.retrieval = RetrievalCfg(top_k)
        class DataCfg:
            def __init__(self):
                self.samples_dir = "data/samples"
        self.data = DataCfg()
        class LlmCfg:
            def __init__(self):
                self.roles = {"generator": "openrouter/auto"}
        self.llm = LlmCfg()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fever", choices=["fever", "hotpotqa", "truthfulqa"])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=50, help="Max samples per run")
    args = parser.parse_args()

    k_values = [3, 5, 10]
    results = {}

    for k in k_values:
        logger.info(f"--- Sweeping top-k = {k} on {args.dataset} ---")
        cfg = MockConfig(top_k=k)
        
        run_req = RunSystemRequest(
            system="rag",
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            run_id=f"sweep_rag_{args.dataset}_k{k}"
        )
        
        # Run Evaluation
        run_res = run_system(run_req, cfg)
        logger.info(f"Finished generation. Failed: {run_res.n_failed}/{run_res.n_done}")
        
        # Compute Metrics
        metrics_req = ComputeMetricsRequest(
            run_id=run_res.run_id,
            dataset=args.dataset,
            system="rag",
            predictions_path=run_res.predictions_path
        )
        metrics_res = compute_metrics(metrics_req, cfg)
        
        results[k] = metrics_res.metrics
        
    print("\n--- Sweep Results ---")
    for k, metrics in results.items():
        acc = metrics.get("accuracy", 0.0)
        abstain = metrics.get("abstention_rate", 0.0)
        print(f"k={k:2d} | Accuracy: {acc:.4f} | Abstention: {abstain:.4f}")

if __name__ == "__main__":
    main()
