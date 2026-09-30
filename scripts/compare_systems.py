#!/usr/bin/env python3
"""
M4 Script: Milestone A Table (Vanilla vs RAG).
Runs evaluation across all dev splits and produces the comparison table.
"""
import argparse
import sys
from pathlib import Path
import logging

from hmafact.schemas.evaluation import RunSystemRequest, ComputeMetricsRequest
from hmafact.evaluation.runner import run_system
from hmafact.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockConfig:
    def __init__(self):
        self.runs_dir = "runs"
        class RetrievalCfg:
            def __init__(self):
                self.top_k = 5
                self.sources = ["local_wiki", "wiki_api"]
                self.index_dir = "data/indices"
        self.retrieval = RetrievalCfg()
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
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    cfg = MockConfig()
    
    datasets = ["truthfulqa", "hotpotqa", "fever"]
    systems = ["vanilla", "rag"]
    
    all_metrics = {sys: {} for sys in systems}

    for dset in datasets:
        for sys_name in systems:
            logger.info(f"Running {sys_name} on {dset}...")
            
            run_req = RunSystemRequest(
                system=sys_name,
                dataset=dset,
                split="dev",
                limit=args.limit,
                run_id=f"milestoneA_{sys_name}_{dset}"
            )
            run_res = run_system(run_req, cfg)
            
            metrics_req = ComputeMetricsRequest(
                run_id=run_res.run_id,
                dataset=dset,
                system=sys_name,
                predictions_path=run_res.predictions_path
            )
            metrics_res = compute_metrics(metrics_req, cfg)
            
            all_metrics[sys_name][dset] = metrics_res.metrics

    print("\n" + "="*90)
    print(" MILESTONE A: Vanilla vs Standard RAG ")
    print("="*90)
    print(f"{'Dataset':<15} | {'System':<10} | {'Accuracy':<10} | {'Abstention':<12} | {'Tokens':<10} | {'Latency':<10}")
    print("-" * 90)
    
    for dset in datasets:
        for sys_name in systems:
            metrics = all_metrics[sys_name][dset]
            acc = metrics.get("accuracy", 0.0)
            abstain = metrics.get("abstention_rate", 0.0)
            tokens = metrics.get("avg_completion_tokens", 0.0)
            latency = metrics.get("avg_latency_ms", 0.0)
            print(f"{dset:<15} | {sys_name:<10} | {acc:<10.4f} | {abstain:<12.4f} | {tokens:<10.1f} | {latency:<10.1f}")
            
    print("="*90)

if __name__ == "__main__":
    main()
