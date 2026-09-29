#!/usr/bin/env python3
"""
scripts/test_evidence_graph.py — CLI Test Runner for S4 Dynamic Evidence Retrieval Graph (Module 6).

Usage:
    python scripts/test_evidence_graph.py --dataset truthfulqa --limit 2
    python scripts/test_evidence_graph.py --dataset fever --limit 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.claims.extractor import extract_claims
from hmafact.data.loader import load_samples
from hmafact.generator.generator import generate_answer
from hmafact.graph.builder import build_evidence_graph
from hmafact.schemas.claims import ExtractClaimsRequest
from hmafact.schemas.generator import GenerateRequest
from hmafact.schemas.graph import BuildGraphRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_evidence_graph_building(dataset: str = "truthfulqa", split: str = "dev", limit: int = 2) -> None:
    samples = load_samples(dataset=dataset, split=split)[:limit]

    print("\n" + "=" * 80)
    print(f" MODULE 6 — DYNAMIC EVIDENCE RETRIEVAL GRAPH TEST ({dataset.upper()} - {split.upper()})")
    print("=" * 80 + "\n")

    for i, s in enumerate(samples, 1):
        mode = "claim" if s.dataset == "fever" else "qa"

        # 1. Generate answer (M2)
        gen_res = generate_answer(
            GenerateRequest(query_id=s.sample_id, input_text=s.input_text, mode=mode, dataset=s.dataset)
        )

        # 2. Extract claims (M5)
        ext_res = extract_claims(
            ExtractClaimsRequest(query_id=s.sample_id, question=s.input_text, long_answer=gen_res.long_answer, mode=mode)
        )

        # 3. Build Evidence Graph (M6)
        graph_res = build_evidence_graph(
            BuildGraphRequest(
                query_id=s.sample_id,
                question=s.input_text,
                claims=[c.model_dump() for c in ext_res.claims],
                k_per_query=2,
            )
        )

        snap = graph_res.graph_snapshot
        node_link = snap.node_link

        # Count node & edge types
        node_types = Counter(n.get("ntype", "UNKNOWN") for n in node_link.get("nodes", []))
        edge_types = Counter(e.get("etype", "UNKNOWN") for e in node_link.get("links", []))

        print(f"[{i}/{len(samples)}] Query: {s.input_text}")
        print(f"Extracted Claims: {ext_res.n_claims} | Retrieved Passages: {graph_res.passage_count}")
        print(f"Graph Construction Statistics:")
        print(f"  - Total Nodes: {snap.node_count}  |  Total Edges: {snap.edge_count}")
        print(f"  - Node Types:  {dict(node_types)}")
        print(f"  - Edge Types:  {dict(edge_types)}")

        # Print sample evidence edges
        print(f"\n  Sample NLI Evidence Edges:")
        evidence_links = [e for e in node_link.get("links", []) if e.get("etype") in {"SUPPORTS", "CONTRADICTS", "RELATED_TO"}]
        for link in evidence_links[:3]:
            nli_dict = link.get("nli", {})
            print(f"    - [{link.get('etype')}] ({link.get('source')} -> {link.get('target')})")
            print(f"      NLI Entail: {nli_dict.get('entail')} | Neutral: {nli_dict.get('neutral')} | Contradict: {nli_dict.get('contradict')}")

        print("\n" + "=" * 80 + "\n")

        # Save sample graph snapshot
        out_file = Path("runs") / f"graph_{s.sample_id.replace(':', '_')}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(node_link, indent=2), encoding="utf-8")
        logger.info("Saved graph snapshot to %s", out_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Module 6 Dynamic Evidence Retrieval Graph Builder")
    parser.add_argument("--dataset", type=str, default="truthfulqa", choices=["truthfulqa", "fever", "hotpotqa"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    test_evidence_graph_building(dataset=args.dataset, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
