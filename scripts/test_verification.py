"""
CLI Test Script for Module 7: S6 Evidence Fusion & S7 Fact Verification Agent.
Runs end-to-end extraction -> retrieval -> graph construction -> fusion -> verification.
"""
from __future__ import annotations

import json
from hmafact.llm.openrouter import OpenRouterClient
from hmafact.schemas.claims import ExtractClaimsRequest
from hmafact.schemas.graph import BuildGraphRequest
from hmafact.claims.extractor import extract_claims
from hmafact.graph.builder import build_evidence_graph
from hmafact.verification.verifier import FactVerifier


def main():
    print("=== Module 7: Fact Verification Agent & Fusion Test ===")

    question = "Where is the Eiffel Tower located and when was it built?"
    long_answer = (
        "The Eiffel Tower is located in Rome, Italy. "
        "It was constructed in 1889 for the Exposition Universelle."
    )

    print(f"\nQuestion: {question}")
    print(f"Long Answer: {long_answer}\n")

    # Step 1: Extract claims
    req_extract = ExtractClaimsRequest(query_id="demo_m7", question=question, long_answer=long_answer)
    claims_resp = extract_claims(req_extract)
    print(f"Extracted {claims_resp.n_claims} atomic claims:")
    for c in claims_resp.claims:
        print(f"  - [{c.claim_id}] {c.text} (Type: {c.claim_type})")

    # Step 2: Build Evidence Graph
    print("\nBuilding Evidence Graph...")
    req_graph = BuildGraphRequest(
        query_id="demo_m7",
        question=question,
        claims=[c.model_dump() for c in claims_resp.claims],
        k_per_query=2,
    )
    graph_resp = build_evidence_graph(req_graph)
    print(f"Graph constructed: {graph_resp.claim_count} claims, {graph_resp.passage_count} passages.")

    # Step 3: Initialize Fact Verifier
    llm_client = None
    try:
        llm_client = OpenRouterClient()
        print("Initialized OpenRouter client for LLM-based verification explanations.")
    except Exception as e:
        print(f"OpenRouter unavailable ({e}), using pure algorithmic verification.")

    verifier = FactVerifier(llm_client=llm_client)

    # Step 4: Run Verification
    print("\nRunning Fact Verification Agent...")
    ver_resp = verifier.verify_claims_graph(
        query_id="demo_m7",
        question=question,
        claims=claims_resp.claims,
        graph=graph_resp.graph_snapshot,
        use_llm=llm_client is not None,
    )

    print("\n=== VERIFICATION RESULTS ===")
    print(f"Overall Verdict: {ver_resp.overall_verdict}")
    print(f"Supported: {ver_resp.supported_count} | Contradicted: {ver_resp.contradicted_count} | Insufficient: {ver_resp.insufficient_count}\n")

    for r in ver_resp.results:
        print(f"Claim ID: {r.claim_id}")
        print(f"  Claim: \"{r.claim_text}\"")
        print(f"  Verdict: {r.verdict} (Confidence: {r.confidence})")
        print(f"  Explanation: {r.explanation}")
        if r.citations:
            print(f"  Citations: {r.citations}")
        if r.corrected_text:
            print(f"  Corrected Statement: \"{r.corrected_text}\"")
        print("-" * 50)


if __name__ == "__main__":
    main()
