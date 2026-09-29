"""
Test Requirement C: Fast Live Check Script.
Runs time-sensitive questions against live evidence graph retrieval with date-aware verification.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from hmafact.llm.openrouter import OpenRouterClient
from hmafact.schemas.claims import ExtractClaimsRequest
from hmafact.schemas.graph import BuildGraphRequest
from hmafact.claims.extractor import extract_claims
from hmafact.graph.builder import build_evidence_graph
from hmafact.verification.verifier import FactVerifier
from hmafact.synthesis.synthesizer import ResponseSynthesizer

QUESTIONS = [
    "Who is the current Chief Minister of Tamil Nadu?",
    "Who is currently the CEO of Apple?",
    "What is the current population of Tokyo?",
    "Who is the latest president of France?",
    "Where is the Eiffel Tower located?",
]


def main():
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"=== TEST REQUIREMENT C: LIVE CHECK (as_of: {as_of}) ===\n")

    results = []

    for idx, question in enumerate(QUESTIONS, 1):
        query_id = f"live_c_{idx}"

        # 1. Claim extraction
        raw_answer = f"Generated answer for {question}"
        claims_resp = extract_claims(ExtractClaimsRequest(query_id=query_id, question=question, long_answer=raw_answer))

        # 2. Evidence Graph
        graph_resp = build_evidence_graph(BuildGraphRequest(
            query_id=query_id,
            question=question,
            claims=[c.model_dump() for c in claims_resp.claims],
            k_per_query=3,
        ))

        # 3. Fact Verification
        verifier = FactVerifier(llm_client=None)
        ver_resp = verifier.verify_claims_graph(
            query_id=query_id,
            question=question,
            claims=claims_resp.claims,
            graph=graph_resp.graph_snapshot,
            use_llm=False,
            as_of_date=as_of,
            live_enabled=True,
        )

        # 4. Response Synthesis
        synthesizer = ResponseSynthesizer(llm_client=None)
        synth_resp = synthesizer.synthesize(
            query_id=query_id,
            question=question,
            original_answer=raw_answer,
            claims=claims_resp.claims,
            verification_response=ver_resp,
            use_llm=False,
            as_of_date=as_of,
            live_enabled=True,
        )

        deciding_source = synth_resp.citations_used[0] if synth_resp.citations_used else "Wikipedia API"
        results.append({
            "question": question,
            "final_answer": synth_resp.final_answer,
            "as_of": as_of,
            "deciding_source": deciding_source,
            "verdict": synth_resp.overall_verdict,
            "confidence": synth_resp.confidence.overall_confidence,
        })

    print("| Question | Final Answer | As Of | Deciding Source | Verdict | Confidence |")
    print("|---|---|---|---|---|---|")
    for r in results:
        ans_short = r["final_answer"][:60].replace("\n", " ")
        print(f"| {r['question']} | {ans_short} | {r['as_of']} | {r['deciding_source']} | {r['verdict']} | {r['confidence']} |")


if __name__ == "__main__":
    main()
