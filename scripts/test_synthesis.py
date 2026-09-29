"""
CLI Test Script for Module 8: S8 Logic Validation, S9 Confidence Estimator, S10 Response Synthesis.
Runs end-to-end pipeline up to final fact-corrected response synthesis.
"""
from __future__ import annotations

import json
from hmafact.llm.openrouter import OpenRouterClient
from hmafact.schemas.claims import ExtractClaimsRequest
from hmafact.schemas.graph import BuildGraphRequest
from hmafact.claims.extractor import extract_claims
from hmafact.graph.builder import build_evidence_graph
from hmafact.verification.verifier import FactVerifier
from hmafact.logic.validator import validate_claims_logic
from hmafact.synthesis.synthesizer import ResponseSynthesizer


def main():
    print("=== Module 8: Response Synthesis & Logic Validation Test ===")

    question = "Where is the Eiffel Tower located and when was it built?"
    original_answer = (
        "The Eiffel Tower is located in Rome, Italy. "
        "It was constructed in 1889 for the Exposition Universelle."
    )

    print(f"\nQuestion: {question}")
    print(f"Original Answer: {original_answer}\n")

    # Step 1: Extract claims
    claims_resp = extract_claims(ExtractClaimsRequest(query_id="demo_m8", question=question, long_answer=original_answer))
    print(f"Extracted {claims_resp.n_claims} atomic claims.")

    # Step 2: Validate inter-claim logic
    logic_res = validate_claims_logic("demo_m8", claims_resp.claims)
    print(f"Logic Validation Score: {logic_res.logic_score} (Consistent: {logic_res.is_consistent})")

    # Step 3: Build Evidence Graph
    graph_resp = build_evidence_graph(BuildGraphRequest(
        query_id="demo_m8",
        question=question,
        claims=[c.model_dump() for c in claims_resp.claims],
        k_per_query=2,
    ))

    # Step 4: Verify Claims
    llm_client = None
    try:
        llm_client = OpenRouterClient()
        print("Initialized OpenRouter client for verification & synthesis.")
    except Exception as e:
        print(f"OpenRouter unavailable ({e}); using algorithmic fallbacks.")

    verifier = FactVerifier(llm_client=llm_client)
    ver_resp = verifier.verify_claims_graph(
        query_id="demo_m8",
        question=question,
        claims=claims_resp.claims,
        graph=graph_resp.graph_snapshot,
        use_llm=llm_client is not None,
    )

    # Step 5: Synthesize Final Fact-Corrected Answer
    synthesizer = ResponseSynthesizer(llm_client=llm_client)
    final_resp = synthesizer.synthesize(
        query_id="demo_m8",
        question=question,
        original_answer=original_answer,
        claims=claims_resp.claims,
        verification_response=ver_resp,
        logic_result=logic_res,
        use_llm=llm_client is not None,
    )

    print("\n=== FINAL SYNTHESIZED RESPONSE ===")
    print(f"Final Answer:\n{final_resp.final_answer}\n")
    print(f"Overall Verdict: {final_resp.overall_verdict}")
    print(f"Overall Confidence: {final_resp.confidence.overall_confidence}")
    print(f"  - Verification Ratio: {final_resp.confidence.verification_ratio}")
    print(f"  - Evidence Coverage: {final_resp.confidence.evidence_coverage}")
    print(f"  - Logic Consistency: {final_resp.confidence.logic_consistency}")
    print(f"Corrections Made: {final_resp.corrections_made}")
    print(f"Citations Used: {final_resp.citations_used}")


if __name__ == "__main__":
    main()
