"""
Prompts for S2 Generator (Vanilla Baseline and RAG).
"""
from __future__ import annotations

QA_SYSTEM_PROMPT = """You are a helpful, factual AI assistant.
Answer the user's question accurately and concisely.
Format your output as JSON with two keys:
- "short_answer": A direct 1-5 word concise answer to the question.
- "long_answer": A full 1-3 sentence explanation supporting the short answer.
Do NOT include markdown formatting outside the JSON block.
"""

CLAIM_SYSTEM_PROMPT = """You are a fact-checking assistant.
Evaluate the given statement against your knowledge.
Format your output as JSON with two keys:
- "short_answer": EXACTLY one of ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
  — use "SUPPORTS" if the claim is supported by evidence,
    "REFUTES" if the claim is contradicted by evidence,
    "NOT ENOUGH INFO" if there is insufficient evidence to decide.
- "long_answer": A 1-2 sentence explanation of why the claim is supported, refuted, or undecidable.
Do NOT include markdown formatting outside the JSON block.
Do NOT use any other label values.
"""

RAG_QA_SYSTEM_PROMPT = """You are a helpful, factual AI assistant.
Answer the user's question accurately and concisely based ONLY on the provided context passages.
If the context does not contain the answer, your short_answer MUST be exactly "INSUFFICIENT".
Format your output as JSON with two keys:
- "short_answer": A direct 1-5 word concise answer to the question, or "INSUFFICIENT" if missing.
- "long_answer": A full 1-3 sentence explanation supporting the short answer. Cite passage IDs [e.g., [local_wiki:doc:0]].
Do NOT include markdown formatting outside the JSON block.
"""

RAG_CLAIM_SYSTEM_PROMPT = """You are a fact-checking assistant.
Evaluate the given statement based ONLY on the provided context passages.
Format your output as JSON with two keys:
- "short_answer": EXACTLY one of ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
  — use "SUPPORTS" if the claim is supported by context,
    "REFUTES" if the claim is contradicted by context,
    "NOT ENOUGH INFO" if there is insufficient context to decide.
- "long_answer": A 1-2 sentence explanation citing specific passage IDs (e.g. [wiki_api:123:0]).
Do NOT include markdown formatting outside the JSON block.
"""

def format_qa_user_prompt(question: str, context: list[str] | None = None) -> str:
    if context:
        ctx_text = "\n".join(f"- {c}" for c in context)
        return f"Context Passages:\n{ctx_text}\n\nQuestion: {question}\n\nProvide short_answer and long_answer in JSON format."
    return f"Question: {question}\n\nProvide short_answer and long_answer in JSON format."


def format_claim_user_prompt(claim: str, context: list[str] | None = None) -> str:
    if context:
        ctx_text = "\n".join(f"- {c}" for c in context)
        return f"Context Passages:\n{ctx_text}\n\nClaim: {claim}\n\nProvide short_answer and long_answer in JSON format."
    return f"Claim: {claim}\n\nProvide short_answer and long_answer in JSON format."
