"""
Prompts for S2 Generator (Vanilla Baseline).
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
Evaluate the given statement.
Format your output as JSON with two keys:
- "short_answer": One of ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
- "long_answer": A 1-2 sentence explanation of why the claim is true, false, or unverified.
Do NOT include markdown formatting outside the JSON block.
"""


def format_qa_user_prompt(question: str, context: list[str] | None = None) -> str:
    if context:
        ctx_text = "\n".join(f"- {c}" for c in context)
        return f"Context Information:\n{ctx_text}\n\nQuestion: {question}\n\nProvide short_answer and long_answer in JSON format."
    return f"Question: {question}\n\nProvide short_answer and long_answer in JSON format."


def format_claim_user_prompt(claim: str) -> str:
    return f"Claim: {claim}\n\nProvide short_answer and long_answer in JSON format."
