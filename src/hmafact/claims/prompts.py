"""
Prompts for Service S5 (Claim Extraction Agent).
"""
from __future__ import annotations
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

CLAIM_EXTRACTION_SYSTEM_PROMPT = (_PROMPTS_DIR / "extract_claims.txt").read_text(encoding="utf-8")
QUERY_GEN_SYSTEM_PROMPT = (_PROMPTS_DIR / "generate_queries.txt").read_text(encoding="utf-8")

def format_extraction_prompt(question: str, long_answer: str) -> str:
    return (
        f"Context Question: {question}\n"
        f"Generated Answer: {long_answer}\n\n"
        f"Decompose the Generated Answer into atomic claims in JSON format."
    )

def format_query_gen_prompt(claims: list[str]) -> str:
    claim_lines = "\n".join(f"[{i}] {claim}" for i, claim in enumerate(claims))
    return (
        f"Generate exactly 2 distinct search queries for each of the following claims.\n\n"
        f"Claims:\n{claim_lines}\n\n"
        f"Output JSON format:\n{{\n"
        f"  \"0\": [\"query 1\", \"query 2\"],\n"
        f"  \"1\": [\"query 1\", \"query 2\"]\n}}"
    )
