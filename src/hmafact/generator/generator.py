"""
Service S2 Generator Implementation.
Generates initial responses for Vanilla baseline and RAG pipeline.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from hmafact.generator.prompts import (
    CLAIM_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    format_claim_user_prompt,
    format_qa_user_prompt,
)
from hmafact.llm.openrouter import generate_response
from hmafact.schemas.generator import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


def _parse_llm_json(raw_text: str) -> tuple[str, str]:
    """
    Parse JSON output from LLM, with fallback for markdown codeblocks or plain text.
    Returns (short_answer, long_answer).
    """
    cleaned = raw_text.strip()
    # Strip ```json ... ``` codeblock markers if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            short = str(data.get("short_answer", "")).strip()
            long = str(data.get("long_answer", "")).strip()
            if short or long:
                return short or raw_text[:50], long or raw_text
    except Exception:
        pass

    # Fallback parsing if LLM didn't format valid JSON
    first_line = raw_text.splitlines()[0] if raw_text else ""
    short = first_line[:60].strip()
    long = raw_text.strip()
    return short, long


def generate_answer(
    request: GenerateRequest,
    model: str | None = None,
    temperature: float = 0.0,
) -> GenerateResponse:
    """
    Execute initial generator LLM call for a query.

    Args:
        request: GenerateRequest containing query_id, input_text, mode, context.
        model: Model identifier override (optional).
        temperature: Sampling temperature (default 0.0 for baseline).

    Returns:
        GenerateResponse object.
    """
    if request.mode == "claim":
        sys_prompt = CLAIM_SYSTEM_PROMPT
        user_prompt = format_claim_user_prompt(request.input_text)
    else:
        sys_prompt = QA_SYSTEM_PROMPT
        user_prompt = format_qa_user_prompt(request.input_text, request.context_passages)

    llm_res = generate_response(
        prompt=user_prompt,
        system_prompt=sys_prompt,
        model=model,
        temperature=temperature,
        max_tokens=300,
    )

    raw_text = llm_res["text"]
    short_ans, long_ans = _parse_llm_json(raw_text)

    return GenerateResponse(
        query_id=request.query_id,
        input_text=request.input_text,
        short_answer=short_ans,
        long_answer=long_ans,
        raw_text=raw_text,
        model=llm_res["model"],
        latency_sec=llm_res["latency_sec"],
        prompt_tokens=llm_res["prompt_tokens"],
        completion_tokens=llm_res["completion_tokens"],
        meta={"dataset": request.dataset, "mode": request.mode},
    )
