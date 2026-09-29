"""
Module 4 — Standard RAG Baseline (Retrieval-Augmented Generation).

Retrieves top-k passages from M3 Retrieval Engine and conditions M2 Generator on context.
"""
from __future__ import annotations

import logging
from typing import Any

from hmafact.generator.generator import generate_answer
from hmafact.retrieval.hybrid import search_evidence
from hmafact.schemas.evidence import EvidencePassage
from hmafact.schemas.generator import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


def run_rag_baseline(
    request: GenerateRequest,
    top_k: int = 5,
    model: str | None = None,
    temperature: float = 0.0,
) -> tuple[GenerateResponse, list[EvidencePassage]]:
    """
    Execute Standard RAG Baseline pipeline for a query.

    1. Retrieve top-k evidence passages using M3 search_evidence().
    2. Format retrieved text as context.
    3. Pass context + query to M2 generate_answer().

    Args:
        request: Input GenerateRequest.
        top_k: Number of evidence passages to retrieve (default 5).
        model: Optional LLM model override.
        temperature: LLM sampling temperature.

    Returns:
        Tuple of (GenerateResponse, list[EvidencePassage]).
    """
    logger.info("RAG Baseline: Retrieving top-%d passages for query: '%s'", top_k, request.input_text)
    passages = search_evidence(query=request.input_text, k=top_k)

    # Format context passage snippets
    context_texts: list[str] = [
        f"[{p.rank}] Source: {p.title}\n{p.text}"
        for p in passages
    ]

    # Create modified request with context passages
    rag_request = request.model_copy(
        update={"context_passages": context_texts}
    )

    # Call Generator LLM with RAG context
    res = generate_answer(rag_request, model=model, temperature=temperature)

    # Attach RAG passage metadata to response
    passage_meta = [
        {
            "passage_id": p.passage_id,
            "title": p.title,
            "url": p.url,
            "score": p.retrieval_score,
            "rank": p.rank,
        }
        for p in passages
    ]

    updated_res = res.model_copy(
        update={
            "meta": {
                **res.meta,
                "rag_enabled": True,
                "retrieved_passages": passage_meta,
            }
        }
    )

    return updated_res, passages
