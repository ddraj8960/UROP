"""
Service S5 Claim Extraction Implementation.
Decomposes LLM answers into atomic, decontextualized claims and extracts entities.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from hmafact.claims.prompts import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    QUERY_GEN_SYSTEM_PROMPT,
    format_extraction_prompt,
    format_query_gen_prompt,
)
from hmafact.claims.time_sensitive import detect_time_sensitivity
from hmafact.llm.openrouter import generate_response
from hmafact.schemas.claims import (
    Claim,
    ClaimType,
    ExtractClaimsRequest,
    ExtractClaimsResponse,
    GenerateQueriesRequest,
    GenerateQueriesResponse,
)

logger = logging.getLogger(__name__)

# Try loading spaCy for entity extraction, fallback to regex if unavailable
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = None
except ImportError:
    _NLP = None


def extract_entities(text: str) -> list[str]:
    """Extract named entities from text using spaCy (or regex fallback)."""
    if _NLP is not None:
        doc = _NLP(text)
        entities = [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 1]
        if entities:
            return list(dict.fromkeys(entities))  # preserve order & deduplicate

    # Regex fallback for capitalized words / proper nouns
    caps = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b", text)
    filtered = [c.strip() for c in caps if c.lower() not in {"the", "a", "an", "this", "that", "there"}]
    return list(dict.fromkeys(filtered))


def _clean_json_array(raw_text: str) -> list[dict[str, Any]]:
    """Parse JSON array output from LLM with fallback handling."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict) and "claims" in data and isinstance(data["claims"], list):
            return [item for item in data["claims"] if isinstance(item, dict)]
    except Exception:
        pass

    # Regex fallback to find JSON objects in text
    matches = re.findall(r"\{[^{}]*\"text\"[^{}]*\}", raw_text)
    results = []
    for m in matches:
        try:
            parsed = json.loads(m)
            if isinstance(parsed, dict) and "text" in parsed:
                results.append(parsed)
        except Exception:
            pass
    return results


def extract_claims(request: ExtractClaimsRequest) -> ExtractClaimsResponse:
    """
    Extract atomic, decontextualized claims from a generated response.

    In 'claim' mode (FEVER), the question itself is returned as a single Claim.
    In 'qa' mode, LLM decomposes long_answer into atomic claims.
    """
    q_is_ts, q_scope = detect_time_sensitivity(request.question)

    # FEVER Mode: single claim pass-through
    if request.mode == "claim":
        entities = extract_entities(request.question)
        c_is_ts, c_scope = detect_time_sensitivity(request.question)
        single_claim = Claim(
            claim_id=f"{request.query_id}:c0",
            text=request.question.strip(),
            source_span=request.question.strip(),
            entities=entities,
            claim_type="entity" if entities else "other",
            time_sensitive=q_is_ts or c_is_ts,
            temporal_scope=c_scope if c_scope != "unspecified" else q_scope,
        )
        return ExtractClaimsResponse(
            query_id=request.query_id,
            claims=[single_claim],
            n_claims=1,
        )

    # QA Mode: Call Agent LLM to extract claims
    user_prompt = format_extraction_prompt(request.question, request.long_answer)
    llm_res = generate_response(
        prompt=user_prompt,
        system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=500,
    )

    parsed_items = _clean_json_array(llm_res["text"])
    claims: list[Claim] = []

    for idx, item in enumerate(parsed_items):
        claim_text = str(item.get("text", "")).strip()
        if not claim_text or len(claim_text) < 5:
            continue

        raw_type = str(item.get("claim_type", "other")).lower()
        valid_types = {"entity", "temporal", "numeric", "relational", "other"}
        ctype: ClaimType = raw_type if raw_type in valid_types else "other"  # type: ignore[assignment]

        entities = extract_entities(claim_text)
        c_is_ts, c_scope = detect_time_sensitivity(claim_text)
        is_ts = q_is_ts or c_is_ts
        t_scope = c_scope if c_scope != "unspecified" else q_scope

        claim = Claim(
            claim_id=f"{request.query_id}:c{idx}",
            text=claim_text,
            source_span=None,
            entities=entities,
            claim_type=ctype,
            time_sensitive=is_ts,
            temporal_scope=t_scope,
        )
        claims.append(claim)

    # Fallback if LLM extraction returned 0 items
    if not claims:
        fallback_text = request.long_answer.splitlines()[0] if request.long_answer else request.question
        c_is_ts, c_scope = detect_time_sensitivity(fallback_text)
        claims.append(
            Claim(
                claim_id=f"{request.query_id}:c0",
                text=fallback_text.strip(),
                entities=extract_entities(fallback_text),
                claim_type="other",
                time_sensitive=q_is_ts or c_is_ts,
                temporal_scope=c_scope if c_scope != "unspecified" else q_scope,
            )
        )

    return ExtractClaimsResponse(
        query_id=request.query_id,
        claims=claims,
        n_claims=len(claims),
    )


def generate_claim_queries(
    request: GenerateQueriesRequest,
    question: str | None = None,
    as_of_year: int | str | None = None,
    live_enabled: bool = False,
    max_queries_ts: int = 3,
) -> GenerateQueriesResponse:
    """
    Generate search queries per claim for retrieval.

    When live_enabled=False, output is 100% benchmark-invariant (same byte output).
    When live_enabled=True and claim is time-sensitive:
      Generates entity-neutral, question-anchored queries.
    """
    result: dict[str, list[str]] = {}

    for claim in request.claims:
        q1 = claim.text
        # Entity-focused query
        if claim.entities:
            q2 = f"{' '.join(claim.entities)} {claim.text.split()[-1] if claim.text.split() else ''}".strip()
        else:
            q2 = claim.text.replace("?", "").strip()

        queries = [q1, q2]

        # WP2: Question-anchored entity-neutral queries for time-sensitive claims
        if live_enabled and claim.time_sensitive:
            neutral_queries: list[str] = []

            # Derive entity-neutral query from user's original question if provided
            if question and question.strip():
                q_clean = question.replace("?", "").strip()
                # Strip specific entity names mentioned in claim to make question entity-neutral
                for ent in claim.entities:
                    q_clean = re.sub(re.escape(ent), "", q_clean, flags=re.IGNORECASE).strip()
                q_clean = re.sub(r"\s+", " ", q_clean)
                if len(q_clean) > 5:
                    neutral_queries.append(q_clean)

            # Fallback neutral query: strip claim entities from claim text
            neutral_claim_text = claim.text
            for ent in claim.entities:
                neutral_claim_text = re.sub(re.escape(ent), "", neutral_claim_text, flags=re.IGNORECASE).strip()
            neutral_claim_text = re.sub(r"\s+", " ", neutral_claim_text)
            if len(neutral_claim_text) > 5:
                neutral_queries.append(neutral_claim_text)

            # Add "as of <year>" query variant
            if as_of_year:
                neutral_queries.append(f"{neutral_queries[0] if neutral_queries else question} as of {as_of_year}")

            # Combine neutral queries first, capped at max_queries_ts
            queries = (neutral_queries + queries)[:max_queries_ts]

        # Deduplicate while preserving order
        dedup_queries = list(dict.fromkeys(q for q in queries if q and q.strip()))
        result[claim.claim_id] = dedup_queries

    return GenerateQueriesResponse(queries=result)
