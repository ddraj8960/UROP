"""
Service S5 Claim Extraction Implementation.
Decomposes LLM answers into atomic, decontextualized claims and extracts entities.
"""
from __future__ import annotations

import json
import logging
import re
import difflib
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
    LLMCallRecord,
)

class ServiceError(Exception):
    pass

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
    caps = re.findall(r"\b([A-Z][\w]+(?:\s+[A-Z][\w]+)*)\b", text)
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


class ClaimPostProcessor:
    @staticmethod
    def process(claims: list[Claim], max_claims: int = 8) -> list[Claim]:
        processed = []
        
        for claim in claims:
            # 1. Sanitize text
            clean_text = claim.text.replace("\n", " ").strip()
            clean_text = re.sub(r'[^\w\s\.,\'"\-?!]', '', clean_text)
            
            if len(clean_text) < 5:
                continue
                
            # 2. Filter out pronoun-led claims (He, She, It, They)
            first_word = clean_text.split()[0].lower()
            if first_word in {"he", "she", "it", "they"}:
                continue
                
            # 3. Merge exact duplicates
            is_dup = False
            for p in processed:
                sim = difflib.SequenceMatcher(None, clean_text.lower(), p.text.lower()).ratio()
                if sim > 0.92:
                    is_dup = True
                    break
            if is_dup: continue
            
            claim.text = clean_text
            processed.append(claim)
            
            # 4. Cap at max_claims
            if len(processed) >= max_claims:
                break
                
        return processed


def extract_claims(request: ExtractClaimsRequest, model: str | None = None) -> ExtractClaimsResponse:
    """
    Extract atomic, decontextualized claims from a generated response.
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
            llm_call=None
        )

    # QA Mode: Call Agent LLM to extract claims
    # Truncate extremely long answers to prevent runaway LLM usage
    long_answer = request.long_answer[:10000]

    user_prompt = format_extraction_prompt(request.question, long_answer)
    try:
        llm_res = generate_response(
            prompt=user_prompt,
            system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1000,
            model=model,
        )
    except Exception as e:
        raise ServiceError(f"Extraction LLM failed: {str(e)}") from e

    llm_call = LLMCallRecord(
        model=llm_res["model"],
        prompt_tokens=llm_res.get("prompt_tokens", 0),
        completion_tokens=llm_res.get("completion_tokens", 0),
        latency_sec=llm_res.get("latency_sec", 0.0)
    )

    parsed_items = _clean_json_array(llm_res["text"])
    raw_claims: list[Claim] = []

    # Temporary ID placeholder
    for idx, item in enumerate(parsed_items):
        claim_text = str(item.get("text", "")).strip()
        raw_type = str(item.get("claim_type", "other")).lower()
        valid_types = {"entity", "temporal", "numeric", "relational", "other"}
        ctype: ClaimType = raw_type if raw_type in valid_types else "other"  # type: ignore[assignment]

        entities = extract_entities(claim_text)
        c_is_ts, c_scope = detect_time_sensitivity(claim_text)
        is_ts = q_is_ts or c_is_ts
        t_scope = c_scope if c_scope != "unspecified" else q_scope

        claim = Claim(
            claim_id="temp",
            text=claim_text,
            source_span=None,
            entities=entities,
            claim_type=ctype,
            time_sensitive=is_ts,
            temporal_scope=t_scope,
        )
        raw_claims.append(claim)

    # Post-process (filter, merge, cap, sanitize)
    claims = ClaimPostProcessor.process(raw_claims)

    # Assign deterministic stable IDs based on final order
    for idx, claim in enumerate(claims):
        claim.claim_id = f"{request.query_id}:c{idx}"

    # Fallback if extraction returned 0 items
    if not claims:
        fallback_text = long_answer.splitlines()[0] if long_answer else request.question
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
        llm_call=llm_call
    )


def generate_claim_queries(
    request: GenerateQueriesRequest,
    question: str | None = None,
    as_of_year: int | str | None = None,
    live_enabled: bool = False,
    max_queries_ts: int = 3,
    model: str | None = None,
) -> GenerateQueriesResponse:
    """
    Generate exactly 2 distinct search queries per claim using an LLM batched call.
    """
    result: dict[str, list[str]] = {}
    if not request.claims:
        return GenerateQueriesResponse(queries=result, llm_call=None)

    claims_text = [c.text for c in request.claims]
    user_prompt = format_query_gen_prompt(claims_text)
    
    try:
        llm_res = generate_response(
            prompt=user_prompt,
            system_prompt=QUERY_GEN_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=800,
            model=model,
        )
    except Exception as e:
        raise ServiceError(f"Query Gen LLM failed: {str(e)}") from e

    llm_call = LLMCallRecord(
        model=llm_res["model"],
        prompt_tokens=llm_res.get("prompt_tokens", 0),
        completion_tokens=llm_res.get("completion_tokens", 0),
        latency_sec=llm_res.get("latency_sec", 0.0)
    )

    # Parse response
    cleaned = llm_res["text"].strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        gen_data = json.loads(cleaned)
    except Exception:
        gen_data = {}

    for idx, claim in enumerate(request.claims):
        queries = []
        if isinstance(gen_data, dict) and str(idx) in gen_data:
            q_list = gen_data[str(idx)]
            if isinstance(q_list, list):
                queries = [str(q).strip() for q in q_list if q and str(q).strip()]
        
        # Deduplicate exactly 2 queries
        dedup_queries = list(dict.fromkeys(queries))
        
        # Fallbacks to guarantee exactly 2 distinct queries
        if not dedup_queries:
            q1 = claim.text
            q2 = f"{' '.join(claim.entities)} {claim.text.split()[-1] if claim.text.split() else ''}".strip()
            dedup_queries = [q1, q2]
            
        while len(dedup_queries) < 2:
            # Just append a generic alternative to ensure distinctiveness
            alt = f"{dedup_queries[0]} evidence"
            if alt not in dedup_queries:
                dedup_queries.append(alt)
            else:
                dedup_queries.append(f"{dedup_queries[0]} fact check")
                
        # Truncate to exactly 2 distinct queries
        dedup_queries = dedup_queries[:2]
        
        # WP2: Time-sensitive handling
        if live_enabled and claim.time_sensitive:
            neutral_queries: list[str] = []
            if question and question.strip():
                q_clean = question.replace("?", "").strip()
                for ent in claim.entities:
                    q_clean = re.sub(re.escape(ent), "", q_clean, flags=re.IGNORECASE).strip()
                q_clean = re.sub(r"\s+", " ", q_clean)
                if len(q_clean) > 5:
                    neutral_queries.append(q_clean)

            if as_of_year:
                neutral_queries.append(f"{neutral_queries[0] if neutral_queries else question} as of {as_of_year}")
                
            dedup_queries = (neutral_queries + dedup_queries)[:max_queries_ts]

        # Sanitize queries to prevent Wikipedia API srsearch injection
        sanitized = []
        for q in dedup_queries:
            s = re.sub(r'[^\w\s\.,\'"\-?!]', '', q)
            sanitized.append(s.strip()[:300]) # Cap query length
            
        result[claim.claim_id] = sanitized

    return GenerateQueriesResponse(queries=result, llm_call=llm_call)
