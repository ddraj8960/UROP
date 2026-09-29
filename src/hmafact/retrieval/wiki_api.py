"""
Wikipedia API retriever (Service S3 component).
Fetches live and cached search snippets, page extracts, and revision timestamps from Wikipedia API.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from hmafact.schemas.evidence import EvidencePassage

logger = logging.getLogger(__name__)

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
CACHE_DIR = Path("data/.cache/wiki")


def _clean_html_snippet(snippet: str) -> str:
    """Remove HTML tags like <span class="searchmatch"> from MediaWiki search snippets."""
    clean = re.sub(r"<[^>]+>", "", snippet)
    return clean.replace("&quot;", '"').replace("&amp;", "&").strip()


def search_wikipedia_api(
    query: str,
    k: int = 5,
    cache_dir: Path = CACHE_DIR,
    ttl_hours: float | None = 24.0,
    refresh: bool = False,
) -> list[EvidencePassage]:
    """
    Search Wikipedia via MediaWiki API and return top-k EvidencePassages.

    Args:
        query: Search query text.
        k: Number of passages to return.
        cache_dir: Directory to cache raw Wikipedia API responses.
        ttl_hours: Cache time-to-live in hours (None = permanent cache).
        refresh: Force cache bypass if True.

    Returns:
        List of EvidencePassage objects with retrieved_at and doc_date populated.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    query_hash = hashlib.md5(f"wiki_search:{query}:{k}".encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{query_hash}.json"

    # Check cache TTL if refresh is False
    if not refresh and cache_file.exists():
        try:
            mtime = cache_file.stat().st_mtime
            age_hours = (time.time() - mtime) / 3600.0
            if ttl_hours is None or age_hours <= ttl_hours:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                return [EvidencePassage.model_validate(p) for p in cached_data]
        except Exception:
            pass  # Fall back to live fetch if cache corrupted or expired

    # MediaWiki search API request
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": k,
        "format": "json",
        "utf8": 1,
        "prop": "revisions",
        "rvprop": "timestamp",
    }

    headers = {"User-Agent": "HMA-Fact FactVerification/1.0 (https://github.com/hma-fact)"}
    now_utc = datetime.now(timezone.utc)

    try:
        resp = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        search_results = data.get("query", {}).get("search", [])

        passages: list[EvidencePassage] = []
        for rank, item in enumerate(search_results, 1):
            page_id = str(item["pageid"])
            title = item["title"]
            snippet = _clean_html_snippet(item.get("snippet", ""))
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            # Extract revision timestamp if present
            doc_date: datetime | None = None
            if "timestamp" in item:
                try:
                    ts_str = item["timestamp"].replace("Z", "+00:00")
                    doc_date = datetime.fromisoformat(ts_str)
                except Exception:
                    doc_date = now_utc

            # Calculate normalized score (rank-based heuristic)
            score = round(1.0 - (rank - 1) * (0.5 / max(k, 1)), 3)

            passage = EvidencePassage(
                passage_id=f"wiki_api:{page_id}:0",
                source="wiki_api",
                doc_id=page_id,
                title=title,
                text=snippet or f"{title}: Wikipedia article.",
                url=page_url,
                retrieval_score=score,
                rank=rank,
                search_query=query,
                retrieved_at=now_utc,
                doc_date=doc_date or now_utc,
                meta={"word_count": item.get("wordcount", 0)},
            )
            passages.append(passage)

        # Cache results
        cache_file.write_text(
            json.dumps([p.model_dump(mode="json") for p in passages], indent=2),
            encoding="utf-8"
        )
        return passages

    except Exception as e:
        logger.error("Wikipedia API search failed for query '%s': %s", query, e)
        return []
