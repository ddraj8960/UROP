"""
Hybrid Retriever and Reciprocal Rank Fusion (RRF) for S3 Retrieval Service.

Combines lexical (BM25) and semantic search results using RRF scoring:
    RRF_Score(doc) = sum_{m in Methods} 1 / (60 + rank_m(doc))
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Sequence

from hmafact.schemas.evidence import EvidencePassage, SourceName
from hmafact.retrieval.wiki_api import search_wikipedia_api

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[list[EvidencePassage]],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 5,
    min_live_passages: int = 0,
) -> list[EvidencePassage]:
    """
    Combine multiple ranked lists of EvidencePassages into a single fused ranking via RRF.

    Args:
        ranked_lists: List of ranked EvidencePassage lists (e.g., [bm25_results, vector_results]).
        rrf_k: Smoothing constant for RRF formula (default 60).
        top_k: Maximum number of passages to return.
        min_live_passages: Reserved slots for live sources (wiki_api/wikidata) in top_k.

    Returns:
        Unified list of EvidencePassage objects with updated RRF scores and ranks.
    """
    scores: dict[str, float] = {}
    passage_map: dict[str, EvidencePassage] = {}

    for ranked_list in ranked_lists:
        for rank, passage in enumerate(ranked_list, 1):
            pid = passage.passage_id
            if pid not in passage_map:
                passage_map[pid] = passage
            # Add reciprocal rank score
            rrf_val = 1.0 / (rrf_k + rank)
            scores[pid] = scores.get(pid, 0.0) + rrf_val

    if not scores:
        return []

    # Sort passage IDs by descending RRF score
    sorted_pids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)

    # Reserve min_live_passages slots for live sources if requested
    if min_live_passages > 0:
        live_pids = [pid for pid in sorted_pids if passage_map[pid].source in ("wiki_api", "wikidata")]
        non_live_pids = [pid for pid in sorted_pids if passage_map[pid].source not in ("wiki_api", "wikidata")]

        selected_live = live_pids[:min_live_passages]
        remaining_slots = max(0, top_k - len(selected_live))
        selected_other = [pid for pid in sorted_pids if pid not in selected_live][:remaining_slots]

        final_pids = sorted(selected_live + selected_other, key=lambda pid: scores[pid], reverse=True)[:top_k]
    else:
        final_pids = sorted_pids[:top_k]

    # Normalize max score to 1.0
    max_score = scores[final_pids[0]] if final_pids else 1.0

    fused: list[EvidencePassage] = []
    for rank, pid in enumerate(final_pids, 1):
        original = passage_map[pid]
        norm_score = round(scores[pid] / max_score, 4) if max_score > 0 else 0.0

        updated_passage = original.model_copy(
            update={
                "retrieval_score": norm_score,
                "rank": rank,
            }
        )
        fused.append(updated_passage)

    return fused


def search_evidence(
    query: str,
    sources: list[SourceName] | None = None,
    k: int = 5,
    live_enabled: bool = False,
    min_live_passages: int = 3,
    ttl_hours: float = 24.0,
    refresh: bool = False,
    local_snapshot_date: str = "2018-06-01",
) -> list[EvidencePassage]:
    """
    Retrieve top-k evidence passages for a query across configured sources.

    Args:
        query: Query string.
        sources: List of sources to query (defaults to ["wiki_api"]).
        k: Number of passages to return.
        live_enabled: Whether live retrieval features (slot reservation, TTL) are enabled.
        min_live_passages: Minimum live passages reserved in top-k when live_enabled.
        ttl_hours: API response cache TTL in hours.
        refresh: Force cache bypass if True.
        local_snapshot_date: ISO date string for local_wiki snapshot.

    Returns:
        List of EvidencePassage objects with retrieved_at and doc_date populated.
    """
    if not sources:
        sources = ["wiki_api"]

    if live_enabled and "wiki_api" not in sources:
        sources.append("wiki_api")

    all_rankings: list[list[EvidencePassage]] = []
    now_utc = datetime.now(timezone.utc)

    if "wiki_api" in sources:
        wiki_passages = search_wikipedia_api(
            query=query,
            k=k * 2,
            ttl_hours=ttl_hours if live_enabled else None,
            refresh=refresh,
        )
        if wiki_passages:
            all_rankings.append(wiki_passages)

    if not all_rankings:
        return []

    # Apply timestamps to local_wiki passages if present
    try:
        snapshot_dt = datetime.fromisoformat(f"{local_snapshot_date}T00:00:00+00:00")
    except Exception:
        snapshot_dt = datetime(2018, 6, 1, tzinfo=timezone.utc)

    for r_list in all_rankings:
        for idx, p in enumerate(r_list):
            updates = {}
            if p.retrieved_at is None:
                updates["retrieved_at"] = now_utc
            if p.doc_date is None:
                updates["doc_date"] = snapshot_dt if p.source == "local_wiki" else now_utc
            if updates:
                r_list[idx] = p.model_copy(update=updates)

    # If single source, return top k
    if len(all_rankings) == 1:
        return all_rankings[0][:k]

    # Multiple sources: fuse via RRF
    return reciprocal_rank_fusion(
        all_rankings,
        top_k=k,
        min_live_passages=min_live_passages if live_enabled else 0,
    )


class HybridRetriever:
    """Retriever class wrapper."""

    def __init__(
        self,
        sources: list[SourceName] | None = None,
        live_enabled: bool = False,
        min_live_passages: int = 3,
        ttl_hours: float = 24.0,
    ):
        self.sources = sources or ["wiki_api"]
        self.live_enabled = live_enabled
        self.min_live_passages = min_live_passages
        self.ttl_hours = ttl_hours

    def search(self, query: str, k: int = 5, refresh: bool = False) -> list[EvidencePassage]:
        return search_evidence(
            query=query,
            sources=self.sources,
            k=k,
            live_enabled=self.live_enabled,
            min_live_passages=self.min_live_passages,
            ttl_hours=self.ttl_hours,
            refresh=refresh,
        )
