"""
Hybrid Retriever and Reciprocal Rank Fusion (RRF) for S3 Retrieval Service.

Combines lexical (BM25) and semantic search results using RRF scoring:
    RRF_Score(doc) = sum_{m in Methods} 1 / (60 + rank_m(doc))
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Sequence, Any

from hmafact.schemas.evidence import EvidencePassage, SourceName, SearchRequest, SearchResponse
from hmafact.retrieval.wiki_api import WikiApiSource
from hmafact.retrieval.local import LocalHybridRetriever
from hmafact.retrieval.protocol import EvidenceSource

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60

# SOURCES Registry
SOURCES: dict[SourceName, type[EvidenceSource]] = {
    "local_wiki": LocalHybridRetriever,
    "wiki_api": WikiApiSource,
}


def fuse_rankings(
    dense: list[tuple[str, float]], 
    sparse: list[tuple[str, float]],
    k_rrf: int = DEFAULT_RRF_K
) -> list[tuple[str, float]]:
    """
    Reciprocal rank fusion: score(d) = sum(1 / (k_rrf + rank)).
    Deterministic; ties broken by passage_id so results never depend on dict ordering.
    """
    scores: dict[str, float] = {}
    
    for rank, (pid, _) in enumerate(dense, 1):
        scores[pid] = scores.get(pid, 0.0) + (1.0 / (k_rrf + rank))
        
    for rank, (pid, _) in enumerate(sparse, 1):
        scores[pid] = scores.get(pid, 0.0) + (1.0 / (k_rrf + rank))
        
    if not scores:
        return []
        
    # Sort deterministically: highest score first, then fallback to passage_id alphabetically
    sorted_pids = sorted(scores.keys(), key=lambda pid: (-scores[pid], pid))
    
    max_score = scores[sorted_pids[0]] if sorted_pids else 1.0
    return [(pid, scores[pid] / max_score if max_score > 0 else 0.0) for pid in sorted_pids]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[list[EvidencePassage]],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 5,
    min_live_passages: int = 0,
) -> list[EvidencePassage]:
    """
    Combine multiple ranked lists of EvidencePassages into a single fused ranking via RRF.
    """
    scores: dict[str, float] = {}
    passage_map: dict[str, EvidencePassage] = {}

    for ranked_list in ranked_lists:
        for rank, passage in enumerate(ranked_list, 1):
            pid = passage.passage_id
            if not pid:
                # Fallback id if empty to prevent collision
                pid = f"unknown_{passage.source}_{hash(passage.text)}"
            if pid not in passage_map:
                passage_map[pid] = passage
            # Add reciprocal rank score
            rrf_val = 1.0 / (rrf_k + rank)
            scores[pid] = scores.get(pid, 0.0) + rrf_val

    if not scores:
        return []

    # Sort deterministically: by score descending, then pid ascending
    sorted_pids = sorted(scores.keys(), key=lambda pid: (-scores[pid], pid))

    # Reserve min_live_passages slots for live sources if requested
    if min_live_passages > 0:
        live_pids = [pid for pid in sorted_pids if passage_map[pid].source in ("wiki_api", "wikidata")]
        non_live_pids = [pid for pid in sorted_pids if passage_map[pid].source not in ("wiki_api", "wikidata")]

        selected_live = live_pids[:min_live_passages]
        remaining_slots = max(0, top_k - len(selected_live))
        selected_other = non_live_pids[:remaining_slots]

        final_pids = sorted(selected_live + selected_other, key=lambda pid: (-scores[pid], pid))[:top_k]
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


def search(request: SearchRequest, cfg: Any = None) -> SearchResponse:
    """
    Batch retrieval across sources for many claims at once.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Instantiate retrievers
    retrievers = []
    for source_name in request.sources:
        if source_name in SOURCES:
            SourceClass = SOURCES[source_name]
            if source_name == "local_wiki":
                index_dir = getattr(cfg.retrieval, "index_dir", "data/indices") if cfg and hasattr(cfg, "retrieval") else "data/indices"
                try:
                    retrievers.append(SourceClass(index_dir=index_dir, cfg=cfg.retrieval if cfg else None))
                except Exception as e:
                    logger.warning(f"Failed to initialize local_wiki: {e}")
            else:
                retrievers.append(SourceClass(cfg=cfg.retrieval if cfg else None))
        else:
            logger.warning(f"Source {source_name} not found in SOURCES registry.")

    results: dict[str, EvidencePassage] = {}

    queries_map = {}
    if isinstance(request.queries, dict):
        queries_map = request.queries
    else:
        # If it's a list, treat the query string as the claim_id itself
        for q in request.queries:
            queries_map[q] = [q]

    for claim_id, query_list in queries_map.items():
        for query in query_list:
            all_rankings = []
            for retriever in retrievers:
                try:
                    passages = retriever.search(query, k=request.k * 2)
                    if passages:
                        all_rankings.append(passages)
                except Exception as e:
                    logger.error(f"Error searching {retriever.name} for query '{query}': {e}")
            
            if not all_rankings:
                continue
                
            fused = reciprocal_rank_fusion(all_rankings, top_k=request.k)
            for passage in fused:
                pid = passage.passage_id
                if pid not in results:
                    passage.meta["retrieved_for"] = [claim_id]
                    passage.search_query = query
                    results[pid] = passage
                else:
                    if claim_id not in results[pid].meta.get("retrieved_for", []):
                        if "retrieved_for" not in results[pid].meta:
                            results[pid].meta["retrieved_for"] = []
                        results[pid].meta["retrieved_for"].append(claim_id)
                        
    return SearchResponse(passages=results)


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
    """Legacy interface for single query, redirects to search() batch handler."""
    if not sources:
        sources = ["wiki_api"]

    if live_enabled and "wiki_api" not in sources:
        sources.append("wiki_api")

    req = SearchRequest(queries=[query], sources=sources, k=k)
    resp = search(req)
    
    # Extract results for this query and sort by score
    passages = list(resp.passages.values())
    passages.sort(key=lambda p: (-p.retrieval_score, p.passage_id))
    return passages[:k]
