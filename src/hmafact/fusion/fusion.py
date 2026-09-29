"""
Service S6: Evidence Fusion Engine.
Aggregates evidence paths from EvidenceGraph, calculates support/contradiction mass,
and detects conflict across passages with WP4 recency-weighting support.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
import networkx as nx

from hmafact.graph.builder import EvidenceGraph
from hmafact.schemas.graph import GraphSnapshot
from hmafact.schemas.fusion import ClaimEvidenceFusion, ConsensusType


def fuse_graph_evidence(
    graph: EvidenceGraph | nx.MultiDiGraph | GraphSnapshot | dict[str, Any],
    support_threshold: float = 0.4,
    contradict_threshold: float = 0.4,
    as_of_date: date | str | None = None,
    live_enabled: bool = False,
    half_life_days: float = 365.0,
) -> list[ClaimEvidenceFusion]:
    """
    Traverses graph evidence links for each CLAIM node and aggregates support/contradiction scores.

    Args:
        graph: EvidenceGraph object, NetworkX MultiDiGraph, GraphSnapshot, or node_link dict.
        support_threshold: Threshold to consider support evidence significant.
        contradict_threshold: Threshold to consider contradiction evidence significant.
        as_of_date: Anchor date for recency calculation.
        live_enabled: Whether recency weighting is applied.
        half_life_days: Half-life in days for recency decay (default 365).

    Returns:
        List of ClaimEvidenceFusion metrics for each claim in the graph.
    """
    nx_graph: nx.MultiDiGraph
    if isinstance(graph, EvidenceGraph):
        nx_graph = graph.graph
    elif isinstance(graph, nx.MultiDiGraph):
        nx_graph = graph
    elif isinstance(graph, GraphSnapshot):
        nx_graph = nx.node_link_graph(graph.node_link)
    elif isinstance(graph, dict):
        if "node_link" in graph:
            nx_graph = nx.node_link_graph(graph["node_link"])
        else:
            nx_graph = nx.node_link_graph(graph)
    else:
        raise ValueError(f"Unsupported graph type: {type(graph)}")

    # Resolve as_of_date object
    as_of_dt: date
    if isinstance(as_of_date, date):
        as_of_dt = as_of_date
    elif isinstance(as_of_date, str) and as_of_date.strip():
        try:
            as_of_dt = date.fromisoformat(as_of_date)
        except Exception:
            as_of_dt = datetime.now(timezone.utc).date()
    else:
        as_of_dt = datetime.now(timezone.utc).date()

    fusions: list[ClaimEvidenceFusion] = []

    # Find all CLAIM nodes (support both 'ntype' and 'type')
    claim_nodes = [
        node for node, data in nx_graph.nodes(data=True)
        if (data.get("ntype") or data.get("type")) == "CLAIM"
    ]

    for claim_node in claim_nodes:
        claim_data = nx_graph.nodes[claim_node]
        claim_text = claim_data.get("text", claim_data.get("label", ""))
        is_time_sensitive = bool(claim_data.get("time_sensitive", False))

        support_passages: list[str] = []
        contradict_passages: list[str] = []
        top_support_score: float = 0.0
        top_contradict_score: float = 0.0
        top_support_citation: str | None = None
        top_contradict_citation: str | None = None

        total_support_score: float = 0.0
        total_contradict_score: float = 0.0
        has_undated_evidence = False

        # Helper to extract NLI scores and weight from edge_data
        def _extract_scores(edge_data: dict[str, Any], target_node_data: dict[str, Any]) -> tuple[float, float, float, float]:
            nli_dict = edge_data.get("nli_scores") or edge_data.get("nli") or {}
            if isinstance(nli_dict, dict):
                entail = float(nli_dict.get("entail", 0.0))
                contradict = float(nli_dict.get("contradict", 0.0))
            else:
                entail = getattr(nli_dict, "entail", 0.0)
                contradict = getattr(nli_dict, "contradict", 0.0)

            weight = float(
                edge_data.get("weight")
                or edge_data.get("score")
                or edge_data.get("relevance")
                or 0.0
            )

            # Recency weighting calculation (WP4)
            recency_weight = 1.0
            if live_enabled and is_time_sensitive:
                raw_doc_date = target_node_data.get("doc_date")
                if raw_doc_date:
                    try:
                        if isinstance(raw_doc_date, str):
                            d_dt = datetime.fromisoformat(raw_doc_date.replace("Z", "+00:00")).date()
                        elif isinstance(raw_doc_date, datetime):
                            d_dt = raw_doc_date.date()
                        elif isinstance(raw_doc_date, date):
                            d_dt = raw_doc_date
                        else:
                            d_dt = as_of_dt

                        age_days = max(0, (as_of_dt - d_dt).days)
                        recency_weight = 0.5 ** (age_days / max(1.0, half_life_days))
                    except Exception:
                        recency_weight = 0.1
                else:
                    nonlocal has_undated_evidence
                    has_undated_evidence = True
                    recency_weight = 0.1

            return entail, contradict, weight, recency_weight

        # Examine outgoing and incoming edges involving this claim
        # Outgoing edges from claim_node
        for _, target, key, edge_data in nx_graph.out_edges(claim_node, keys=True, data=True):
            edge_type = edge_data.get("etype") or edge_data.get("type")
            target_data = nx_graph.nodes.get(target, {})
            target_type = target_data.get("ntype") or target_data.get("type")

            if target_type == "PASSAGE" or "passage" in target.lower():
                passage_id = target
                passage_title = target_data.get("title") or target_data.get("passage_id") or target
                entail, contradict, weight, r_weight = _extract_scores(edge_data, target_data)

                if edge_type == "SUPPORTS" or (entail > contradict and entail > 0):
                    raw_score = entail if entail > 0 else (weight if weight > 0 else 0.5)
                    score = raw_score * r_weight
                    support_passages.append(passage_id)
                    total_support_score += score
                    if score > top_support_score:
                        top_support_score = score
                        top_support_citation = passage_title

                elif edge_type == "CONTRADICTS" or (contradict > entail and contradict > 0):
                    raw_score = contradict if contradict > 0 else (weight if weight > 0 else 0.5)
                    score = raw_score * r_weight
                    contradict_passages.append(passage_id)
                    total_contradict_score += score
                    if score > top_contradict_score:
                        top_contradict_score = score
                        top_contradict_citation = passage_title

        # Check incoming edges to claim_node (e.g., PASSAGE -> CLAIM)
        for source, _, key, edge_data in nx_graph.in_edges(claim_node, keys=True, data=True):
            edge_type = edge_data.get("etype") or edge_data.get("type")
            source_data = nx_graph.nodes.get(source, {})
            source_type = source_data.get("ntype") or source_data.get("type")

            if source_type == "PASSAGE" or "passage" in source.lower():
                passage_id = source
                passage_title = source_data.get("title") or source_data.get("passage_id") or source
                entail, contradict, weight, r_weight = _extract_scores(edge_data, source_data)

                if edge_type == "SUPPORTS" or (entail > contradict and entail > 0):
                    raw_score = entail if entail > 0 else (weight if weight > 0 else 0.5)
                    score = raw_score * r_weight
                    if passage_id not in support_passages:
                        support_passages.append(passage_id)
                        total_support_score += score
                        if score > top_support_score:
                            top_support_score = score
                            top_support_citation = passage_title

                elif edge_type == "CONTRADICTS" or (contradict > entail and contradict > 0):
                    raw_score = contradict if contradict > 0 else (weight if weight > 0 else 0.5)
                    score = raw_score * r_weight
                    if passage_id not in contradict_passages:
                        contradict_passages.append(passage_id)
                        total_contradict_score += score
                        if score > top_contradict_score:
                            top_contradict_score = score
                            top_contradict_citation = passage_title

        # Conflict detection
        conflict_detected = (
            total_support_score >= support_threshold and
            total_contradict_score >= contradict_threshold
        )

        # Consensus calculation
        consensus: ConsensusType
        if conflict_detected:
            consensus = "CONFLICT"
        elif total_support_score >= 0.4 and total_support_score > total_contradict_score * 1.2:
            consensus = "SUPPORT"
        elif total_contradict_score >= 0.4 and total_contradict_score > total_support_score * 1.2:
            consensus = "CONTRADICT"
        elif total_support_score > 0 or total_contradict_score > 0:
            consensus = "NEUTRAL"
        else:
            consensus = "INSUFFICIENT"

        fusions.append(
            ClaimEvidenceFusion(
                claim_id=str(claim_node),
                claim_text=claim_text,
                support_passages=support_passages,
                contradict_passages=contradict_passages,
                total_support_score=round(total_support_score, 4),
                total_contradict_score=round(total_contradict_score, 4),
                conflict_detected=conflict_detected,
                top_support_citation=top_support_citation,
                top_contradict_citation=top_contradict_citation,
                consensus=consensus,
                meta={
                    "top_support_score": round(top_support_score, 4),
                    "top_contradict_score": round(top_contradict_score, 4),
                    "undated_evidence": has_undated_evidence,
                    "time_sensitive": is_time_sensitive,
                }
            )
        )

    return fusions
