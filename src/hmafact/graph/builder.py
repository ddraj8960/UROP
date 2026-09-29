"""
Service S4 Dynamic Evidence Retrieval Graph Builder.
Constructs a NetworkX MultiDiGraph connecting Queries, Claims, Search Queries, Passages, and Entities.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
from hmafact.claims.extractor import extract_entities, generate_claim_queries
from hmafact.graph.nli import score_nli
from hmafact.retrieval.hybrid import search_evidence
from hmafact.schemas.claims import Claim, GenerateQueriesRequest
from hmafact.schemas.evidence import EvidencePassage
from hmafact.schemas.graph import BuildGraphRequest, BuildGraphResponse, GraphSnapshot

logger = logging.getLogger(__name__)


class EvidenceGraph:
    """
    Wrapper around NetworkX MultiDiGraph representing per-query dynamic evidence retrieval.
    """

    def __init__(self, query_id: str, question: str):
        self.query_id = query_id
        self.question = question
        self.graph = nx.MultiDiGraph()
        self.passages: dict[str, EvidencePassage] = {}

        # Add root QUERY node
        self.graph.add_node(
            query_id,
            ntype="QUERY",
            text=question,
        )

    def add_claim(self, claim: Claim) -> None:
        """Add a CLAIM node and link to root QUERY via HAS_CLAIM edge."""
        self.graph.add_node(
            claim.claim_id,
            ntype="CLAIM",
            text=claim.text,
            claim_type=claim.claim_type,
            entities=claim.entities,
        )
        self.graph.add_edge(self.query_id, claim.claim_id, etype="HAS_CLAIM")

        # Add ENTITY nodes and MENTIONS edges
        for ent in claim.entities:
            ent_node_id = f"entity:{ent.lower().replace(' ', '_')}"
            if not self.graph.has_node(ent_node_id):
                self.graph.add_node(ent_node_id, ntype="ENTITY", name=ent)
            self.graph.add_edge(claim.claim_id, ent_node_id, etype="MENTIONS")

    def add_search_queries(self, claim_id: str, queries: list[str]) -> None:
        """Add SEARCH_QUERY nodes and link to CLAIM via GENERATED_QUERY edge."""
        for idx, q_text in enumerate(queries):
            q_node_id = f"{claim_id}:sq{idx}"
            self.graph.add_node(q_node_id, ntype="SEARCH_QUERY", text=q_text)
            self.graph.add_edge(claim_id, q_node_id, etype="GENERATED_QUERY")

    def add_passage_evidence(
        self,
        claim_id: str,
        sq_id: str,
        passage: EvidencePassage,
        tau_e: float = 0.60,
        tau_c: float = 0.60,
    ) -> None:
        """
        Add PASSAGE, DOCUMENT, SOURCE nodes and link via RETRIEVED, PART_OF, FROM_SOURCE,
        and score NLI to add SUPPORTS, CONTRADICTS, or RELATED_TO edges to CLAIM.
        """
        pid = passage.passage_id
        self.passages[pid] = passage

        # Add PASSAGE node if not present
        if not self.graph.has_node(pid):
            self.graph.add_node(
                pid,
                ntype="PASSAGE",
                title=passage.title,
                text=passage.text,
                url=passage.url,
                score=passage.retrieval_score,
            )

        # Link SEARCH_QUERY -> PASSAGE via RETRIEVED edge
        self.graph.add_edge(sq_id, pid, etype="RETRIEVED", score=passage.retrieval_score)

        # Score NLI between PASSAGE and CLAIM
        claim_text = self.graph.nodes[claim_id]["text"]
        nli_scores = score_nli(premise=passage.text, hypothesis=claim_text)

        # Determine edge type based on threshold
        if nli_scores.entail >= tau_e:
            edge_type = "SUPPORTS"
        elif nli_scores.contradict >= tau_c:
            edge_type = "CONTRADICTS"
        else:
            edge_type = "RELATED_TO"

        self.graph.add_edge(
            pid,
            claim_id,
            etype=edge_type,
            nli=nli_scores.model_dump(),
            relevance=passage.retrieval_score,
        )

        # Cross-claim entity link: add MENTIONS edges for passage entities
        p_entities = extract_entities(passage.text)
        for ent in p_entities:
            ent_node_id = f"entity:{ent.lower().replace(' ', '_')}"
            if not self.graph.has_node(ent_node_id):
                self.graph.add_node(ent_node_id, ntype="ENTITY", name=ent)
            self.graph.add_edge(pid, ent_node_id, etype="MENTIONS")

    def to_snapshot(self, iteration: int = 0) -> GraphSnapshot:
        """Serialize NetworkX graph to GraphSnapshot object."""
        node_link_dict = nx.node_link_data(self.graph)
        return GraphSnapshot(
            node_link=node_link_dict,
            iteration=iteration,
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
        )


def build_evidence_graph(request: BuildGraphRequest) -> BuildGraphResponse:
    """
    Build a dynamic evidence retrieval graph for a query and its claims.

    Args:
        request: BuildGraphRequest object.

    Returns:
        BuildGraphResponse object.
    """
    eg = EvidenceGraph(query_id=request.query_id, question=request.question)

    # Convert claim dicts to Claim objects if needed
    claims: list[Claim] = []
    for c_dict in request.claims:
        if isinstance(c_dict, Claim):
            claims.append(c_dict)
        else:
            claims.append(Claim.model_validate(c_dict))

    # Add claims & generate queries
    queries_resp = generate_claim_queries(GenerateQueriesRequest(claims=claims))

    for c in claims:
        eg.add_claim(c)
        sq_list = queries_resp.queries.get(c.claim_id, [c.text])
        eg.add_search_queries(c.claim_id, sq_list)

        # Retrieve passages for search queries
        for idx, sq_text in enumerate(sq_list):
            sq_id = f"{c.claim_id}:sq{idx}"
            passages = search_evidence(query=sq_text, k=request.k_per_query)
            for p in passages:
                eg.add_passage_evidence(claim_id=c.claim_id, sq_id=sq_id, passage=p)

    snapshot = eg.to_snapshot()

    return BuildGraphResponse(
        query_id=request.query_id,
        graph_snapshot=snapshot,
        passage_count=len(eg.passages),
        claim_count=len(claims),
    )
