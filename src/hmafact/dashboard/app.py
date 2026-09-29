"""
FastAPI Dashboard Application for HMA-Fact (Modules 0-8).
Serves interactive Web Dashboard API and UI static assets.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from hmafact.llm.openrouter import OpenRouterClient
from hmafact.schemas.claims import ExtractClaimsRequest
from hmafact.schemas.graph import BuildGraphRequest
from hmafact.schemas.generator import GenerateRequest
from hmafact.generator.generator import generate_answer
from hmafact.claims.extractor import extract_claims
from hmafact.graph.builder import build_evidence_graph
from hmafact.fusion.fusion import fuse_graph_evidence
from hmafact.verification.verifier import FactVerifier
from hmafact.logic.validator import validate_claims_logic
from hmafact.confidence.estimator import estimate_confidence
from hmafact.synthesis.synthesizer import ResponseSynthesizer
from hmafact.baselines.rag import run_rag_baseline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HMA-Fact Interactive Multi-Agent Verification Dashboard",
    description="Service-Oriented Hierarchical Multi-Agent System with Dynamic Evidence Retrieval Graphs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SAMPLES_DIR = Path("data/samples")


class VerifyRequestPayload(BaseModel):
    query_id: str = Field(default="live_query", description="Unique query identifier")
    question: str = Field(..., description="User question or statement under test")
    mode: str = Field(default="qa", description="'qa' or 'claim'")
    use_llm: bool = Field(default=True, description="Whether to invoke OpenRouter LLM")


@app.get("/")
def read_root():
    """Serve main dashboard SPA HTML page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"status": "HMA-Fact API running", "docs": "/docs"})


@app.get("/api/samples")
def get_benchmark_samples():
    """Retrieve sample questions from TruthfulQA, FEVER, and HotpotQA dev splits."""
    samples = []
    datasets = ["truthfulqa", "fever", "hotpotqa"]

    for ds in datasets:
        file_path = SAMPLES_DIR / f"{ds}_dev.jsonl"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[:10]  # Take first 10 per dataset
                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        samples.append({
                            "sample_id": data.get("sample_id"),
                            "dataset": data.get("dataset"),
                            "input_text": data.get("input_text"),
                            "gold_answer": data.get("gold_answer"),
                            "gold_label": data.get("gold_label"),
                        })
    return {"samples": samples}


@app.post("/api/verify")
def verify_query(payload: VerifyRequestPayload):
    """
    Runs end-to-end HMA-Fact verification pipeline (Modules 2 - 8).
    Returns synthesized response, claim verdicts, logic score, confidence, and visual graph data.
    """
    try:
        query_id = payload.query_id
        question = payload.question
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Step 1: Initial Answer Generation (S2)
        gen_resp = generate_answer(
            GenerateRequest(query_id=query_id, input_text=question, mode=payload.mode)
        )
        long_answer = gen_resp.long_answer

        # Step 2: Atomic Claim Extraction (S5)
        claims_resp = extract_claims(
            ExtractClaimsRequest(query_id=query_id, question=question, long_answer=long_answer, mode=payload.mode)
        )
        claims = claims_resp.claims

        # Step 3: Logic Validation (S8)
        logic_res = validate_claims_logic(query_id=query_id, claims=claims, live_enabled=True)

        # Step 4: Build Dynamic Evidence Graph (S4 + S4a)
        graph_resp = build_evidence_graph(
            BuildGraphRequest(
                query_id=query_id,
                question=question,
                claims=[c.model_dump() for c in claims],
                k_per_query=2,
            )
        )
        graph_snapshot = graph_resp.graph_snapshot

        # Step 5: Fact Verification & Evidence Fusion (S6 + S7)
        llm_client = None
        if payload.use_llm:
            try:
                llm_client = OpenRouterClient()
            except Exception:
                llm_client = None

        verifier = FactVerifier(llm_client=llm_client)
        ver_resp = verifier.verify_claims_graph(
            query_id=query_id,
            question=question,
            claims=claims,
            graph=graph_snapshot,
            use_llm=llm_client is not None,
            as_of_date=as_of,
            live_enabled=True,
        )

        # Step 6: Confidence Estimation (S9) & Response Synthesis (S10)
        synthesizer = ResponseSynthesizer(llm_client=llm_client)
        synth_resp = synthesizer.synthesize(
            query_id=query_id,
            question=question,
            original_answer=long_answer,
            claims=claims,
            verification_response=ver_resp,
            logic_result=logic_res,
            use_llm=llm_client is not None,
            as_of_date=as_of,
            live_enabled=True,
        )

        # Format vis-network graph data for UI rendering
        vis_graph = format_vis_graph(graph_snapshot.node_link)

        return {
            "query_id": query_id,
            "question": question,
            "original_answer": long_answer,
            "final_answer": synth_resp.final_answer,
            "overall_verdict": synth_resp.overall_verdict,
            "confidence": synth_resp.confidence.model_dump(),
            "logic_result": logic_res.model_dump(),
            "claims": [c.model_dump() for c in claims],
            "claim_results": [r.model_dump() for r in ver_resp.results],
            "corrections_made": synth_resp.corrections_made,
            "citations_used": synth_resp.citations_used,
            "vis_graph": vis_graph,
            "model_used": gen_resp.model,
            "as_of": as_of,
        }

    except Exception as e:
        logger.exception("Verification pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
def compare_baselines(payload: VerifyRequestPayload):
    """
    Runs Vanilla (M2) vs Standard RAG (M4) vs HMA-Fact Multi-Agent (M8) side-by-side.
    """
    try:
        query_id = payload.query_id
        question = payload.question

        # 1. Vanilla LLM baseline
        vanilla = generate_answer(GenerateRequest(query_id=query_id, input_text=question))

        # 2. Standard RAG baseline
        rag = run_rag_baseline(query_id=query_id, question=question, top_k=3)

        # 3. HMA-Fact Multi-Agent framework
        hma_res = verify_query(payload)

        return {
            "query_id": query_id,
            "question": question,
            "vanilla_answer": vanilla.long_answer,
            "rag_answer": rag["long_answer"],
            "rag_passages": rag["passages_used"],
            "hmafact_answer": hma_res["final_answer"],
            "hmafact_verdict": hma_res["overall_verdict"],
            "hmafact_confidence": hma_res["confidence"]["overall_confidence"],
        }
    except Exception as e:
        logger.exception("Comparison error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def format_vis_graph(node_link_dict: dict[str, Any]) -> dict[str, Any]:
    """Converts NetworkX node_link dict to vis-network nodes and edges format."""
    nodes = []
    edges = []

    for node in node_link_dict.get("nodes", []):
        node_id = str(node.get("id"))
        ntype = node.get("ntype", "UNKNOWN")
        text = node.get("text") or node.get("name") or node.get("title") or node_id

        # Color mapping per node type
        color_map = {
            "QUERY": {"background": "#f59e0b", "border": "#d97706"},        # Amber
            "CLAIM": {"background": "#3b82f6", "border": "#1d4ed8"},        # Blue
            "SEARCH_QUERY": {"background": "#8b5cf6", "border": "#6d28d9"}, # Violet
            "PASSAGE": {"background": "#10b981", "border": "#047857"},      # Emerald
            "ENTITY": {"background": "#ec4899", "border": "#be185d"},       # Pink
        }

        c = color_map.get(ntype, {"background": "#6b7280", "border": "#374151"})

        nodes.append({
            "id": node_id,
            "label": f"[{ntype[:4]}] {text[:35]}..." if len(text) > 35 else f"[{ntype[:4]}] {text}",
            "title": f"<b>{ntype}</b><br>{text}",
            "color": c,
            "shape": "box" if ntype in ("CLAIM", "PASSAGE") else "ellipse",
            "font": {"color": "#ffffff", "face": "Inter, sans-serif", "size": 13},
        })

    for link in node_link_dict.get("links", []):
        source = str(link.get("source"))
        target = str(link.get("target"))
        etype = link.get("etype") or link.get("type") or "RELATED"

        # Color mapping per edge type
        edge_color = "#9ca3af"  # Default gray
        if etype == "SUPPORTS":
            edge_color = "#10b981"  # Emerald green
        elif etype == "CONTRADICTS":
            edge_color = "#ef4444"  # Red
        elif etype == "HAS_CLAIM":
            edge_color = "#60a5fa"  # Light blue

        edges.append({
            "from": source,
            "to": target,
            "label": etype,
            "color": {"color": edge_color, "highlight": "#38bdf8"},
            "arrows": "to",
            "font": {"color": "#9ca3af", "size": 10, "align": "horizontal"},
        })

    return {"nodes": nodes, "edges": edges}
