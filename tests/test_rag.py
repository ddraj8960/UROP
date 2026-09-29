"""
tests/test_rag.py — Unit tests for Module 4 (Standard RAG Baseline).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.baselines.rag import run_rag_baseline
from hmafact.schemas.generator import GenerateRequest, GenerateResponse
from hmafact.schemas.evidence import EvidencePassage


class TestRAGBaseline:
    @pytest.mark.live
    def test_run_rag_baseline_execution(self):
        req = GenerateRequest(
            query_id="rag_test:1",
            input_text="What is the capital of France?",
            mode="qa",
            dataset="truthfulqa",
        )
        res, passages = run_rag_baseline(req, top_k=2)

        assert isinstance(res, GenerateResponse)
        assert isinstance(passages, list)
        assert len(passages) > 0
        assert isinstance(passages[0], EvidencePassage)
        assert res.meta.get("rag_enabled") is True
        assert "retrieved_passages" in res.meta
        assert len(res.short_answer) > 0
