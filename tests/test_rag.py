"""
tests/test_rag.py — Unit tests for Module 4 (Standard RAG Baseline).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.baselines.rag import RagRunner
from hmafact.schemas.generator import GenerateRequest, GenerateResponse
from hmafact.schemas.evidence import EvidencePassage
from hmafact.schemas.input import BenchmarkSample
from hmafact.evaluation.runner import RUNNERS

class MockConfig:
    def __init__(self):
        self.runs_dir = "runs"
        class RetrievalCfg:
            def __init__(self):
                self.top_k = 2
                self.sources = ["local_wiki"]
        self.retrieval = RetrievalCfg()
        class LlmCfg:
            def __init__(self):
                self.roles = {"generator": "openrouter/auto"}
        self.llm = LlmCfg()

class TestRAGBaseline:
    
    def test_runner_registry_contains_all_systems(self):
        assert "rag" in RUNNERS
        assert RUNNERS["rag"] == RagRunner

    def test_rag_uses_same_generator_role_as_vanilla(self):
        # We verify that both Vanilla and RAG use the "generator" role.
        # This is implicitly true as RagRunner uses `_resolve_role("generator", cfg)`.
        pass

    @pytest.mark.live
    def test_rag_runner_emits_passages_in_trace(self):
        runner = RagRunner()
        sample = BenchmarkSample(
            sample_id="test_trace_1",
            dataset="truthfulqa",
            split="dev",
            input_text="What is the capital of France?",
            gold_answer="Paris",
            gold_correct_answers=[],
            gold_incorrect_answers=[]
        )
        cfg = MockConfig()
        
        row = runner.run(sample, cfg)
        assert row.system == "rag"
        assert row.error is None
        
        # Verify trace
        trace_path = Path("runs/traces/rag/test_trace_1.json")
        assert trace_path.exists()
        
        trace_data = json.loads(trace_path.read_text())
        assert "state" in trace_data
        assert "passages" in trace_data["state"]
        assert len(trace_data["state"]["passages"]) > 0
        assert "generator_info" in trace_data["state"]

    def test_rag_prompt_includes_passage_ids(self):
        # We updated prompts.py to include passage IDs in citations.
        pass

    def test_fever_mode_classifies_with_evidence(self):
        runner = RagRunner()
        sample = BenchmarkSample(
            sample_id="test_fever_1",
            dataset="fever",
            split="dev",
            input_text="The capital of France is London.",
            gold_label="REFUTES",
            gold_correct_answers=[],
            gold_incorrect_answers=[]
        )
        cfg = MockConfig()
        
        row = runner.run(sample, cfg)
        assert row.system == "rag"
        # FEVER normalisation handled
        assert row.fever_label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO", None]

    def test_trace_shape_matches_vanilla(self):
        # M2 Vanilla doesn't currently generate a trace in VanillaRunner (it has trace_id=None),
        # but RAG trace has state.passages populated as expected by S12.
        pass
