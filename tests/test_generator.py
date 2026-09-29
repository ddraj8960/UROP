"""
tests/test_generator.py — Unit tests for Module 2 (S2 Generator).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.generator.generator import _parse_llm_json, generate_answer
from hmafact.schemas.generator import GenerateRequest, GenerateResponse


class TestGeneratorSchemas:
    def test_generate_request_validation(self):
        req = GenerateRequest(query_id="q1", input_text="What is 2+2?", mode="qa")
        assert req.query_id == "q1"
        assert req.mode == "qa"

    def test_generate_response_validation(self):
        res = GenerateResponse(
            query_id="q1",
            input_text="What is 2+2?",
            short_answer="4",
            long_answer="2 plus 2 equals 4.",
            raw_text='{"short_answer": "4", "long_answer": "2 plus 2 equals 4."}',
            model="meta-llama/llama-3.1-8b-instruct",
            latency_sec=0.5,
        )
        assert res.short_answer == "4"
        assert res.query_id == "q1"


class TestJSONParsing:
    def test_valid_json_parsing(self):
        raw = '{"short_answer": "Paris", "long_answer": "Paris is the capital of France."}'
        short, long = _parse_llm_json(raw)
        assert short == "Paris"
        assert long == "Paris is the capital of France."

    def test_markdown_codeblock_parsing(self):
        raw = '```json\n{"short_answer": "Paris", "long_answer": "Capital of France."}\n```'
        short, long = _parse_llm_json(raw)
        assert short == "Paris"
        assert long == "Capital of France."

    def test_fallback_plain_text(self):
        raw = "The capital of France is Paris."
        short, long = _parse_llm_json(raw)
        assert short == "The capital of France is Paris."
        assert long == "The capital of France is Paris."


class TestGeneratorExecution:
    @pytest.mark.live
    def test_live_generator_call(self):
        req = GenerateRequest(query_id="test:1", input_text="What is the capital of France?", mode="qa")
        res = generate_answer(req)
        assert isinstance(res, GenerateResponse)
        assert len(res.short_answer) > 0
        assert len(res.long_answer) > 0
        assert res.latency_sec > 0
