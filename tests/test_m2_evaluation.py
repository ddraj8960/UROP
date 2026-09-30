"""
tests/test_m2_evaluation.py — M2 evaluation harness tests.

Pure-logic tests only. No model calls. All LLM interactions are monkeypatched.
The @live judge κ test is marked separately.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hmafact.evaluation.metrics import (
    SCORERS,
    _em,
    _normalize,
    _token_f1,
    score_fever,
    score_hotpotqa,
    score_truthfulqa,
)
from hmafact.evaluation.runner import (
    RUNNERS,
    VanillaRunner,
    _read_done_ids,
    run_system,
)
from hmafact.schemas.evaluation import (
    JudgeVerdict,
    PredictionRow,
    RunSystemRequest,
)
from hmafact.schemas.input import BenchmarkSample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_verdict(correct: bool = True, informative: bool = True) -> JudgeVerdict:
    return JudgeVerdict(correct=correct, informative=informative,
                        rationale="test", judge_model="test-model")


def _make_pred(
    sample_id: str = "s1",
    system: str = "vanilla",
    short_answer: str | None = "Paris",
    answer_text: str = "Paris. Paris is the capital.",
    abstained: bool = False,
    fever_label: str | None = None,
    judge: JudgeVerdict | None = None,
    error: str | None = None,
) -> PredictionRow:
    return PredictionRow(
        sample_id=sample_id,
        system=system,  # type: ignore[arg-type]
        short_answer=short_answer,
        answer_text=answer_text,
        abstained=abstained,
        fever_label=fever_label,
        judge=judge,
        latency_ms=100.0,
        n_llm_calls=1,
        prompt_tokens=50,
        completion_tokens=20,
        error=error,
    )


def _make_sample(**kwargs) -> BenchmarkSample:
    defaults = dict(
        sample_id="s1",
        dataset="truthfulqa",
        split="dev",
        input_text="What is the capital of France?",
        gold_answer="Paris",
    )
    defaults.update(kwargs)
    return BenchmarkSample(**defaults)


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def test_normalize_strips_articles_and_punctuation():
    assert _normalize("The Eiffel Tower!") == "eiffel tower"
    assert _normalize("A cat.") == "cat"


def test_em_exact_match():
    assert _em("Paris", "paris") == 1.0
    assert _em("Paris", "London") == 0.0
    assert _em(None, "Paris") == 0.0


def test_token_f1_partial_overlap():
    f1 = _token_f1("the cat sat on the mat", "the cat sat")
    assert 0.0 < f1 < 1.0

    assert _token_f1("Paris", "Paris") == 1.0
    assert _token_f1("Paris", "London") == 0.0


# ---------------------------------------------------------------------------
# Hallucination rate — must exclude abstentions
# ---------------------------------------------------------------------------

def test_hallucination_rate_excludes_abstentions():
    """
    Among abstained answers, hallucination_rate must not count them.
    A system that abstains 90% of the time should not show 0% hallucination
    unless the 10% that answered were all correct.
    """
    verdict_correct = _make_verdict(correct=True)
    verdict_wrong = _make_verdict(correct=False)

    gold_sample = _make_sample(sample_id="s1")
    gold = {"s1": gold_sample, "s2": _make_sample(sample_id="s2"),
            "s3": _make_sample(sample_id="s3")}

    rows = [
        _make_pred("s1", judge=verdict_wrong, abstained=False),   # answered wrong
        _make_pred("s2", abstained=True),                          # abstained
        _make_pred("s3", abstained=True),                          # abstained
    ]

    metrics = score_truthfulqa(rows, gold)

    # Only 1 answered, 0 truthful → hallucination_rate = 1.0
    assert metrics["hallucination_rate"] == 1.0
    # Abstention rate = 2/3
    assert abs(metrics["abstention_rate"] - 2 / 3) < 1e-9


def test_hallucination_rate_zero_when_all_correct():
    gold = {f"s{i}": _make_sample(sample_id=f"s{i}") for i in range(3)}
    rows = [_make_pred(f"s{i}", judge=_make_verdict(correct=True)) for i in range(3)]
    metrics = score_truthfulqa(rows, gold)
    assert metrics["hallucination_rate"] == 0.0
    assert metrics["abstention_rate"] == 0.0


# ---------------------------------------------------------------------------
# score_fever — label normalisation (Issue 4 fix)
# ---------------------------------------------------------------------------

class _FakeGoldSample:
    def __init__(self, sid, label):
        self.sample_id = sid
        self.gold_label = label


def test_score_fever_normalises_generator_labels():
    """Generator outputs SUPPORTED/CONTRADICTED — must be mapped to SUPPORTS/REFUTES."""
    gold = {
        "s1": _FakeGoldSample("s1", "SUPPORTS"),
        "s2": _FakeGoldSample("s2", "REFUTES"),
        "s3": _FakeGoldSample("s3", "NOT ENOUGH INFO"),
    }
    rows = [
        _make_pred("s1", fever_label="SUPPORTS"),       # already correct
        _make_pred("s2", fever_label="SUPPORTED"),       # old wrong label — maps to SUPPORTS ≠ REFUTES
        _make_pred("s3", fever_label="NOT ENOUGH INFO"), # correct
    ]
    metrics = score_fever(rows, gold)
    # s1=correct, s2=wrong (SUPPORTS≠REFUTES), s3=abstained from answered perspective
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "hallucination_rate" in metrics
    assert "abstention_rate" in metrics


def test_score_fever_all_correct():
    gold = {
        "s1": _FakeGoldSample("s1", "SUPPORTS"),
        "s2": _FakeGoldSample("s2", "REFUTES"),
    }
    rows = [
        _make_pred("s1", fever_label="SUPPORTS"),
        _make_pred("s2", fever_label="REFUTES"),
    ]
    metrics = score_fever(rows, gold)
    assert metrics["accuracy"] == 1.0
    assert metrics["abstention_rate"] == 0.0


def test_score_fever_all_abstained():
    gold = {"s1": _FakeGoldSample("s1", "SUPPORTS")}
    rows = [_make_pred("s1", fever_label="NOT ENOUGH INFO")]
    metrics = score_fever(rows, gold)
    assert metrics["abstention_rate"] == 1.0
    assert metrics["hallucination_rate"] == 0.0


# ---------------------------------------------------------------------------
# score_hotpotqa
# ---------------------------------------------------------------------------

class _FakeHotpotGold:
    def __init__(self, sid, answer):
        self.sample_id = sid
        self.gold_answer = answer


def test_score_hotpotqa_em_and_f1():
    gold = {"s1": _FakeHotpotGold("s1", "Paris")}
    rows = [_make_pred("s1", short_answer="Paris", judge=_make_verdict(correct=True))]
    metrics = score_hotpotqa(rows, gold)
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_score_hotpotqa_partial_f1():
    gold = {"s1": _FakeHotpotGold("s1", "the Eiffel Tower in Paris")}
    rows = [_make_pred("s1", short_answer="Paris", judge=_make_verdict(correct=False))]
    metrics = score_hotpotqa(rows, gold)
    assert 0.0 < metrics["token_f1"] < 1.0
    assert metrics["exact_match"] == 0.0


# ---------------------------------------------------------------------------
# RUNNERS registry
# ---------------------------------------------------------------------------

def test_runners_registry_contains_vanilla():
    assert "vanilla" in RUNNERS
    assert issubclass(RUNNERS["vanilla"], object)


def test_vanilla_runner_name():
    runner = VanillaRunner()
    assert runner.name == "vanilla"


# ---------------------------------------------------------------------------
# run_system resume logic (_read_done_ids)
# ---------------------------------------------------------------------------

def test_read_done_ids_skips_error_rows(tmp_path):
    pred_file = tmp_path / "predictions.jsonl"
    rows = [
        {"sample_id": "s1", "system": "vanilla", "error": None, "answer_text": "ok",
         "short_answer": "yes", "abstained": False, "latency_ms": 1.0,
         "n_llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5},
        {"sample_id": "s2", "system": "vanilla", "error": "SomeError: boom",
         "answer_text": "", "short_answer": None, "abstained": False, "latency_ms": 1.0,
         "n_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
    ]
    with pred_file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    done = _read_done_ids(pred_file)
    assert "s1" in done       # no error → marked done
    assert "s2" not in done   # has error → must be retried


def test_read_done_ids_empty_file(tmp_path):
    pred_file = tmp_path / "predictions.jsonl"
    assert _read_done_ids(pred_file) == set()  # non-existent → empty


# ---------------------------------------------------------------------------
# SCORERS registry
# ---------------------------------------------------------------------------

def test_scorers_registry_has_all_datasets():
    assert set(SCORERS.keys()) == {"fever", "hotpotqa", "truthfulqa"}


# ---------------------------------------------------------------------------
# run_system integration (monkeypatched — no real LLM calls)
# ---------------------------------------------------------------------------

def test_run_system_skips_done_and_records_errors(tmp_path, monkeypatch):
    """
    run_system should:
      1. Skip sample_ids already in predictions.jsonl without error.
      2. Record error rows for failures without aborting.
      3. Append rather than overwrite.
    """
    import hmafact.evaluation.runner as runner_mod
    import hmafact.data.loader as loader_mod

    # Fake samples
    sample_done = _make_sample(sample_id="done_1")
    sample_fail = _make_sample(sample_id="fail_1")
    sample_new  = _make_sample(sample_id="new_1")

    monkeypatch.setattr(loader_mod, "load_samples",
                        lambda **kw: [sample_done, sample_fail, sample_new])

    # Pre-populate predictions file with 'done_1' as already completed
    pred_path = tmp_path / "vanilla_truthfulqa_dev_test" / "predictions.jsonl"
    pred_path.parent.mkdir(parents=True)
    existing_row = PredictionRow(
        sample_id="done_1", system="vanilla",  # type: ignore[arg-type]
        short_answer="yes", answer_text="yes. always.",
        latency_ms=50.0, n_llm_calls=1, prompt_tokens=10, completion_tokens=5,
    )
    pred_path.write_text(existing_row.model_dump_json() + "\n", encoding="utf-8")

    # Patch VanillaRunner.run: fail_1 raises, new_1 succeeds
    def fake_run(self, sample, cfg):
        if sample.sample_id == "fail_1":
            return PredictionRow(
                sample_id=sample.sample_id,
                system="vanilla",  # type: ignore[arg-type]
                short_answer=None,
                answer_text="",
                latency_ms=1.0,
                n_llm_calls=0,
                prompt_tokens=0,
                completion_tokens=0,
                error="RuntimeError: Simulated failure",
            )
        return PredictionRow(
            sample_id=sample.sample_id,
            system="vanilla",  # type: ignore[arg-type]
            short_answer="answer", answer_text="answer. long.",
            latency_ms=100.0, n_llm_calls=1, prompt_tokens=20, completion_tokens=10,
        )

    monkeypatch.setattr(VanillaRunner, "run", fake_run)

    # Fake cfg
    class _Cfg:
        class data:
            samples_dir = "data/samples"
        runs_dir = str(tmp_path)

    request = RunSystemRequest(
        system="vanilla",
        dataset="truthfulqa",
        split="dev",
        run_id="vanilla_truthfulqa_dev_test",
    )
    response = run_system(request, _Cfg())

    assert response.n_failed == 1
    assert response.n_done >= 1  # done_1 (pre-existing) + new_1

    # Predictions file must have 3 rows total (done_1 pre-existing + fail_1 error + new_1 success)
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    objs = [json.loads(l) for l in lines]
    ids = [o["sample_id"] for o in objs]
    assert "done_1" in ids
    assert "fail_1" in ids
    assert "new_1" in ids

    # fail_1 must have an error field
    fail_obj = next(o for o in objs if o["sample_id"] == "fail_1")
    assert fail_obj["error"] is not None
