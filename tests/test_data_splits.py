"""
tests/test_data_splits.py — Offline unit tests for M1.

Run with:  uv run pytest tests/test_data_splits.py
Live tests: uv run pytest tests/test_data_splits.py -m live
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

# ── Make src importable when running from the project root ──────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.data.labels import FEVER_TO_CANONICAL, fever_to_canonical
from hmafact.data.loader import TestSplitLockedError, load_samples
from hmafact.data.splits import AllocationError, allocate, stratified_split
from hmafact.schemas.input import BenchmarkSample

SAMPLES_DIR = Path("data/samples")

EXPECTED_COUNTS = {
    "truthfulqa_dev": 100, "truthfulqa_test": 300,
    "fever_dev": 100, "fever_test": 300,
    "hotpotqa_dev": 100, "hotpotqa_test": 300,
}

DATASETS = ["truthfulqa", "fever", "hotpotqa"]


# ===========================================================================
# 1. Allocator unit tests (pure — no I/O)
# ===========================================================================

class TestAllocate:
    def test_sum_equals_total(self):
        result = allocate({"a": 100, "b": 50, "c": 25}, 30)
        assert sum(result.values()) == 30

    def test_proportional(self):
        result = allocate({"bridge": 6000, "comparison": 1400}, 400)
        assert result["bridge"] + result["comparison"] == 400
        assert result["bridge"] > result["comparison"]

    def test_single_stratum(self):
        result = allocate({"only": 50}, 20)
        assert result == {"only": 20}

    def test_quota_exceeds_stratum_raises(self):
        # total=5 but the only stratum has 2 items → should raise
        with pytest.raises(AllocationError):
            allocate({"tiny": 2}, 5)

    def test_zero_total(self):
        result = allocate({"a": 10, "b": 20}, 0)
        assert sum(result.values()) == 0

    def test_empty_strata_raises(self):
        with pytest.raises(AllocationError):
            allocate({}, 10)


# ===========================================================================
# 2. Stratified split unit tests (pure — no I/O)
# ===========================================================================

class _FakeItem:
    def __init__(self, sample_id: str, label: str):
        self.sample_id = sample_id
        self.label = label


class TestStratifiedSplit:
    def _make_items(self, counts: dict[str, int]) -> list[_FakeItem]:
        items = []
        for label, n in counts.items():
            for i in range(n):
                items.append(_FakeItem(f"{label}:{i:04d}", label))
        return items

    def test_correct_counts(self):
        items = self._make_items({"A": 200, "B": 100, "C": 50})
        dev, test = stratified_split(items, key="label", n_dev=50, n_test=100, seed=42)
        assert len(dev) == 50
        assert len(test) == 100

    def test_disjoint(self):
        items = self._make_items({"A": 300, "B": 200})
        dev, test = stratified_split(items, key="label", n_dev=100, n_test=200, seed=42)
        dev_ids = {i.sample_id for i in dev}
        test_ids = {i.sample_id for i in test}
        assert dev_ids.isdisjoint(test_ids)

    def test_deterministic(self):
        items = self._make_items({"X": 500, "Y": 500})
        dev1, test1 = stratified_split(items, key="label", n_dev=100, n_test=200, seed=42)
        dev2, test2 = stratified_split(items, key="label", n_dev=100, n_test=200, seed=42)
        assert [i.sample_id for i in dev1] == [i.sample_id for i in dev2]
        assert [i.sample_id for i in test1] == [i.sample_id for i in test2]

    def test_different_seed_gives_different_result(self):
        items = self._make_items({"A": 500})
        _, test1 = stratified_split(items, key="label", n_dev=50, n_test=100, seed=42)
        _, test2 = stratified_split(items, key="label", n_dev=50, n_test=100, seed=99)
        assert {i.sample_id for i in test1} != {i.sample_id for i in test2}

    def test_explicit_quotas(self):
        items = self._make_items({"SUPPORTS": 500, "REFUTES": 500, "NOT ENOUGH INFO": 500})
        dev, test = stratified_split(
            items, key="label", n_dev=100, n_test=300, seed=42,
            dev_quotas={"SUPPORTS": 34, "REFUTES": 33, "NOT ENOUGH INFO": 33},
            test_quotas={"SUPPORTS": 100, "REFUTES": 100, "NOT ENOUGH INFO": 100},
        )
        dev_counts = Counter(i.label for i in dev)
        test_counts = Counter(i.label for i in test)
        assert dev_counts["SUPPORTS"] == 34
        assert dev_counts["REFUTES"] == 33
        assert test_counts["SUPPORTS"] == 100


# ===========================================================================
# 3. Schema round-trip test
# ===========================================================================

class TestSchemaRoundTrip:
    def test_roundtrip(self):
        s = BenchmarkSample(
            sample_id="fever:12345",
            dataset="fever",
            split="dev",
            input_text="The Earth is flat.",
            gold_label="REFUTES",
            gold_evidence=[("Earth", 0), ("Flat_Earth", 3)],
            meta={"evidence_sets": [[["Earth", 0]]]},
        )
        raw = s.model_dump_json()
        restored = BenchmarkSample.model_validate_json(raw)
        assert restored.sample_id == s.sample_id
        assert restored.gold_evidence == s.gold_evidence
        assert restored.gold_label == s.gold_label


# ===========================================================================
# 4. Integration tests on committed JSONL files
#    (skipped if files don't exist yet — won't break CI before M1 runs)
# ===========================================================================

def _load_jsonl(path: Path) -> list[BenchmarkSample]:
    return [BenchmarkSample.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="Splits not built yet")
class TestCommittedFiles:
    @pytest.mark.parametrize("filename,expected", EXPECTED_COUNTS.items())
    def test_row_counts(self, filename: str, expected: int):
        path = SAMPLES_DIR / f"{filename}.jsonl"
        assert path.exists(), f"Missing: {path}"
        samples = _load_jsonl(path)
        assert len(samples) == expected, f"{filename}: expected {expected}, got {len(samples)}"

    @pytest.mark.parametrize("ds", DATASETS)
    def test_schema_validation(self, ds: str):
        for split in ("dev", "test"):
            path = SAMPLES_DIR / f"{ds}_{split}.jsonl"
            samples = _load_jsonl(path)
            for s in samples:
                assert s.dataset == ds
                assert s.split == split

    @pytest.mark.parametrize("ds", DATASETS)
    def test_no_id_overlap(self, ds: str):
        dev = _load_jsonl(SAMPLES_DIR / f"{ds}_dev.jsonl")
        test = _load_jsonl(SAMPLES_DIR / f"{ds}_test.jsonl")
        dev_ids = {s.sample_id for s in dev}
        test_ids = {s.sample_id for s in test}
        assert dev_ids.isdisjoint(test_ids), f"{ds}: dev/test overlap detected"

    @pytest.mark.parametrize("ds", DATASETS)
    def test_unique_sample_ids_within_file(self, ds: str):
        for split in ("dev", "test"):
            samples = _load_jsonl(SAMPLES_DIR / f"{ds}_{split}.jsonl")
            ids = [s.sample_id for s in samples]
            assert len(ids) == len(set(ids)), f"{ds}/{split}: duplicate sample_ids"

    def test_fever_dev_class_balance(self):
        samples = _load_jsonl(SAMPLES_DIR / "fever_dev.jsonl")
        counts = Counter(s.gold_label for s in samples)
        assert counts["SUPPORTS"] == 34
        assert counts["REFUTES"] == 33
        assert counts["NOT ENOUGH INFO"] == 33

    def test_fever_test_class_balance(self):
        samples = _load_jsonl(SAMPLES_DIR / "fever_test.jsonl")
        counts = Counter(s.gold_label for s in samples)
        assert counts["SUPPORTS"] == 100
        assert counts["REFUTES"] == 100
        assert counts["NOT ENOUGH INFO"] == 100

    def test_fever_evidence_presence(self):
        for split in ("dev", "test"):
            samples = _load_jsonl(SAMPLES_DIR / f"fever_{split}.jsonl")
            for s in samples:
                if s.gold_label in ("SUPPORTS", "REFUTES"):
                    assert len(s.gold_evidence) >= 1, (
                        f"{s.sample_id}: SUPPORTS/REFUTES claim has no evidence"
                    )
                if s.gold_label == "NOT ENOUGH INFO":
                    assert s.gold_evidence == [], (
                        f"{s.sample_id}: NEI claim should have empty evidence"
                    )

    def test_fever_titles_underscore_form(self):
        """Wiki titles must be in raw FEVER form (underscores, no spaces)."""
        samples = _load_jsonl(SAMPLES_DIR / "fever_dev.jsonl")
        for s in samples:
            for title, _ in s.gold_evidence:
                assert " " not in title, f"{s.sample_id}: title has spaces: '{title}'"

    def test_hotpotqa_level_hard(self):
        for split in ("dev", "test"):
            samples = _load_jsonl(SAMPLES_DIR / f"hotpotqa_{split}.jsonl")
            for s in samples:
                assert s.meta.get("level") == "hard", f"{s.sample_id}: level != hard"

    def test_hotpotqa_context_has_10_paragraphs(self):
        for split in ("dev", "test"):
            samples = _load_jsonl(SAMPLES_DIR / f"hotpotqa_{split}.jsonl")
            for s in samples:
                assert 1 <= len(s.meta.get("context", [])) <= 10, (
                    f"{s.sample_id}: context has {len(s.meta.get('context', []))} paragraphs"
                )

    def test_hotpotqa_gold_evidence_in_context(self):
        samples = _load_jsonl(SAMPLES_DIR / "hotpotqa_dev.jsonl")
        for s in samples:
            context_titles = {p["title"] for p in s.meta.get("context", [])}
            for title, _ in s.gold_evidence:
                assert title in context_titles, (
                    f"{s.sample_id}: gold evidence title '{title}' not in context"
                )

    def test_truthfulqa_required_fields(self):
        for split in ("dev", "test"):
            samples = _load_jsonl(SAMPLES_DIR / f"truthfulqa_{split}.jsonl")
            for s in samples:
                assert s.gold_correct_answers, f"{s.sample_id}: empty gold_correct_answers"
                assert s.gold_incorrect_answers, f"{s.sample_id}: empty gold_incorrect_answers"
                assert s.gold_answer, f"{s.sample_id}: empty gold_answer"
                assert s.meta.get("source") is not None, f"{s.sample_id}: missing meta.source"

    def test_ablation_subset_counts(self):
        for ds in DATASETS:
            samples = _load_jsonl(SAMPLES_DIR / f"{ds}_test.jsonl")
            ablation = [s for s in samples if s.ablation_subset]
            assert len(ablation) == 150, f"{ds}: ablation subset has {len(ablation)} items, expected 150"

    def test_manifest_sha256(self):
        manifest_path = SAMPLES_DIR / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("manifest.json not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import hashlib
        for filename, meta in manifest.get("files", {}).items():
            path = SAMPLES_DIR / filename
            if path.exists():
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                assert actual == meta["sha256"], f"{filename} sha256 mismatch"


# ===========================================================================
# 5. Loader tests
# ===========================================================================

class TestLoader:
    def test_test_without_final_raises(self, tmp_path):
        with pytest.raises(TestSplitLockedError):
            load_samples("fever", "test", final=False, data_dir=tmp_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_samples("fever", "dev", data_dir=tmp_path)

    @pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="Splits not built yet")
    def test_dev_load_and_limit(self):
        samples = load_samples("fever", "dev", limit=10)
        assert len(samples) == 10
        assert all(isinstance(s, BenchmarkSample) for s in samples)


# ===========================================================================
# 6. Label map tests
# ===========================================================================

class TestLabels:
    def test_all_keys_present(self):
        assert set(FEVER_TO_CANONICAL.keys()) == {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}

    def test_fever_to_canonical(self):
        assert fever_to_canonical("SUPPORTS") == "SUPPORTED"
        assert fever_to_canonical("REFUTES") == "CONTRADICTED"
        assert fever_to_canonical("NOT ENOUGH INFO") == "INSUFFICIENT_EVIDENCE"

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError):
            fever_to_canonical("PARTIALLY_SUPPORTED")


# ===========================================================================
# 7. Live network tests (skipped by default)
# ===========================================================================

@pytest.mark.live
class TestLiveFetch:
    def test_truthfulqa_row_count(self):
        from hmafact.data.prepare_benchmarks import fetch_truthfulqa
        samples, _ = fetch_truthfulqa()
        assert len(samples) == 817, f"Expected 817, got {len(samples)}"

    def test_hotpotqa_row_count(self):
        from hmafact.data.prepare_benchmarks import fetch_hotpotqa
        samples, _ = fetch_hotpotqa()
        assert len(samples) == 7405, f"Expected 7405, got {len(samples)}"

    def test_fever_download(self, tmp_path):
        from hmafact.data.prepare_benchmarks import fetch_fever
        samples, sha = fetch_fever(cache_path=tmp_path / "fever_raw.jsonl")
        # FEVER labelled_dev has ~19k unique claims
        assert len(samples) > 10000, f"Suspiciously few FEVER claims: {len(samples)}"
        assert len(sha) == 64  # sha256 hex string
