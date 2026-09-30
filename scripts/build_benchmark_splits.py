#!/usr/bin/env python3
"""
build_benchmark_splits.py — CLI orchestrator for M1.

Usage:
    uv run python scripts/build_benchmark_splits.py
    uv run python scripts/build_benchmark_splits.py --seed 42 --output-dir data/samples
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml  # type: ignore[import]
except ImportError:
    yaml = None  # type: ignore[assignment]

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmafact.data.prepare_benchmarks import (
    HOTPOTQA_DEV_URL,
    fetch_fever,
    fetch_hotpotqa,
    fetch_truthfulqa,
)
from hmafact.data.splits import AllocationError, allocate, stratified_split
from hmafact.schemas.input import BenchmarkSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("configs/default.yaml")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path = _DEFAULT_CONFIG) -> dict:
    """Load default.yaml; return empty dict if unavailable or yaml not installed."""
    if yaml is None:
        logger.warning("PyYAML not installed — using built-in defaults, not default.yaml.")
        return {}
    if not config_path.exists():
        logger.warning("Config not found at %s — using built-in defaults.", config_path)
        return {}
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_excluded_ids(path: Path) -> frozenset[str]:
    """
    Read the bake-off leakage exclusion list.
    Lines starting with '#' and blank lines are ignored.
    Returns a frozenset of sample_id strings.
    """
    if not path.exists():
        logger.warning("excluded_ids.txt not found at %s — no IDs will be excluded.", path)
        return frozenset()
    ids: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    logger.info("Loaded %d excluded IDs from %s.", len(ids), path)
    return frozenset(ids)


def apply_exclusions(
    samples: list, excluded_ids: frozenset[str], dataset: str
) -> tuple[list, int]:
    """
    Remove samples whose sample_id is in excluded_ids.
    Returns (filtered_samples, n_removed).
    """
    before = len(samples)
    filtered = [s for s in samples if s.sample_id not in excluded_ids]
    n_removed = before - len(filtered)
    if n_removed:
        logger.warning(
            "%s: removed %d sample(s) via excluded_ids.txt.", dataset, n_removed
        )
    return filtered, n_removed


# ---------------------------------------------------------------------------
# FEVER-specific fixed quotas (100 per class test, 34/33/33 dev)
# ---------------------------------------------------------------------------
FEVER_TEST_QUOTAS = {"SUPPORTS": 100, "REFUTES": 100, "NOT ENOUGH INFO": 100}
FEVER_DEV_QUOTAS = {"SUPPORTS": 34, "REFUTES": 33, "NOT ENOUGH INFO": 33}


def assign_split_field(samples: list[BenchmarkSample],
                        split: str) -> list[BenchmarkSample]:
    """Return new BenchmarkSample objects with the `split` field set."""
    return [s.model_copy(update={"split": split}) for s in samples]


def mark_ablation_subset(test_samples: list[BenchmarkSample],
                          n_ablation: int,
                          seed: int,
                          key: str,
                          ) -> list[BenchmarkSample]:
    """
    Mark `n_ablation` test samples as ablation_subset=True using stratified selection.
    Returns the full test list with ablation flags set.
    `key` can be a direct BenchmarkSample attribute (e.g. 'gold_label')
    or a key inside meta (e.g. 'category', 'type', 'qtype').
    """
    def _get_val(s: BenchmarkSample) -> str:
        if hasattr(s, key) and not callable(getattr(s, key)):
            v = getattr(s, key)
            if v is not None:
                return str(v)
        # Fall back to meta dict — supports 'category', 'type', 'qtype'
        return str(s.meta.get(key, s.meta.get("type", "unknown")))

    strata_sizes = Counter(_get_val(s) for s in test_samples)
    try:
        quotas = allocate(dict(strata_sizes), n_ablation)
    except AllocationError:
        quotas = {k: min(v, n_ablation // max(len(strata_sizes), 1)) for k, v in strata_sizes.items()}

    rng = random.Random(seed + 999)

    from collections import defaultdict
    groups: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for s in test_samples:
        groups[_get_val(s)].append(s)

    ablation_ids: set[str] = set()
    for stratum, quota in quotas.items():
        pool = sorted(groups[stratum], key=lambda x: x.sample_id)
        rng.shuffle(pool)
        for s in pool[:quota]:
            ablation_ids.add(s.sample_id)

    return [s.model_copy(update={"ablation_subset": s.sample_id in ablation_ids})
            for s in test_samples]


def write_jsonl(samples: list[BenchmarkSample], path: Path) -> tuple[str, dict[str, int]]:
    """
    Write samples as JSONL (sorted keys, UTF-8, \\n endings, trailing newline).
    Returns (sha256_of_written_file, stratum_counts_dict).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [s.model_dump_json() + "\n" for s in samples]

    # For FEVER test: interleave labels round-robin (S, R, N, ...)
    if "fever" in path.name and "test" in path.name:
        groups: dict[str, list[str]] = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}
        for line in lines:
            import json as _json
            lbl = _json.loads(line)["gold_label"]
            groups[lbl].append(line)
        interleaved: list[str] = []
        order = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        max_len = max((len(v) for v in groups.values()), default=0)
        for idx in range(max_len):
            for k in order:
                if idx < len(groups[k]):
                    interleaved.append(groups[k][idx])
        lines = interleaved

    content = "".join(lines)
    raw_bytes = content.encode("utf-8")
    path.write_bytes(raw_bytes)
    sha = hashlib.sha256(raw_bytes).hexdigest()
    logger.info("Wrote %d rows → %s  (sha256: %s)", len(samples), path, sha[:16] + "...")

    # Compute per-stratum counts for manifest reporting
    import json as _json
    stratum_counts: Counter[str] = Counter()
    for line in lines:
        obj = _json.loads(line)
        # Try dataset-specific stratum keys in priority order
        for key in ("gold_label", "type", "category"):
            val = obj.get(key) or obj.get("meta", {}).get(key)
            if val:
                stratum_counts[str(val)] += 1
                break

    return sha, dict(stratum_counts)


def build_manifest(
    output_dir: Path,
    file_meta: dict[str, dict],
    dataset_meta: dict[str, dict],
    seed: int,
    n_dev: int,
    n_test: int,
    n_ablation: int,
    excluded_ids_applied: int = 0,
) -> None:
    """Write manifest.json to output_dir."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    import platform, pydantic  # noqa: E401
    manifest = {
        "seed": seed,
        "n_dev": n_dev,
        "n_test": n_test,
        "n_ablation": n_ablation,
        "excluded_ids_applied": excluded_ids_applied,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "pydantic_version": pydantic.__version__,
        "files": file_meta,
        "datasets": dataset_meta,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("Manifest written to %s/manifest.json", output_dir)


def build_readme(output_dir: Path, seed: int, dataset_meta: dict) -> None:
    lines = [
        "# data/samples — Benchmark Split Provenance\n",
        f"**Random seed:** {seed}  ",
        "**Dev size:** 100 per dataset (300 total)  ",
        "**Test size:** 300 per dataset (900 total)  ",
        "**Ablation subset:** 150 per dataset, flagged with `ablation_subset=true`  \n",
        "## Datasets\n",
    ]
    for ds, meta in dataset_meta.items():
        lines += [
            f"### {ds}",
            f"- **Source:** {meta.get('source_url', meta.get('hf_id', 'N/A'))}",
            f"- **Resolved revision:** `{meta.get('revision', 'N/A')}`",
            f"- **Raw rows:** {meta.get('raw_rows', 'N/A')}",
            f"- **After filtering:** {meta.get('filtered_rows', 'N/A')}",
            "",
        ]
    lines += [
        "## Caveats",
        "- TruthfulQA has ~38 categories; very small categories may have 0 dev items.",
        "- HotpotQA comparison questions are ~20% of the sample (~60 test items); "
          "comparison-specific breakdowns will be noisy.",
        "- FEVER NLI model (DeBERTa) was trained on FEVER *train*; we evaluate on *dev*. "
          "Disclosed per master plan §2.",
        "- HotpotQA yes/no answers (comparison type) are included in `gold_answer`.",
        "",
    ]
    readme_path = output_dir.parent / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("README written to %s", readme_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build M1 benchmark splits.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: read from configs/default.yaml, then 42).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fever-cache", type=Path, default=None,
                        help="Path to cache the raw FEVER download.")
    parser.add_argument("--excluded-ids", type=Path, default=None,
                        help="Path to bake-off exclusion list (default: data/excluded_ids.txt).")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args()

    # ---- Load config then overlay CLI args ----
    cfg = load_config(args.config)
    data_cfg: dict = cfg.get("data", {})

    seed: int = args.seed if args.seed is not None else data_cfg.get("seed", 42)
    n_dev: int = data_cfg.get("n_dev", 100)
    n_test: int = data_cfg.get("n_test", 300)
    n_ablation: int = data_cfg.get("n_ablation", 150)
    output_dir: Path = args.output_dir or Path(data_cfg.get("samples_dir", "data/samples"))
    fever_cache: Path = args.fever_cache or Path("data/.cache/fever_dev_raw.jsonl")
    excluded_ids_path: Path = args.excluded_ids or Path("data/excluded_ids.txt")

    logger.info(
        "Config: seed=%d n_dev=%d n_test=%d n_ablation=%d output_dir=%s",
        seed, n_dev, n_test, n_ablation, output_dir,
    )

    # ---- Load exclusion list ----
    excluded_ids = load_excluded_ids(excluded_ids_path)
    total_excluded_applied = 0

    file_meta: dict[str, dict] = {}
    dataset_meta: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # TruthfulQA                                                           #
    # ------------------------------------------------------------------ #
    # Pass the pinned revision from config if present (null = latest).
    tqa_cfg = data_cfg.get("datasets", {}).get("truthfulqa", {})
    tqa_revision_pin: str | None = tqa_cfg.get("revision") or None
    tqa_samples, tqa_revision = fetch_truthfulqa(revision=tqa_revision_pin)
    dataset_meta["truthfulqa"] = {
        "hf_id": "truthfulqa/truthful_qa/generation/validation",
        "revision": str(tqa_revision),
        "raw_rows": len(tqa_samples),
        "filtered_rows": len(tqa_samples),
    }

    tqa_samples, n_excl = apply_exclusions(tqa_samples, excluded_ids, "truthfulqa")
    total_excluded_applied += n_excl
    dataset_meta["truthfulqa"]["filtered_rows"] = len(tqa_samples)

    # Stratify TruthfulQA by meta["category"] using a lightweight wrapper
    class _TQAItem:
        def __init__(self, s: BenchmarkSample):
            self._s = s
            self.sample_id = s.sample_id
            self.category = s.meta.get("category", "unknown")

    tqa_wrapped = [_TQAItem(s) for s in tqa_samples]
    tqa_dev_w, tqa_test_w = stratified_split(
        tqa_wrapped, key="category", n_dev=n_dev, n_test=n_test, seed=seed
    )
    tqa_dev_samples = assign_split_field([w._s for w in tqa_dev_w], "dev")
    tqa_test_samples = assign_split_field([w._s for w in tqa_test_w], "test")
    tqa_test_samples = mark_ablation_subset(tqa_test_samples, n_ablation, seed, "category")

    sha_dev, strata_dev = write_jsonl(tqa_dev_samples, output_dir / "truthfulqa_dev.jsonl")
    sha_test, strata_test = write_jsonl(tqa_test_samples, output_dir / "truthfulqa_test.jsonl")
    file_meta["truthfulqa_dev.jsonl"] = {"rows": len(tqa_dev_samples), "sha256": sha_dev,
                                          "excluded_ids_applied": n_excl, "strata": strata_dev}
    file_meta["truthfulqa_test.jsonl"] = {"rows": len(tqa_test_samples), "sha256": sha_test,
                                           "excluded_ids_applied": 0, "strata": strata_test}

    # ------------------------------------------------------------------ #
    # FEVER (Option B — direct download)                                   #
    # ------------------------------------------------------------------ #
    fever_samples, fever_revision = fetch_fever(cache_path=fever_cache)

    # Filter: SUPPORTS/REFUTES must have at least one evidence pair
    fever_filtered = [
        s for s in fever_samples
        if s.gold_label == "NOT ENOUGH INFO" or len(s.gold_evidence) > 0
    ]
    logger.info("FEVER: %d → %d after filtering claims without evidence.",
                len(fever_samples), len(fever_filtered))

    fever_filtered, n_excl = apply_exclusions(fever_filtered, excluded_ids, "fever")
    total_excluded_applied += n_excl
    dataset_meta["fever"] = {
        "source_url": "https://fever.ai/download/fever/shared_task_dev.jsonl",
        "revision": fever_revision,
        "raw_rows": len(fever_samples),
        "filtered_rows": len(fever_filtered),
    }

    fever_dev_samples_raw, fever_test_samples_raw = stratified_split(
        fever_filtered,
        key="gold_label",
        n_dev=n_dev,
        n_test=n_test,
        seed=seed,
        dev_quotas=FEVER_DEV_QUOTAS,
        test_quotas=FEVER_TEST_QUOTAS,
    )
    fever_dev_samples = assign_split_field(fever_dev_samples_raw, "dev")
    fever_test_samples = assign_split_field(fever_test_samples_raw, "test")
    fever_test_samples = mark_ablation_subset(fever_test_samples, n_ablation, seed, "gold_label")

    sha_dev, strata_dev = write_jsonl(fever_dev_samples, output_dir / "fever_dev.jsonl")
    sha_test, strata_test = write_jsonl(fever_test_samples, output_dir / "fever_test.jsonl")
    file_meta["fever_dev.jsonl"] = {"rows": len(fever_dev_samples), "sha256": sha_dev,
                                     "excluded_ids_applied": n_excl, "strata": strata_dev}
    file_meta["fever_test.jsonl"] = {"rows": len(fever_test_samples), "sha256": sha_test,
                                      "excluded_ids_applied": 0, "strata": strata_test}

    # ------------------------------------------------------------------ #
    # HotpotQA                                                             #
    # ------------------------------------------------------------------ #
    hotpot_samples, hotpot_revision = fetch_hotpotqa(
        cache_path=Path("data/.cache/hotpot_qa_dev.parquet")
    )

    hotpot_samples, n_excl = apply_exclusions(hotpot_samples, excluded_ids, "hotpotqa")
    total_excluded_applied += n_excl
    dataset_meta["hotpotqa"] = {
        "source_url": HOTPOTQA_DEV_URL,
        "revision": hotpot_revision,
        "raw_rows": len(hotpot_samples) + n_excl,
        "filtered_rows": len(hotpot_samples),
    }

    class _HotpotItem:
        def __init__(self, s: BenchmarkSample):
            self._s = s
            self.sample_id = s.sample_id
            self.qtype = s.meta.get("type", "unknown")

    hotpot_wrapped = [_HotpotItem(s) for s in hotpot_samples]
    hotpot_dev_w, hotpot_test_w = stratified_split(
        hotpot_wrapped, key="qtype", n_dev=n_dev, n_test=n_test, seed=seed
    )
    hotpot_dev_samples = assign_split_field([w._s for w in hotpot_dev_w], "dev")
    hotpot_test_samples = assign_split_field([w._s for w in hotpot_test_w], "test")
    hotpot_test_samples = mark_ablation_subset(hotpot_test_samples, n_ablation, seed, "qtype")

    sha_dev, strata_dev = write_jsonl(hotpot_dev_samples, output_dir / "hotpotqa_dev.jsonl")
    sha_test, strata_test = write_jsonl(hotpot_test_samples, output_dir / "hotpotqa_test.jsonl")
    file_meta["hotpotqa_dev.jsonl"] = {"rows": len(hotpot_dev_samples), "sha256": sha_dev,
                                        "excluded_ids_applied": n_excl, "strata": strata_dev}
    file_meta["hotpotqa_test.jsonl"] = {"rows": len(hotpot_test_samples), "sha256": sha_test,
                                         "excluded_ids_applied": 0, "strata": strata_test}

    # ------------------------------------------------------------------ #
    # Manifest + README                                                    #
    # ------------------------------------------------------------------ #
    build_manifest(
        output_dir, file_meta, dataset_meta, seed,
        n_dev=n_dev, n_test=n_test, n_ablation=n_ablation,
        excluded_ids_applied=total_excluded_applied,
    )
    build_readme(output_dir, seed, dataset_meta)

    logger.info(
        "M1 complete. All 6 JSONL files written to %s/  (excluded_ids_applied=%d)",
        output_dir, total_excluded_applied,
    )


if __name__ == "__main__":
    main()
