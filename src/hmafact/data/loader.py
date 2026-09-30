"""
loader.py — Runtime loader with test-split lock.
Enforces §7 risk 13: test data may only be loaded when final=True,
and every such access is logged.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from hmafact.schemas.input import BenchmarkSample

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLES_DIR = Path("data/samples")
_ACCESS_LOG = Path("runs/test_access.log")


class TestSplitLockedError(RuntimeError):
    """Raised when test split is accessed without final=True."""
    __test__ = False


class IntegrityError(RuntimeError):
    """Raised when a split file's sha256 does not match the manifest."""
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _verify_manifest(path: Path, samples_dir: Path) -> None:
    """Raise IntegrityError if file sha256 doesn't match the manifest."""
    manifest_path = samples_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning("manifest.json not found — skipping sha256 verification.")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_key = path.name
    expected_sha = manifest.get("files", {}).get(file_key, {}).get("sha256")
    if not expected_sha:
        logger.warning("No sha256 entry for %s in manifest — skipping.", file_key)
        return
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise IntegrityError(
            f"INTEGRITY FAILURE: {file_key} sha256 mismatch!\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {actual_sha}\n"
            "The file may have been corrupted or modified after the split was built.\n"
            "Re-run `uv run python scripts/build_benchmark_splits.py` to regenerate."
        )
    logger.debug("Integrity OK: %s (sha256: %s...)", file_key, actual_sha[:16])


def load_samples(
    dataset: Literal["truthfulqa", "fever", "hotpotqa"],
    split: Literal["dev", "test"],
    *,
    final: bool = False,
    limit: int | None = None,
    data_dir: Path | None = None,
) -> list[BenchmarkSample]:
    """
    Load a frozen benchmark split from disk.

    Args:
        dataset:  One of "truthfulqa", "fever", "hotpotqa".
        split:    "dev" or "test".
        final:    Must be True to load the test split. Logs the access.
        limit:    If set, return only the first N rows (for quick smoke-tests).
        data_dir: Override the samples directory (default: data/samples/).

    Returns:
        List of BenchmarkSample objects, re-validated on load.

    Raises:
        TestSplitLockedError: if split="test" and final=False.
    """
    if split == "test" and not final:
        raise TestSplitLockedError(
            "Test split is locked. Pass final=True only in M14 final evaluation runs. "
            "Use split='dev' for all development, tuning, and calibration work."
        )

    samples_dir = data_dir or _DEFAULT_SAMPLES_DIR
    path = Path(samples_dir) / f"{dataset}_{split}.jsonl"

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run `uv run python scripts/build_benchmark_splits.py` to generate it."
        )

    _verify_manifest(path, Path(samples_dir))

    if split == "test" and final:
        _log_test_access(dataset)

    lines = path.read_text(encoding="utf-8").splitlines()
    samples: list[BenchmarkSample] = []
    for line in lines:
        if not line.strip():
            continue
        # Re-validate on load: rebuilds tuples from JSON lists
        sample = BenchmarkSample.model_validate_json(line)
        samples.append(sample)

    if limit is not None:
        samples = samples[:limit]

    return samples


def _log_test_access(dataset: str) -> None:
    """Append one line to runs/test_access.log for audit trail."""
    import subprocess  # noqa: PLC0415

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    _ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = f"{timestamp} | dataset={dataset} | git={git_commit}\n"
    with _ACCESS_LOG.open("a", encoding="utf-8") as f:
        f.write(entry)
    logger.info("Test access logged: %s", entry.strip())
