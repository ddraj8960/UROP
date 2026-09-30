"""
schemas/data.py — Contract models for the M1 data pipeline.

These Pydantic models define the typed API surface for build_benchmark_splits.py
and loader.py so that callers and tests can rely on structured, validated I/O
rather than raw dicts.

Exports:
    BuildSplitsRequest   — parameters passed into the split-build pipeline
    LoadSplitRequest     — parameters for loading a single frozen split
    SplitFileMetadata    — per-file entry recorded in the manifest
    StratumBreakdown     — per-stratum sample count for a split
    SplitManifest        — full manifest.json schema
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# BuildSplitsRequest
# ---------------------------------------------------------------------------

class BuildSplitsRequest(BaseModel):
    """
    Parameters for the build_benchmark_splits pipeline.

    Passed from CLI args → main() → dataset fetchers / splitters so that
    the entire pipeline is deterministic and re-runnable from a single object.
    """
    seed: int = Field(42, ge=0, description="Random seed for reproducibility.")
    n_dev: int = Field(100, gt=0, description="Target dev-split size per dataset.")
    n_test: int = Field(300, gt=0, description="Target test-split size per dataset.")
    n_ablation: int = Field(150, gt=0, description="Frozen ablation subset size per dataset.")
    output_dir: Path = Field(
        Path("data/samples"),
        description="Directory where JSONL files and manifest.json are written.",
    )
    excluded_ids_path: Path = Field(
        Path("data/excluded_ids.txt"),
        description="Path to the bake-off leakage exclusion list.",
    )
    fever_cache: Path = Field(
        Path("data/.cache/fever_dev_raw.jsonl"),
        description="Local cache path for the raw FEVER download.",
    )
    hotpot_cache: Path = Field(
        Path("data/.cache/hotpot_qa_dev.parquet"),
        description="Local cache path for the raw HotpotQA parquet download.",
    )

    @field_validator("n_ablation")
    @classmethod
    def ablation_lte_test(cls, v: int, info) -> int:
        n_test = info.data.get("n_test", v)
        if v > n_test:
            raise ValueError(
                f"n_ablation ({v}) must be <= n_test ({n_test})."
            )
        return v


# ---------------------------------------------------------------------------
# LoadSplitRequest
# ---------------------------------------------------------------------------

class LoadSplitRequest(BaseModel):
    """
    Parameters for loader.load_samples().

    Using a typed request object lets callers be explicit about intent and
    avoids the common mistake of passing `final=True` carelessly.
    """
    dataset: Literal["truthfulqa", "fever", "hotpotqa"] = Field(
        ..., description="Which benchmark dataset to load."
    )
    split: Literal["dev", "test"] = Field(
        ..., description="Which split to load."
    )
    final: bool = Field(
        False,
        description=(
            "Must be True to unlock the test split. "
            "Set only in M14 final evaluation runs."
        ),
    )
    limit: int | None = Field(
        None,
        gt=0,
        description="If set, return only the first N rows (for smoke-tests).",
    )
    data_dir: Path | None = Field(
        None,
        description="Override the default samples directory.",
    )


# ---------------------------------------------------------------------------
# SplitFileMetadata
# ---------------------------------------------------------------------------

class SplitFileMetadata(BaseModel):
    """
    Per-file entry in the manifest's 'files' dict.
    Matches the dict written by write_jsonl() in build_benchmark_splits.py.
    """
    rows: int = Field(..., ge=0, description="Number of samples in this file.")
    sha256: str = Field(..., min_length=64, max_length=64, description="Hex SHA-256 of the raw UTF-8 bytes.")
    excluded_ids_applied: int = Field(
        0,
        ge=0,
        description="Number of sample_ids removed from this file via excluded_ids.txt.",
    )
    strata: dict[str, int] = Field(
        default_factory=dict,
        description="Per-stratum sample count, e.g. {'bridge': 240, 'comparison': 60}.",
    )


# ---------------------------------------------------------------------------
# StratumBreakdown
# ---------------------------------------------------------------------------

class StratumBreakdown(BaseModel):
    """
    Per-stratum statistics for a single split of a single dataset.
    Used by manifest reporting and unit tests.
    """
    stratum: str = Field(..., description="Stratum label, e.g. 'bridge', 'SUPPORTS', 'Misconceptions'.")
    n_dev: int = Field(..., ge=0)
    n_test: int = Field(..., ge=0)
    n_ablation: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# SplitManifest
# ---------------------------------------------------------------------------

class SplitManifest(BaseModel):
    """
    Full schema for data/samples/manifest.json.

    Written by build_manifest() in build_benchmark_splits.py.
    Loaded and validated by loader._verify_manifest() for integrity checking.
    """
    seed: int = Field(..., ge=0)
    n_dev: int = Field(..., gt=0)
    n_test: int = Field(..., gt=0)
    n_ablation: int = Field(..., gt=0)
    excluded_ids_applied: int = Field(
        0,
        ge=0,
        description="Total number of sample_ids removed across all datasets via excluded_ids.txt.",
    )
    built_at: datetime = Field(..., description="UTC timestamp when splits were built.")
    git_commit: str = Field(..., description="Git commit SHA at build time.")
    python_version: str
    pydantic_version: str
    files: dict[str, SplitFileMetadata] = Field(
        default_factory=dict,
        description="Keyed by filename, e.g. 'truthfulqa_dev.jsonl'.",
    )
    datasets: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-dataset provenance: source_url, revision, raw_rows, filtered_rows.",
    )
