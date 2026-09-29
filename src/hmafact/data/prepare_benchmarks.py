"""
prepare_benchmarks.py — Per-dataset fetchers and converters.

TruthfulQA : Hugging Face datasets library
FEVER      : Direct download from fever.ai (Option B — no HF loader)
HotpotQA   : Hugging Face datasets library
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests

from hmafact.schemas.input import BenchmarkSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha_id(question: str, prefix: str) -> str:
    """Stable sample_id from content hash (avoids fragile row-index IDs)."""
    h = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{h}"


# ---------------------------------------------------------------------------
# TruthfulQA
# ---------------------------------------------------------------------------

def fetch_truthfulqa(revision: str | None = None) -> tuple[list[BenchmarkSample], str]:
    """
    Download TruthfulQA (generation config) from Hugging Face.

    Returns (samples, resolved_revision_sha).
    """
    from datasets import load_dataset  # type: ignore[import]

    logger.info("Fetching TruthfulQA from Hugging Face...")
    # datasets 5.x requires full namespace/name format
    kwargs: dict[str, Any] = {"path": "truthfulqa/truthful_qa", "name": "generation", "split": "validation"}
    if revision:
        kwargs["revision"] = revision

    ds = load_dataset(**kwargs)
    # datasets 5.x: info.version may not exist; use dataset_info commit hash or fallback
    resolved_revision: str = getattr(ds.info, "version", None) or getattr(ds.info, "dataset_name", "unknown")

    samples: list[BenchmarkSample] = []
    for row in ds:
        sid = _sha_id(row["question"], "truthfulqa")
        samples.append(
            BenchmarkSample(
                sample_id=sid,
                dataset="truthfulqa",
                split="dev",          # placeholder; overwritten by the splitter
                input_text=row["question"],
                gold_answer=row["best_answer"],
                gold_correct_answers=list(row["correct_answers"]),
                gold_incorrect_answers=list(row["incorrect_answers"]),
                meta={
                    "category": row.get("category", ""),
                    "type": row.get("type", ""),
                    "source": row.get("source", ""),
                },
            )
        )

    logger.info("TruthfulQA: %d rows fetched.", len(samples))
    return samples, resolved_revision


# ---------------------------------------------------------------------------
# FEVER — Option B: direct download from fever.ai
# ---------------------------------------------------------------------------

FEVER_DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"


def fetch_fever(cache_path: Path | None = None) -> tuple[list[BenchmarkSample], str]:
    """
    Download FEVER shared_task_dev.jsonl directly from fever.ai.

    Each line is one claim (already grouped; no row-per-evidence issue).
    Returns (samples, sha256_of_downloaded_file).

    Args:
        cache_path: If provided and the file already exists, skip the download.
    """
    if cache_path and cache_path.exists():
        logger.info("FEVER: using cached file at %s", cache_path)
        raw_bytes = cache_path.read_bytes()
    else:
        logger.info("FEVER: downloading from %s", FEVER_DEV_URL)
        resp = requests.get(FEVER_DEV_URL, timeout=120)
        resp.raise_for_status()
        raw_bytes = resp.content
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw_bytes)
            logger.info("FEVER: saved to %s", cache_path)

    revision_sha = hashlib.sha256(raw_bytes).hexdigest()
    lines = raw_bytes.decode("utf-8").splitlines()

    samples: list[BenchmarkSample] = []
    seen_ids: set[int] = set()

    for line in lines:
        if not line.strip():
            continue
        row: dict[str, Any] = json.loads(line)
        native_id: int = row["id"]
        if native_id in seen_ids:
            continue  # deduplicate (shouldn't happen in shared_task_dev, but defensive)
        seen_ids.add(native_id)

        label: str = row["label"]  # "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"
        evidence_sets: list[list[tuple[str, int]]] = []
        flat_evidence: list[tuple[str, int]] = []
        seen_pairs: set[tuple[str, int]] = set()

        for ann_set in row.get("evidence", []):
            ev_set: list[tuple[str, int]] = []
            for sent in ann_set:
                # sent = [annotation_id, evidence_id, wiki_url, sentence_id]
                wiki_url = sent[2]  # e.g. "Nikolaj_Coster-Waldau" (raw underscore form)
                sent_id = sent[3]   # int or None
                if wiki_url and sent_id is not None and sent_id >= 0:
                    pair = (wiki_url, int(sent_id))
                    ev_set.append(pair)
                    if pair not in seen_pairs:
                        flat_evidence.append(pair)
                        seen_pairs.add(pair)
            if ev_set:
                evidence_sets.append(ev_set)

        # FEVER NEI rows may have empty evidence — keep gold_evidence=[]
        # SUPPORTS/REFUTES rows without valid evidence are filtered later in sampling

        samples.append(
            BenchmarkSample(
                sample_id=f"fever:{native_id}",
                dataset="fever",
                split="dev",  # placeholder; overwritten by the splitter
                input_text=row["claim"],
                gold_label=label,  # type: ignore[arg-type]
                gold_evidence=sorted(flat_evidence),
                meta={
                    "evidence_sets": [
                        [[title, sid] for title, sid in ev_set]
                        for ev_set in evidence_sets
                    ],
                    "verifiable": row.get("verifiable", ""),
                },
            )
        )

    logger.info("FEVER: %d unique claims parsed.", len(samples))
    return samples, revision_sha


# ---------------------------------------------------------------------------
# HotpotQA (distractor, validation)
# ---------------------------------------------------------------------------

HOTPOTQA_DEV_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)


def fetch_hotpotqa(cache_path: Path | None = None) -> tuple[list[BenchmarkSample], str]:
    """
    Download HotpotQA distractor validation parquet directly from HuggingFace CDN.
    Parses the parquet with pyarrow — no datasets library, no Xet-protocol hang.

    Returns (samples, sha256_of_downloaded_file).

    Args:
        cache_path: If provided and file exists, skip the download.
    """
    import pyarrow.parquet as pq  # type: ignore[import]
    import io

    if cache_path and cache_path.exists():
        logger.info("HotpotQA: using cached file at %s", cache_path)
        raw_bytes = cache_path.read_bytes()
    else:
        logger.info("HotpotQA: downloading parquet from HuggingFace CDN (~27MB)...")
        resp = requests.get(HOTPOTQA_DEV_URL, timeout=120, stream=True)
        resp.raise_for_status()

        chunks = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded % (5 * 1024 * 1024) < 1024 * 1024:  # log every ~5MB
                logger.info("HotpotQA: downloaded %.1f MB...", downloaded / 1024 / 1024)
        raw_bytes = b"".join(chunks)
        logger.info("HotpotQA: download complete (%.1f MB)", len(raw_bytes) / 1024 / 1024)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw_bytes)
            logger.info("HotpotQA: saved to %s", cache_path)

    revision_sha = hashlib.sha256(raw_bytes).hexdigest()

    # Parse parquet
    table = pq.read_table(io.BytesIO(raw_bytes))
    df = table.to_pydict()
    n_rows = len(df["id"])

    samples: list[BenchmarkSample] = []
    for i in range(n_rows):
        level = df["level"][i]
        assert level == "hard", f"HotpotQA row {i}: expected level=hard, got {level}"

        # Supporting facts: {"title": [...], "sent_id": [...]}
        sf = df["supporting_facts"][i]
        gold_ev = list(zip(sf["title"], sf["sent_id"]))

        # Context: {"title": [...], "sentences": [[...],...]}
        ctx = df["context"][i]
        context = [
            {"title": t, "sentences": list(sents)}
            for t, sents in zip(ctx["title"], ctx["sentences"])
        ]

        samples.append(
            BenchmarkSample(
                sample_id=f"hotpotqa:{df['id'][i]}",
                dataset="hotpotqa",
                split="dev",  # placeholder; overwritten by the splitter
                input_text=df["question"][i],
                gold_answer=df["answer"][i],
                gold_evidence=gold_ev,
                meta={
                    "type": df["type"][i],    # "bridge" or "comparison"
                    "level": df["level"][i],  # always "hard"
                    "context": context,        # 10 paragraphs for M3 corpus construction
                },
            )
        )

    logger.info("HotpotQA: %d rows parsed.", len(samples))
    return samples, revision_sha

    """
    Download HotpotQA distractor dev set directly from the official CMU source.
    Option B — direct download, no HuggingFace datasets library (avoids hanging).

    Returns (samples, sha256_of_downloaded_file).

    Args:
        cache_path: If provided and file exists, skip the download.
    """
    if cache_path and cache_path.exists():
        logger.info("HotpotQA: using cached file at %s", cache_path)
        raw_bytes = cache_path.read_bytes()
    else:
        logger.info("HotpotQA: downloading from %s", HOTPOTQA_DEV_URL)
        resp = requests.get(HOTPOTQA_DEV_URL, timeout=300, stream=True)
        resp.raise_for_status()

        chunks = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded % (10 * 1024 * 1024) == 0:  # log every 10MB
                logger.info("HotpotQA: downloaded %.1f MB...", downloaded / 1024 / 1024)
        raw_bytes = b"".join(chunks)
        logger.info("HotpotQA: download complete (%.1f MB)", len(raw_bytes) / 1024 / 1024)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw_bytes)
            logger.info("HotpotQA: saved to %s", cache_path)

    revision_sha = hashlib.sha256(raw_bytes).hexdigest()
    data: list[dict] = json.loads(raw_bytes.decode("utf-8"))

    samples: list[BenchmarkSample] = []
    for row in data:
        assert row["level"] == "hard", (
            f"HotpotQA distractor dev should be all 'hard'; got '{row['level']}' for id {row['_id']}"
        )

        # Gold supporting facts: list of (title, sent_id)
        gold_ev = [(fact[0], fact[1]) for fact in row["supporting_facts"]]

        # Context paragraphs: list of {title, sentences} — 10 items per question (M3 requirement)
        context = [
            {"title": para[0], "sentences": list(para[1])}
            for para in row["context"]
        ]

        samples.append(
            BenchmarkSample(
                sample_id=f"hotpotqa:{row['_id']}",
                dataset="hotpotqa",
                split="dev",  # placeholder; overwritten by the splitter
                input_text=row["question"],
                gold_answer=row["answer"],
                gold_evidence=gold_ev,
                meta={
                    "type": row["type"],    # "bridge" or "comparison"
                    "level": row["level"],  # always "hard"
                    "context": context,     # 10 paragraphs for M3 corpus construction
                },
            )
        )

    logger.info("HotpotQA: %d rows parsed.", len(samples))
    return samples, revision_sha

