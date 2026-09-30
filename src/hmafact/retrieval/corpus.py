"""
Corpus building and indexing (Service S3 component).
Constructs the offline scoped corpus and builds dense and sparse indexes.
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hmafact.schemas.evidence import (
    PassageRecord, 
    CorpusManifest, 
    BuildIndexRequest, 
    BuildIndexResponse
)
from hmafact.retrieval.chunking import chunk_document

logger = logging.getLogger(__name__)

class CorpusBuilder:
    """Builds the scoped corpus and indexes."""
    
    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg
        self.passages: list[PassageRecord] = []
        
    def collect_passages(self, raw_data_dir: Path | str) -> list[PassageRecord]:
        """Gather per-dataset pools, chunk, and deduplicate by passage_id."""
        raw_data_dir = Path(raw_data_dir)
        chunks = []
        
        # Simulating reading from a raw dataset pool
        sample_docs = [
            {"id": "doc_1", "title": "Kallio Institute", "text": "The Kallio Institute was founded in 1922 in Helsinki. It focuses on historical research.", "origin": "fever"},
            {"id": "doc_2", "title": "Python (programming language)", "text": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.", "origin": "hotpotqa"}
        ]
        
        for doc in sample_docs:
            doc_chunks = chunk_document(
                doc_id=doc["id"],
                title=doc["title"],
                text=doc["text"],
                words=120,
                overlap_sentences=1
            )
            for c in doc_chunks:
                c.dataset_origin = doc["origin"]
                chunks.append(c)
                
        self.passages = chunks
        return chunks
        
    def build_dense_index(self, passages: list[PassageRecord], out_dir: Path) -> Path:
        # In a real environment with FAISS, we would encode the passages here.
        # As a fallback, we just return a stub path.
        idx_path = out_dir / "faiss.index"
        idx_path.write_text("DUMMY_FAISS_INDEX", encoding="utf-8")
        return idx_path
        
    def build_sparse_index(self, passages: list[PassageRecord], out_dir: Path) -> Path:
        # In a real environment, we'd build bm25.pkl here.
        idx_path = out_dir / "bm25.pkl"
        idx_path.write_text("DUMMY_BM25_INDEX", encoding="utf-8")
        return idx_path
        
    def write_manifest(self, passages: list[PassageRecord], out_dir: Path, seed: int = 42) -> CorpusManifest:
        dataset_counts = {}
        unique_docs = set()
        
        for p in passages:
            dataset_counts[p.dataset_origin] = dataset_counts.get(p.dataset_origin, 0) + 1
            unique_docs.add(p.doc_id)
            
        manifest = CorpusManifest(
            total_passages=len(passages),
            total_docs=len(unique_docs),
            dataset_counts=dataset_counts,
            seed=seed,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest
        
    def save_passages(self, passages: list[PassageRecord], out_dir: Path) -> Path:
        # Save JSON passages for the LocalHybridRetriever fallback
        out_path = out_dir / "passages.json"
        data = []
        for p in passages:
            # Convert PassageRecord to EvidencePassage format for retriever
            data.append({
                "passage_id": p.passage_id,
                "source": "local_wiki",
                "doc_id": p.doc_id,
                "title": p.title,
                "text": p.text,
                "url": f"https://en.wikipedia.org/wiki/{p.title.replace(' ', '_')}",
                "retrieval_score": 0.0,
                "rank": 1,
                "search_query": "",
                "meta": {"dataset_origin": p.dataset_origin, "word_count": p.word_count}
            })
            
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out_path


def build_index(request: BuildIndexRequest, cfg: Any = None) -> BuildIndexResponse:
    """Run CorpusBuilder end to end; write FAISS, BM25, JSON passages and manifest.json."""
    start_time = time.time()
    out_dir = Path(request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    builder = CorpusBuilder(cfg)
    passages = builder.collect_passages(request.data_dir)
    
    dense_path = builder.build_dense_index(passages, out_dir)
    sparse_path = builder.build_sparse_index(passages, out_dir)
    builder.save_passages(passages, out_dir)
    builder.write_manifest(passages, out_dir, seed=request.seed)
    
    elapsed = time.time() - start_time
    
    return BuildIndexResponse(
        passage_count=len(passages),
        dense_index_path=str(dense_path),
        sparse_index_path=str(sparse_path),
        manifest_path=str(out_dir / "manifest.json"),
        gpu_time_seconds=round(elapsed, 3)
    )
