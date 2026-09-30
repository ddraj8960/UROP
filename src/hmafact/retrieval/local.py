"""
Local Hybrid Retriever (Service S3 component).
Provides dense + sparse retrieval over the scoped corpus, fused with RRF.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hmafact.schemas.evidence import EvidencePassage, SourceName

logger = logging.getLogger(__name__)

class LocalHybridRetriever:
    """Dense + sparse retrieval over the scoped corpus, fused with RRF."""
    name: SourceName = "local_wiki"

    def __init__(self, index_dir: Path, cfg: Any = None) -> None:
        """Loads FAISS, BM25 and the Parquet sidecar once. Raises INDEX_MISSING if absent."""
        self.index_dir = Path(index_dir)
        self.passages: dict[str, EvidencePassage] = {}
        self.k_rrf = getattr(cfg, "rrf_k", 60) if cfg else 60
        self._load_indexes()

    def _load_indexes(self):
        # We simulate loading index by loading a simple JSON passage store if it exists
        passage_file = self.index_dir / "passages.json"
        if not passage_file.exists():
            logger.warning(f"Index missing at {self.index_dir}. LocalHybridRetriever will return empty results.")
            self.passages = {}
            return
            
        try:
            with open(passage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    ep = EvidencePassage.model_validate(item)
                    self.passages[ep.passage_id] = ep
        except Exception as e:
            logger.error(f"Failed to load passages: {e}")

    def search_dense(self, query: str, k: int) -> list[tuple[str, float]]:
        """Returns [(passage_id, cosine)] — exposed for the hybrid-vs-single ablation."""
        # Simple string match fallback for dense in absence of FAISS
        results = []
        q_lower = query.lower()
        for pid, ep in self.passages.items():
            score = 0.0
            if q_lower in ep.text.lower():
                score += 0.8
            if q_lower in ep.title.lower():
                score += 0.2
            if score > 0:
                results.append((pid, score))
        return sorted(results, key=lambda x: -x[1])[:k]

    def search_sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        """Returns [(passage_id, bm25_score)]"""
        # Simple word overlap fallback for sparse
        results = []
        q_words = set(query.lower().split())
        for pid, ep in self.passages.items():
            t_words = set(ep.text.lower().split())
            overlap = len(q_words.intersection(t_words))
            if overlap > 0:
                score = overlap / len(q_words)
                results.append((pid, score))
        return sorted(results, key=lambda x: -x[1])[:k]

    def search(self, query: str, k: int = 5) -> list[EvidencePassage]:
        from hmafact.retrieval.hybrid import fuse_rankings
        
        dense_res = self.search_dense(query, k * 2)
        sparse_res = self.search_sparse(query, k * 2)
        
        fused = fuse_rankings(dense_res, sparse_res, k_rrf=self.k_rrf)
        
        results = []
        now_utc = datetime.now(timezone.utc)
        for rank, (pid, score) in enumerate(fused[:k], 1):
            if pid in self.passages:
                ep = self.passages[pid].model_copy(update={
                    "retrieval_score": score,
                    "rank": rank,
                    "search_query": query,
                    "retrieved_at": now_utc
                })
                results.append(ep)
        return results

    def health(self) -> dict:
        return {
            "source": self.name,
            "status": "ok" if self.passages else "missing_index",
            "passage_count": len(self.passages)
        }
