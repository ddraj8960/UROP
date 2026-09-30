"""
Retrieval source protocol for S3 Retrieval Service.
"""
from typing import Protocol
from hmafact.schemas.evidence import EvidencePassage, SourceName

class EvidenceSource(Protocol):
    """One knowledge source. Stateless per call; may hold warm state internally."""
    name: SourceName

    def search(self, query: str, k: int = 5) -> list[EvidencePassage]:
        """Return up to k passages ranked by relevance, scores normalised to 0-1."""
        ...

    def health(self) -> dict:
        """Readiness plus size/version info for /healthz."""
        ...
