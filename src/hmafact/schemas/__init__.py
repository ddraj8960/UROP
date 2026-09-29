"""
hmafact schemas package
"""
from hmafact.schemas.input import BenchmarkSample, UserQuery
from hmafact.schemas.generator import GenerateRequest, GenerateResponse
from hmafact.schemas.claims import Claim, ExtractClaimsRequest, ExtractClaimsResponse, GenerateQueriesRequest, GenerateQueriesResponse
from hmafact.schemas.evidence import EvidencePassage, SearchRequest, SearchResponse, SourceName
from hmafact.schemas.graph import GraphSnapshot, BuildGraphRequest, BuildGraphResponse, NLIScores
from hmafact.schemas.fusion import ClaimEvidenceFusion, FuseGraphEvidenceRequest, FuseGraphEvidenceResponse
from hmafact.schemas.verification import ClaimVerificationResult, VerifyClaimsRequest, VerifyClaimsResponse, ClaimVerdictType
from hmafact.schemas.logic import LogicContradiction, LogicValidationResult
from hmafact.schemas.confidence import ConfidenceMetrics
from hmafact.schemas.synthesis import SynthesizedResponse

__all__ = [
    "BenchmarkSample",
    "UserQuery",
    "GenerateRequest",
    "GenerateResponse",
    "Claim",
    "ExtractClaimsRequest",
    "ExtractClaimsResponse",
    "GenerateQueriesRequest",
    "GenerateQueriesResponse",
    "EvidencePassage",
    "SearchRequest",
    "SearchResponse",
    "SourceName",
    "GraphSnapshot",
    "BuildGraphRequest",
    "BuildGraphResponse",
    "NLIScores",
    "ClaimEvidenceFusion",
    "FuseGraphEvidenceRequest",
    "FuseGraphEvidenceResponse",
    "ClaimVerificationResult",
    "VerifyClaimsRequest",
    "VerifyClaimsResponse",
    "ClaimVerdictType",
    "LogicContradiction",
    "LogicValidationResult",
    "ConfidenceMetrics",
    "SynthesizedResponse",
]
