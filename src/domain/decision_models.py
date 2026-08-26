from dataclasses import dataclass
from datetime import datetime
import numpy as np
from src.domain.enums import DecisionType

@dataclass(frozen=True, slots=True)
class CandidateMatch:
    person_id: str
    rank: int
    primary_similarity: float
    index_score: float
    retrieved_at: datetime

@dataclass(frozen=True, slots=True)
class SimilarityMatrix:
    values: np.ndarray  # M x N matrix
    current_embedding_ids: list[str]
    candidate_embedding_ids: list[str]

@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    person_id: str
    rank: int
    primary_similarity: float
    similarity_matrix: SimilarityMatrix
    best_similarity: float
    best_per_sample: list[float]
    mean_top_similarity: float
    median_top_similarity: float
    consistency_score: float
    positive_match_ratio: float
    average_sample_quality: float
    verification_score: float

@dataclass(frozen=True, slots=True)
class DecisionResult:
    decision_id: str
    session_id: str
    track_id: int
    camera_id: str
    decision: DecisionType
    person_id: str | None
    confidence: float
    best_candidate_score: float | None
    second_candidate_score: float | None
    margin: float | None
    quality_score: float
    consistency_score: float
    positive_match_ratio: float
    temporal_confirmation: float
    reason_codes: list[str]
    timestamp: datetime

@dataclass(frozen=True, slots=True)
class IdentityUpdateDecision:
    should_update: bool
    update_primary: bool
    add_secondary: bool
    replace_secondary_id: str | None
    confidence: float
    reason_codes: list[str]

@dataclass(frozen=True, slots=True)
class NewIdentityDecision:
    should_create: bool
    primary_embedding: np.ndarray | None
    secondary_embeddings: list[np.ndarray]
    quality_score: float
    evidence_id: str
    reason_codes: list[str]
