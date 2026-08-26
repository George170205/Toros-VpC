from dataclasses import dataclass
from datetime import datetime
import numpy as np
from src.domain.enums import PersonStatus, EmbeddingType
from src.domain.vision_models import FacePose

@dataclass(slots=True)
class PersonIdentity:
    person_id: str
    public_code: str
    status: PersonStatus
    primary_embedding_id: str | None
    first_seen: datetime
    last_seen: datetime
    total_visits: int

@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    embedding_id: str
    person_id: str
    embedding_type: EmbeddingType
    vector: np.ndarray
    quality_score: float
    pose: FacePose | None
    model_name: str
    model_version: str
    dimension: int
    active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class EmbeddingEvidence:
    sample_id: str
    track_id: int
    embedding: np.ndarray
    quality_score: float
    pose: FacePose | None
    weight: float
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class IdentityEvidence:
    evidence_id: str
    session_id: str
    track_id: int
    camera_id: str
    embeddings: list[EmbeddingEvidence]
    aggregated_embedding: np.ndarray
    total_samples: int
    valid_samples: int
    rejected_samples: int
    average_quality: float
    embedding_consistency: float
    aggregation_weights: list[float]
    created_at: datetime
