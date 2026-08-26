from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True, slots=True)
class VisitEvent:
    event_id: str
    person_id: str
    session_id: str
    track_id: int
    camera_id: str
    timestamp: datetime
    decision_id: str
    confidence: float
    entry_zone: str | None = None

@dataclass(frozen=True, slots=True)
class IdentificationDecisionRecord:
    decision_id: str
    track_id: int
    camera_id: str
    selected_person_id: str | None
    decision_type: str
    decision_confidence: float
    best_candidate_score: float | None
    second_candidate_score: float | None
    margin: float | None
    average_quality: float
    consistency_score: float
    positive_match_ratio: float
    temporal_confirmation: float
    reason_codes: list[str]
    decision_engine_version: str
    configuration_id: str
    created_at: datetime

@dataclass(frozen=True, slots=True)
class Camera:
    camera_id: str
    name: str
    location: str | None
    entry_zone: str | None
    resolution_width: int
    resolution_height: int
    target_fps: float
    active: bool

@dataclass(frozen=True, slots=True)
class SystemConfiguration:
    configuration_id: str
    version: str
    min_quality: float
    top_k: int
    candidate_delta: float
    min_verification_score: float
    min_margin: float
    min_consistency: float
    max_secondary_embeddings: int
    primary_update_lambda: float
    created_at: datetime
    active: bool
