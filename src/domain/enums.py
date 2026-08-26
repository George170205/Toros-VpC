from enum import Enum, auto

class TrackState(Enum):
    NEW = auto()
    COLLECTING = auto()
    READY = auto()
    GENERATING_EMBEDDINGS = auto()
    VALIDATING_EVIDENCE = auto()
    SEARCHING = auto()
    VERIFYING = auto()
    IDENTIFIED = auto()
    AMBIGUOUS = auto()
    UNKNOWN = auto()
    EXPIRED = auto()
    FINISHED = auto()

class DecisionType(Enum):
    MATCH = auto()
    AMBIGUOUS = auto()
    UNKNOWN = auto()

class EmbeddingType(Enum):
    PRIMARY = auto()
    SECONDARY = auto()

class PersonStatus(Enum):
    ACTIVE = auto()
    INACTIVE = auto()
    REVIEW = auto()
    MERGED = auto()
    DELETED = auto()
