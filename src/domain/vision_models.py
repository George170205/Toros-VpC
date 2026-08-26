from datetime import datetime
import numpy as np
from dataclasses import dataclass, field

@dataclass(slots=True)
class Frame:
    frame_id: int
    camera_id: str
    timestamp: datetime
    image: np.ndarray

@dataclass(frozen=True, slots=True)
class FaceDetection:
    detection_id: str
    frame_id: int
    bbox: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    landmarks: np.ndarray            # 5 keypoints (x, y)
    confidence: float

@dataclass(frozen=True, slots=True)
class TrackedFace:
    track_id: int
    frame_id: int
    camera_id: str
    timestamp: datetime
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    detection_confidence: float

@dataclass(frozen=True, slots=True)
class FacePose:
    yaw: float
    pitch: float
    roll: float

@dataclass(frozen=True, slots=True)
class QualityResult:
    overall_score: float
    sharpness_score: float
    brightness_score: float
    size_score: float
    pose_score: float
    visibility_score: float
    pose: FacePose
    is_acceptable: bool

@dataclass(slots=True)
class FaceSample:
    sample_id: str
    track_id: int
    frame_id: int
    camera_id: str
    timestamp: datetime
    face_image: np.ndarray
    quality: QualityResult
