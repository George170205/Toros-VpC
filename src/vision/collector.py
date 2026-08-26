import uuid
import math
from datetime import datetime
from dataclasses import dataclass, field
from src.domain.enums import TrackState
from src.domain.vision_models import FaceSample, TrackedFace
from src.vision.exceptions import SampleCollectionError

@dataclass(slots=True)
class TrackSession:
    session_id: str
    track_id: int
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    state: TrackState = TrackState.NEW
    total_frames: int = 0
    valid_frames: int = 0
    rejected_frames: int = 0
    samples: list[FaceSample] = field(default_factory=list)
    identification_attempts: int = 0
    visit_registered: bool = False
    revision: int = 0
    public_code: str | None = None

class SampleCollector:
    """Colector que almacena muestras de rostros manteniendo calidad y diversidad de pose."""
    def __init__(self, max_samples: int = 5, min_valid_samples: int = 3, session_timeout_sec: float = 2.5):
        self.max_samples = max_samples
        self.min_valid_samples = min_valid_samples
        self.session_timeout_sec = session_timeout_sec

    def add_sample(self, session: TrackSession, sample: FaceSample) -> None:
        try:
            session.total_frames += 1
            session.last_seen = sample.timestamp

            # Si la muestra no es aceptable por calidad, se descarta y registra
            if not sample.quality.is_acceptable:
                session.rejected_frames += 1
                return

            session.valid_frames += 1

            # Si hay espacio, añadirla directamente
            if len(session.samples) < self.max_samples:
                session.samples.append(sample)
                session.revision += 1
                return

            # Si la cola está llena, evaluar reemplazo por calidad o diversidad de pose
            # Encontrar la muestra en el set con pose más similar
            most_similar_idx = -1
            min_pose_dist = float('inf')
            
            new_pose = sample.quality.pose
            for idx, existing in enumerate(session.samples):
                ext_pose = existing.quality.pose
                # Distancia euclidiana en el espacio de pose (yaw, pitch, roll)
                dist = math.sqrt(
                    (new_pose.yaw - ext_pose.yaw) ** 2 +
                    (new_pose.pitch - ext_pose.pitch) ** 2 +
                    (new_pose.roll - ext_pose.roll) ** 2
                )
                if dist < min_pose_dist:
                    min_pose_dist = dist
                    most_similar_idx = idx

            # Si es muy similar a una existente (ej. menos de 12 grados de diferencia total),
            # y tiene mayor calidad, reemplazamos esa muestra redundante
            if min_pose_dist < 12.0:
                if sample.quality.overall_score > session.samples[most_similar_idx].quality.overall_score:
                    session.samples[most_similar_idx] = sample
                    session.revision += 1
                return

            # Si no es redundante, reemplazamos la de menor calidad general
            worst_idx = -1
            worst_score = float('inf')
            for idx, existing in enumerate(session.samples):
                if existing.quality.overall_score < worst_score:
                    worst_score = existing.quality.overall_score
                    worst_idx = idx

            if sample.quality.overall_score > worst_score:
                session.samples[worst_idx] = sample
                session.revision += 1
        except Exception as e:
            raise SampleCollectionError(f"Error al recolectar muestra para sesión {session.session_id}: {e}") from e

    def is_ready(self, session: TrackSession) -> bool:
        """Determina si la sesión está lista por suficiencia de muestras o por timeout."""
        num_samples = len(session.samples)
        if num_samples == 0:
            return False

        # Criterio 1: Suficiencia de muestras
        if num_samples >= self.min_valid_samples:
            return True

        # Criterio 2: Timeout (duración del track) si hay al menos una muestra
        duration = (session.last_seen - session.first_seen).total_seconds()
        if duration >= self.session_timeout_sec and num_samples >= 1:
            return True

        return False
