from datetime import datetime
import numpy as np
from src.domain.vision_models import Frame, FaceDetection, TrackedFace
from src.vision.exceptions import TrackingError

def calculate_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Calcula la Intersección sobre la Unión (IoU) entre dos bboxes (xmin, ymin, xmax, ymax)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    union_area = float(box1_area + box2_area - inter_area)
    if union_area == 0.0:
        return 0.0
    return inter_area / union_area

class Tracklet:
    def __init__(self, track_id: int, frame_id: int, bbox: tuple[int, int, int, int], landmarks: np.ndarray, confidence: float, timestamp: datetime):
        self.track_id = track_id
        self.last_bbox = bbox
        self.last_landmarks = landmarks
        self.last_confidence = confidence
        self.last_seen_frame = frame_id
        self.last_seen_time = timestamp
        self.first_seen_time = timestamp

    def update(self, frame_id: int, bbox: tuple[int, int, int, int], landmarks: np.ndarray, confidence: float, timestamp: datetime):
        self.last_bbox = bbox
        self.last_landmarks = landmarks
        self.last_confidence = confidence
        self.last_seen_frame = frame_id
        self.last_seen_time = timestamp

class SimpleFaceTracker:
    """Rastreador temporal (tracker) que asocia detecciones basándose en IoU y correspondencia codiciosa."""
    def __init__(self, iou_threshold: float = 0.35, max_lost_frames: int = 25):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.active_tracks: list[Tracklet] = []
        self._next_track_id = 1

    def update(self, frame: Frame, detections: list[FaceDetection]) -> list[TrackedFace]:
        try:
            matched_detections = set()
            matched_tracks = set()
            
            # 1. Primera Fase: Asociación estricta por Overlap (IoU >= iou_threshold)
            associations_iou = []
            for d_idx, det in enumerate(detections):
                for t_idx, track in enumerate(self.active_tracks):
                    iou = calculate_iou(det.bbox, track.last_bbox)
                    if iou >= self.iou_threshold:
                        associations_iou.append((iou, d_idx, t_idx))
                        
            # Ordenar por IoU descendente
            associations_iou.sort(key=lambda x: x[0], reverse=True)
            
            # Asignar por IoU
            for iou, d_idx, t_idx in associations_iou:
                if d_idx not in matched_detections and t_idx not in matched_tracks:
                    matched_detections.add(d_idx)
                    matched_tracks.add(t_idx)
                    det = detections[d_idx]
                    self.active_tracks[t_idx].update(
                        frame_id=frame.frame_id,
                        bbox=det.bbox,
                        landmarks=det.landmarks,
                        confidence=det.confidence,
                        timestamp=frame.timestamp
                    )
            
            # 2. Segunda Fase: Asociación por Distancia de Centroides para tracks remanentes (fallback)
            # Útil cuando los FPS de procesamiento son bajos y la cara se desplaza rápido perdiendo overlap (IoU = 0)
            associations_dist = []
            for d_idx, det in enumerate(detections):
                if d_idx in matched_detections:
                    continue
                
                det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                det_w = det.bbox[2] - det.bbox[0]
                det_h = det.bbox[3] - det.bbox[1]
                max_dist = max(det_w, det_h) * 1.5  # Distancia máxima permitida en base al tamaño de la cara
                
                for t_idx, track in enumerate(self.active_tracks):
                    if t_idx in matched_tracks:
                        continue
                    
                    track_cx = (track.last_bbox[0] + track.last_bbox[2]) / 2.0
                    track_cy = (track.last_bbox[1] + track.last_bbox[3]) / 2.0
                    
                    # Distancia euclídea
                    dist = float(np.sqrt((det_cx - track_cx)**2 + (det_cy - track_cy)**2))
                    if dist <= max_dist:
                        associations_dist.append((dist, d_idx, t_idx))
                        
            # Ordenar por distancia ascendente (más cercanos primero)
            associations_dist.sort(key=lambda x: x[0])
            
            # Asignar por Distancia de Centroides
            for dist, d_idx, t_idx in associations_dist:
                if d_idx not in matched_detections and t_idx not in matched_tracks:
                    matched_detections.add(d_idx)
                    matched_tracks.add(t_idx)
                    det = detections[d_idx]
                    self.active_tracks[t_idx].update(
                        frame_id=frame.frame_id,
                        bbox=det.bbox,
                        landmarks=det.landmarks,
                        confidence=det.confidence,
                        timestamp=frame.timestamp
                    )

            # 3. Crear nuevos tracks para detecciones no asociadas de confianza alta
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_detections and det.confidence >= 0.45:
                    new_track = Tracklet(
                        track_id=self._next_track_id,
                        frame_id=frame.frame_id,
                        bbox=det.bbox,
                        landmarks=det.landmarks,
                        confidence=det.confidence,
                        timestamp=frame.timestamp
                    )
                    self._next_track_id += 1
                    self.active_tracks.append(new_track)
            
            # 4. Limpiar tracks inactivos
            active_still = []
            for track in self.active_tracks:
                if frame.frame_id - track.last_seen_frame <= self.max_lost_frames:
                    active_still.append(track)
            self.active_tracks = active_still
            
            # 5. Generar lista de TrackedFace para el frame actual
            tracked_faces = []
            for track in self.active_tracks:
                if track.last_seen_frame == frame.frame_id:
                    tracked_faces.append(TrackedFace(
                        track_id=track.track_id,
                        frame_id=frame.frame_id,
                        camera_id=frame.camera_id,
                        timestamp=frame.timestamp,
                        bbox=track.last_bbox,
                        landmarks=track.last_landmarks,
                        detection_confidence=track.last_confidence
                    ))
            return tracked_faces
        except Exception as e:
            raise TrackingError(f"Error al actualizar el tracker en frame {frame.frame_id}: {e}") from e

    def reset(self) -> None:
        self.active_tracks.clear()
        self._next_track_id = 1
