import cv2
import numpy as np
import math
from src.domain.vision_models import Frame, TrackedFace, FacePose, QualityResult
from src.vision.exceptions import QualityAnalysisError

class FaceQualityAnalyzer:
    """Analizador de calidad facial que estima nitidez, brillo, tamaño y pose (yaw, pitch, roll)."""
    def __init__(self, min_quality_threshold: float = 0.45):
        self.min_quality_threshold = min_quality_threshold

    def analyze(self, frame: Frame, face: TrackedFace) -> QualityResult:
        try:
            img = frame.image
            h, w = img.shape[:2]
            
            # 1. Obtener coordenadas válidas del bbox
            xmin = max(0, face.bbox[0])
            ymin = max(0, face.bbox[1])
            xmax = min(w, face.bbox[2])
            ymax = min(h, face.bbox[3])
            
            face_w = xmax - xmin
            face_h = ymax - ymin
            
            if face_w <= 0 or face_h <= 0:
                return self._empty_quality_result()

            # Extraer recorte facial en escala de grises
            face_crop = img[ymin:ymax, xmin:xmax]
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            # 2. Score de Tamaño (Size Score)
            # Esperamos rostros de al menos 90px; idealmente >= 160px
            size_score = min(1.0, face_w / 160.0)

            # 3. Score de Nitidez (Sharpness Score)
            # Usar la varianza del Laplaciano
            lap_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            # 100 es aceptable, 180+ es muy nítido
            sharpness_score = min(1.0, lap_var / 180.0)

            # 4. Score de Brillo (Brightness Score)
            # Evaluado respecto al valor medio 127.5 (buena iluminación)
            mean_brightness = float(np.mean(face_gray))
            brightness_score = max(0.0, 1.0 - abs(mean_brightness - 127.5) / 127.5)

            # 5. Estimación de Pose (Yaw, Pitch, Roll) a partir de 5 Landmarks
            # Landmarks: 0: ojo_izq, 1: ojo_der, 2: nariz, 3: boca_izq, 4: boca_der
            kps = face.landmarks
            
            # Roll (Z-axis rotation): Ángulo del vector entre los ojos
            eye_dx = kps[1][0] - kps[0][0]
            eye_dy = kps[1][1] - kps[0][1]
            roll = math.degrees(math.atan2(eye_dy, eye_dx))

            # Yaw (Y-axis rotation): Simetría de la nariz respecto a los ojos
            eye_width = max(1.0, abs(kps[1][0] - kps[0][0]))
            nose_left_dist = abs(kps[2][0] - kps[0][0])
            nose_right_dist = abs(kps[1][0] - kps[2][0])
            # La diferencia normalizada
            yaw_ratio = (nose_right_dist - nose_left_dist) / eye_width
            # Escalar aproximadamente a grados (por ejemplo, yaw_ratio * 60)
            yaw = float(np.clip(yaw_ratio * 60.0, -90.0, 90.0))

            # Pitch (X-axis rotation): Altura relativa de la nariz respecto a los ojos y la boca
            eye_y = (kps[0][1] + kps[1][1]) / 2.0
            mouth_y = (kps[3][1] + kps[4][1]) / 2.0
            mouth_eye_height = max(1.0, abs(mouth_y - eye_y))
            nose_y_dist = kps[2][1] - eye_y
            # En un rostro frontal, la nariz está aproximadamente al 45% de la distancia ojos-boca
            pitch_ratio = (nose_y_dist / mouth_eye_height) - 0.45
            # Escalar a grados
            pitch = float(np.clip(pitch_ratio * 60.0, -90.0, 90.0))

            # 6. Score de Pose (Pose Score)
            # Penaliza desviaciones de la mirada frontal (0,0,0)
            yaw_penalty = abs(yaw) / 40.0
            pitch_penalty = abs(pitch) / 35.0
            roll_penalty = abs(roll) / 25.0
            pose_score = max(0.0, 1.0 - (yaw_penalty + pitch_penalty + roll_penalty))

            # 7. Visibilidad y Oclusiones (Visibility Score)
            # Para el prototipo, estimamos una visibilidad alta a partir de landmarks coherentes
            visibility_score = 1.0 if face.detection_confidence >= 0.45 else face.detection_confidence

            # 8. Score de Calidad Compuesto
            # S = 0.40 * Pose + 0.25 * Nitidez + 0.15 * Tamaño + 0.10 * Brillo + 0.10 * Visibilidad
            overall_score = (
                0.40 * pose_score +
                0.25 * sharpness_score +
                0.15 * size_score +
                0.10 * brightness_score +
                0.10 * visibility_score
            )
            
            is_acceptable = overall_score >= self.min_quality_threshold

            return QualityResult(
                overall_score=float(overall_score),
                sharpness_score=float(sharpness_score),
                brightness_score=float(brightness_score),
                size_score=float(size_score),
                pose_score=float(pose_score),
                visibility_score=float(visibility_score),
                pose=FacePose(yaw=yaw, pitch=pitch, roll=roll),
                is_acceptable=is_acceptable
            )
        except Exception as e:
            raise QualityAnalysisError(f"Error al analizar calidad de cara en frame {frame.frame_id}: {e}") from e

    def _empty_quality_result(self) -> QualityResult:
        return QualityResult(
            overall_score=0.0,
            sharpness_score=0.0,
            brightness_score=0.0,
            size_score=0.0,
            pose_score=0.0,
            visibility_score=0.0,
            pose=FacePose(yaw=0.0, pitch=0.0, roll=0.0),
            is_acceptable=False
        )
