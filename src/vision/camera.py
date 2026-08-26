import cv2
import time
from datetime import datetime
from src.domain.vision_models import Frame
from src.vision.exceptions import CameraOpenError, FrameReadError, CameraDisconnectedError

class CameraSource:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._cap = None
        self._frame_count = 0
        self._is_file = not camera_id.isdigit()

    def open(self) -> None:
        """Abre la cámara física o el archivo de video."""
        # Convertir a entero si representa un índice de cámara física
        source = int(self.camera_id) if self.camera_id.isdigit() else self.camera_id
        
        try:
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                raise CameraOpenError(f"No se pudo abrir el origen de cámara/video: {self.camera_id}")
            self._frame_count = 0
        except Exception as e:
            if not isinstance(e, CameraOpenError):
                raise CameraOpenError(f"Error inesperado al abrir la cámara {self.camera_id}: {e}")
            raise e

    def read(self) -> Frame:
        """Lee el siguiente frame de la cámara o video."""
        if self._cap is None or not self._cap.isOpened():
            raise CameraDisconnectedError(f"La cámara {self.camera_id} no está abierta o se desconectó.")

        ret, frame = self._cap.read()
        if not ret:
            if self._is_file:
                # Si es un archivo de video, reiniciar el stream al inicio para pruebas continuas
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    raise FrameReadError(f"Fin del archivo de video y no se pudo reiniciar: {self.camera_id}")
            else:
                raise CameraDisconnectedError(f"Fallo al leer frame de la cámara física {self.camera_id}.")

        self._frame_count += 1
        return Frame(
            frame_id=self._frame_count,
            camera_id=self.camera_id,
            timestamp=datetime.now(),
            image=frame
        )

    def close(self) -> None:
        """Cierra el flujo de captura de video."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        """Devuelve si la captura de video está abierta."""
        return self._cap is not None and self._cap.isOpened()
