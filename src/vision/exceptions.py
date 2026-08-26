class VisionPipelineError(Exception):
    """Clase base para todos los errores del pipeline de visión."""
    pass

class CameraError(VisionPipelineError):
    """Clase base para errores relacionados con la cámara."""
    pass

class CameraOpenError(CameraError):
    """Error al intentar abrir la cámara o archivo de video."""
    pass

class FrameReadError(CameraError):
    """Error al intentar leer un frame."""
    pass

class CameraDisconnectedError(CameraError):
    """Se disparó cuando la cámara se desconecta repentinamente."""
    pass

class InvalidFrameError(VisionPipelineError):
    """El frame obtenido no es válido (ej. vacío)."""
    pass

class DetectionError(VisionPipelineError):
    """Error durante la inferencia de detección facial."""
    pass

class TrackingError(VisionPipelineError):
    """Error durante la actualización del tracker."""
    pass

class QualityAnalysisError(VisionPipelineError):
    """Error durante el análisis de calidad."""
    pass

class SampleCollectionError(VisionPipelineError):
    """Error en la recolección de muestras."""
    pass
