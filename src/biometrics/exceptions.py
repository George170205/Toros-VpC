class BiometricPipelineError(Exception):
    """Clase base para todos los errores del pipeline biométrico."""
    pass

class EmbeddingGenerationError(BiometricPipelineError):
    """Error al generar embeddings usando ArcFace."""
    pass

class InvalidEmbeddingError(BiometricPipelineError):
    """El embedding generado o recibido es inválido (ej. dimensiones incorrectas o valores no numéricos)."""
    pass

class InsufficientEvidenceError(BiometricPipelineError):
    """No hay suficientes muestras en la sesión para realizar el reconocimiento o agregación."""
    pass

class EvidenceAggregationError(BiometricPipelineError):
    """Error al fusionar o agregar múltiples embeddings."""
    pass

class UnstableEvidenceError(BiometricPipelineError):
    """La evidencia es demasiado inconsistente internamente para ser confiable (alto desvío estándar/outliers)."""
    pass
