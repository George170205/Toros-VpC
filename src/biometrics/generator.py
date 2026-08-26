import os
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from src.domain.vision_models import FaceSample
from src.domain.identity_models import EmbeddingEvidence
from src.biometrics.exceptions import EmbeddingGenerationError, InvalidEmbeddingError

class ArcFaceEmbeddingGenerator:
    """Generador de embeddings faciales de 512 dimensiones usando el modelo ArcFace ONNX."""
    def __init__(self, model_file: str):
        self.model_file = model_file
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"No se encontró el archivo del modelo ArcFace: {self.model_file}")

        # Configurar Execution Providers (GPU si está disponible)
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider'] + providers

        self.session = ort.InferenceSession(self.model_file, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocesa el recorte de cara al tamaño 112x112 y formato de entrada de ArcFace."""
        # Redimensionar a 112x112
        resized = cv2.resize(face_image, (112, 112))
        
        # ArcFace espera entrada RGB en rango [-1, 1]
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized = resized.astype(np.float32)
        resized = (resized - 127.5) / 127.5  # Normalizar a [-1, 1]
        
        # Transponer de HWC a CHW
        blob = np.transpose(resized, (2, 0, 1))
        return blob

    def generate(self, sample: FaceSample) -> EmbeddingEvidence:
        """Genera un embedding individual normalizado L2."""
        try:
            blob = self._preprocess(sample.face_image)
            # Agregar dimensión de batch: (1, 3, 112, 112)
            input_blob = np.expand_dims(blob, axis=0)
            
            outputs = self.session.run([self.output_name], {self.input_name: input_blob})
            embedding = outputs[0][0]
            
            # Normalización L2
            norm = np.linalg.norm(embedding)
            if norm == 0.0:
                raise InvalidEmbeddingError("El modelo generó un vector nulo (norma 0).")
            embedding = embedding / norm

            return EmbeddingEvidence(
                sample_id=sample.sample_id,
                track_id=sample.track_id,
                embedding=embedding,
                quality_score=sample.quality.overall_score,
                pose=sample.quality.pose,
                weight=0.0,  # Se calcula en la agregación
                timestamp=sample.timestamp
            )
        except Exception as e:
            if isinstance(e, InvalidEmbeddingError):
                raise e
            raise EmbeddingGenerationError(f"Error al generar embedding para muestra {sample.sample_id}: {e}") from e

    def generate_batch(self, samples: list[FaceSample]) -> list[EmbeddingEvidence]:
        """Genera embeddings procesando cada muestra secuencialmente para ajustarse al batch_size=1 fijo de ArcFace, evitando el overhead de recompilación en ONNX."""
        if not samples:
            return []
        try:
            evidences = []
            for sample in samples:
                evidence = self.generate(sample)
                evidences.append(evidence)
            return evidences
        except Exception as e:
            if isinstance(e, InvalidEmbeddingError):
                raise e
            raise EmbeddingGenerationError(f"Error al procesar lote de embeddings faciales: {e}") from e
