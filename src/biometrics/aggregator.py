import uuid
import numpy as np
from datetime import datetime
from src.domain.enums import TrackState
from src.domain.vision_models import FaceSample
from src.domain.identity_models import EmbeddingEvidence, IdentityEvidence
from src.biometrics.exceptions import EvidenceAggregationError, InsufficientEvidenceError, UnstableEvidenceError

class EvidenceAggregator:
    """Agregador de evidencia biométrica que filtra outliers y fusiona embeddings mediante pesos de calidad."""
    def __init__(self, outlier_threshold: float = 0.72, min_consistency: float = 0.72):
        self.outlier_threshold = outlier_threshold
        self.min_consistency = min_consistency

    def aggregate(
        self,
        session_id: str,
        track_id: int,
        camera_id: str,
        embeddings: list[EmbeddingEvidence]
    ) -> IdentityEvidence:
        if not embeddings:
            raise InsufficientEvidenceError("No hay embeddings disponibles para agregar.")

        try:
            n_samples = len(embeddings)
            stable_embeddings = list(embeddings)
            outlier_ids = set()

            # 1. Detección de Outliers (solo si hay al menos 3 muestras)
            if n_samples >= 3:
                # Construir matriz de similitud de coseno
                sim_matrix = np.zeros((n_samples, n_samples))
                for i in range(n_samples):
                    for j in range(i, n_samples):
                        sim = float(np.dot(embeddings[i].embedding, embeddings[j].embedding))
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
                
                # Calcular similitud media de cada muestra respecto a las demás
                # (excluyendo la diagonal, es decir, sim = 1 consigo misma)
                mean_similarities = []
                for i in range(n_samples):
                    other_sims = [sim_matrix[i, j] for j in range(n_samples) if i != j]
                    mean_similarities.append(np.mean(other_sims))
                
                # Filtrar outliers
                stable_embeddings = []
                for idx, mean_sim in enumerate(mean_similarities):
                    if mean_sim < self.outlier_threshold:
                        outlier_ids.add(embeddings[idx].sample_id)
                    else:
                        stable_embeddings.append(embeddings[idx])
                        
                # Si nos quedamos sin muestras estables, restauramos la mejor para no fallar
                if not stable_embeddings:
                    best_idx = np.argmax(mean_similarities)
                    stable_embeddings = [embeddings[best_idx]]
                    outlier_ids.remove(embeddings[best_idx].sample_id)

            n_stable = len(stable_embeddings)
            total_quality = sum(e.quality_score for e in stable_embeddings)
            
            # 2. Calcular pesos basados en calidad
            weights = []
            if total_quality > 0:
                weights = [e.quality_score / total_quality for e in stable_embeddings]
            else:
                weights = [1.0 / n_stable] * n_stable

            # Actualizar el atributo 'weight' en cada EmbeddingEvidence
            updated_embeddings = []
            for idx, e in enumerate(stable_embeddings):
                updated_embeddings.append(EmbeddingEvidence(
                    sample_id=e.sample_id,
                    track_id=e.track_id,
                    embedding=e.embedding,
                    quality_score=e.quality_score,
                    pose=e.pose,
                    weight=weights[idx],
                    timestamp=e.timestamp
                ))

            # 3. Agregación ponderada y normalización L2 del vector resultante
            aggregated_vector = np.zeros_like(embeddings[0].embedding)
            for idx, e in enumerate(updated_embeddings):
                aggregated_vector += e.weight * e.embedding
                
            norm = np.linalg.norm(aggregated_vector)
            if norm > 0:
                aggregated_vector = aggregated_vector / norm

            # 4. Consistencia interna (similitud media entre las muestras estables finales)
            if n_stable > 1:
                stable_sims = []
                for i in range(n_stable):
                    for j in range(i + 1, n_stable):
                        stable_sims.append(float(np.dot(updated_embeddings[i].embedding, updated_embeddings[j].embedding)))
                consistency = float(np.mean(stable_sims))
            else:
                consistency = 1.0  # Consistencia perfecta con 1 muestra

            # Validar consistencia mínima
            if consistency < self.min_consistency and n_stable > 1:
                # Lanzar advertencia controlada para indicar que la evidencia es inestable
                raise UnstableEvidenceError(
                    f"Consistencia interna insuficiente ({consistency:.3f} < {self.min_consistency:.3f})"
                )

            avg_quality = float(np.mean([e.quality_score for e in updated_embeddings]))

            return IdentityEvidence(
                evidence_id=f"evid_{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                track_id=track_id,
                camera_id=camera_id,
                embeddings=updated_embeddings,
                aggregated_embedding=aggregated_vector,
                total_samples=n_samples,
                valid_samples=n_stable,
                rejected_samples=len(outlier_ids),
                average_quality=avg_quality,
                embedding_consistency=consistency,
                aggregation_weights=weights,
                created_at=datetime.now()
            )
        except Exception as e:
            if isinstance(e, (InsufficientEvidenceError, UnstableEvidenceError)):
                raise e
            raise EvidenceAggregationError(f"Error inesperado al agregar evidencia biométrica: {e}") from e
