import numpy as np
from typing import Callable
from src.domain.decision_models import CandidateMatch, SimilarityMatrix, CandidateEvidence
from src.domain.identity_models import StoredEmbedding, IdentityEvidence
from src.biometrics.exceptions import BiometricPipelineError

class SecondaryVerificationService:
    """Servicio de Verificación Secundaria que construye matrices de similitud y extrae métricas biométricas."""
    def __init__(
        self,
        positive_threshold: float = 0.70,
        weights: dict[str, float] = None
    ):
        self.positive_threshold = positive_threshold
        
        # Pesos por defecto para el score compuesto de verificación
        self.weights = weights or {
            "primary_similarity": 0.30,
            "mean_top_similarity": 0.25,
            "median_top_similarity": 0.15,
            "consistency": 0.10,
            "positive_ratio": 0.15,
            "sample_quality": 0.05
        }
        
        # Validar que los pesos sumen aproximadamente 1
        total_w = sum(self.weights.values())
        if not (0.99 <= total_w <= 1.01):
            raise ValueError(f"Los pesos del score de verificación deben sumar 1.0 (actual: {total_w})")

    def verify_candidates(
        self,
        evidence: IdentityEvidence,
        matches: list[CandidateMatch],
        load_profile_embeddings: Callable[[str], list[StoredEmbedding]]
    ) -> list[CandidateEvidence]:
        """Calcula métricas detalladas de verificación para cada candidato Top-K."""
        if not matches or not evidence.embeddings:
            return []

        candidates_evidence = []
        try:
            # Embeddings actuales de la sesión (M muestras)
            current_embs = [e.embedding for e in evidence.embeddings]
            current_ids = [e.sample_id for e in evidence.embeddings]
            M = len(current_embs)

            for match in matches:
                # 1. Cargar embeddings históricos del perfil (PRIMARY + SECONDARY) (N muestras)
                profile_embs = load_profile_embeddings(match.person_id)
                if not profile_embs:
                    continue

                stored_vectors = [p.vector for p in profile_embs]
                stored_ids = [p.embedding_id for p in profile_embs]
                N = len(stored_vectors)

                # 2. Construir Matriz de Similitud M x N
                sim_values = np.zeros((M, N))
                for i in range(M):
                    for j in range(N):
                        sim_values[i, j] = float(np.dot(current_embs[i], stored_vectors[j]))

                sim_matrix = SimilarityMatrix(
                    values=sim_values,
                    current_embedding_ids=current_ids,
                    candidate_embedding_ids=stored_ids
                )

                # 3. Extraer métricas biométricas
                best_similarity = float(np.max(sim_values))
                
                # Mejor correspondencia histórica para cada muestra actual
                best_per_sample = [float(np.max(sim_values[i, :])) for i in range(M)]
                
                mean_top_similarity = float(np.mean(best_per_sample))
                median_top_similarity = float(np.median(best_per_sample))
                
                # Consistencia (desviación estándar normalizada)
                if M > 1:
                    std_dev = np.std(best_per_sample)
                    consistency_score = float(max(0.0, 1.0 - std_dev * 3.0))
                else:
                    consistency_score = 1.0

                # Ratio de correspondencias positivas
                positives = sum(1 for val in best_per_sample if val >= self.positive_threshold)
                positive_match_ratio = float(positives / M)

                # Similitud con el PRIMARY
                # Localizar el embedding PRIMARY en los cargados
                primary_idx = -1
                for idx, p in enumerate(profile_embs):
                    if p.embedding_type.name == "PRIMARY":
                        primary_idx = idx
                        break
                
                # Si encontramos el primary, calculamos la similitud directa con el agregado
                if primary_idx != -1:
                    primary_similarity = float(np.dot(evidence.aggregated_embedding, stored_vectors[primary_idx]))
                else:
                    primary_similarity = match.primary_similarity

                # 4. Calcular Verification Score Compuesto
                v_score = (
                    self.weights["primary_similarity"] * primary_similarity +
                    self.weights["mean_top_similarity"] * mean_top_similarity +
                    self.weights["median_top_similarity"] * median_top_similarity +
                    self.weights["consistency"] * consistency_score +
                    self.weights["positive_ratio"] * positive_match_ratio +
                    self.weights["sample_quality"] * evidence.average_quality
                )

                candidates_evidence.append(CandidateEvidence(
                    person_id=match.person_id,
                    rank=match.rank,
                    primary_similarity=primary_similarity,
                    similarity_matrix=sim_matrix,
                    best_similarity=best_similarity,
                    best_per_sample=best_per_sample,
                    mean_top_similarity=mean_top_similarity,
                    median_top_similarity=median_top_similarity,
                    consistency_score=consistency_score,
                    positive_match_ratio=positive_match_ratio,
                    average_sample_quality=evidence.average_quality,
                    verification_score=float(v_score)
                ))

            # Ordenar candidatos por el verification_score descendente
            candidates_evidence.sort(key=lambda x: x.verification_score, reverse=True)
            return candidates_evidence

        except Exception as e:
            raise BiometricPipelineError(f"Error en verificación secundaria para candidato: {e}") from e
