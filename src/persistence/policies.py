import uuid
import numpy as np
from datetime import datetime
from src.domain.enums import EmbeddingType
from src.domain.decision_models import DecisionResult, CandidateEvidence, IdentityUpdateDecision, NewIdentityDecision
from src.domain.identity_models import StoredEmbedding, IdentityEvidence
from src.utils.config import SystemConfiguration

class NewIdentityPolicy:
    """Política que decide si un resultado UNKNOWN justifica la creación de una nueva identidad."""
    def __init__(self, config: SystemConfiguration):
        self.config = config

    def evaluate(
        self,
        evidence: IdentityEvidence,
        decision: DecisionResult,
        candidates: list[CandidateEvidence]
    ) -> NewIdentityDecision:
        reasons = []
        
        # Criterio 1: Calidad mínima de la evidencia actual
        if evidence.average_quality < self.config.min_quality:
            reasons.append("REJECT_LOW_QUALITY_EVIDENCE")
            
        # Criterio 2: Consistencia interna del track actual
        if evidence.embedding_consistency < self.config.min_consistency:
            reasons.append("REJECT_INCONSISTENT_TRACK")

        # Criterio 3: Descarte de candidatos existentes
        # Si el mejor candidato tiene similitud alta (por ejemplo, > 0.60),
        # no creamos la identidad porque podría ser un falso UNKNOWN por bajo threshold
        if candidates and len(candidates) > 0:
            best_cand = candidates[0]
            if best_cand.verification_score > (self.config.min_verification_score - 0.08):
                reasons.append("REJECT_PLAUSIBLE_EXISTING_CANDIDATE")

        should_create = len(reasons) == 0
        if should_create:
            reasons.append("CREATE_NEW_IDENTITY_APPROVED")

        # Extraer los embeddings individuales para guardarlos como secundarios iniciales
        sec_vectors = [e.embedding for e in evidence.embeddings[:self.config.max_secondary_embeddings]]

        return NewIdentityDecision(
            should_create=should_create,
            primary_embedding=evidence.aggregated_embedding if should_create else None,
            secondary_embeddings=sec_vectors if should_create else [],
            quality_score=evidence.average_quality,
            evidence_id=evidence.evidence_id,
            reason_codes=reasons
        )


class IdentityUpdatePolicy:
    """Política que decide si un MATCH biométrico confirmado debe actualizar el perfil (PRIMARY o SECONDARY)."""
    def __init__(self, config: SystemConfiguration):
        self.config = config

    def evaluate(
        self,
        evidence: IdentityEvidence,
        decision: DecisionResult,
        primary: StoredEmbedding,
        secondaries: list[StoredEmbedding]
    ) -> IdentityUpdateDecision:
        reasons = []
        
        # Condición inicial: debe ser un MATCH confirmado de alta confianza
        if decision.confidence < self.config.min_verification_score:
            return IdentityUpdateDecision(
                should_update=False,
                update_primary=False,
                add_secondary=False,
                replace_secondary_id=None,
                confidence=decision.confidence,
                reason_codes=["NO_UPDATE_LOW_CONFIDENCE"]
            )

        # 1. Evaluar actualización suave del PRIMARY
        # Exigimos confianza muy alta y calidad superior al promedio registrado originalmente
        should_update_primary = False
        if (decision.confidence >= (self.config.min_verification_score + 0.10) and 
            evidence.average_quality > primary.quality_score and 
            evidence.embedding_consistency >= self.config.min_consistency):
            should_update_primary = True
            reasons.append("UPDATE_PRIMARY_APPROVED")

        # 2. Evaluar actualización/adición de SECONDARY por calidad o diversidad
        should_add_secondary = False
        replace_secondary_id = None
        
        # Tomar la mejor muestra individual de la sesión actual
        best_sample_idx = int(np.argmax([e.quality_score for e in evidence.embeddings]))
        best_sample_evidence = evidence.embeddings[best_sample_idx]
        best_sample_vector = best_sample_evidence.embedding
        best_sample_quality = best_sample_evidence.quality_score

        # Calcular similitud máxima con los secundarios existentes
        max_sim_to_secondaries = 0.0
        for sec in secondaries:
            sim = float(np.dot(best_sample_vector, sec.vector))
            if sim > max_sim_to_secondaries:
                max_sim_to_secondaries = sim

        # Si el nuevo embedding es sumamente similar a un secundario existente (ej. > 0.94)
        # pero tiene mejor calidad, marcamos para reemplazarlo (redundancia de pose pero mejor calidad)
        redundant_id = None
        if max_sim_to_secondaries > 0.94:
            for sec in secondaries:
                sim = float(np.dot(best_sample_vector, sec.vector))
                if sim > 0.94 and best_sample_quality > sec.quality_score:
                    redundant_id = sec.embedding_id
                    break

        if redundant_id is not None:
            should_add_secondary = True
            replace_secondary_id = redundant_id
            reasons.append("REPLACE_REDUNDANT_SECONDARY_FOR_QUALITY")
        # Si no es redundante (similitud < 0.88, indica ángulo/iluminación diversa)
        elif max_sim_to_secondaries < 0.88 and best_sample_quality >= self.config.min_quality:
            # Hay espacio disponible
            if len(secondaries) < self.config.max_secondary_embeddings:
                should_add_secondary = True
                reasons.append("ADD_SECONDARY_FOR_DIVERSITY")
            else:
                # Reemplazar el secundario de menor calidad si la nueva muestra tiene mejor calidad
                worst_sec = min(secondaries, key=lambda x: x.quality_score)
                if best_sample_quality > (worst_sec.quality_score + 0.05):
                    should_add_secondary = True
                    replace_secondary_id = worst_sec.embedding_id
                    reasons.append("REPLACE_WEAKEST_SECONDARY_FOR_DIVERSITY")

        should_update = should_update_primary or should_add_secondary
        if not should_update:
            reasons.append("NO_UPDATE_REDUNDANT_OR_LOW_QUALITY")

        return IdentityUpdateDecision(
            should_update=should_update,
            update_primary=should_update_primary,
            add_secondary=should_add_secondary,
            replace_secondary_id=replace_secondary_id,
            confidence=decision.confidence,
            reason_codes=reasons
        )

    def calculate_updated_primary(self, old_vector: np.ndarray, current_vector: np.ndarray) -> np.ndarray:
        """Calcula el vector principal actualizado mediante decaimiento exponencial P_new = norm(L * P_old + (1 - L) * E_curr)."""
        l = self.config.primary_update_lambda
        new_vec = l * old_vector + (1.0 - l) * current_vector
        
        # Normalizar L2
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm
        return new_vec
