import uuid
from datetime import datetime
from src.domain.enums import DecisionType
from src.domain.decision_models import CandidateEvidence, DecisionResult
from src.domain.identity_models import IdentityEvidence
from src.utils.config import SystemConfiguration

class HierarchicalDecisionEngine:
    """Motor de decisión jerárquico por barreras que evalúa la fuerza y coherencia de los candidatos biométricos."""
    def __init__(self, config: SystemConfiguration):
        self.config = config

    def decide(
        self,
        session_id: str,
        track_id: int,
        camera_id: str,
        evidence: IdentityEvidence,
        candidates: list[CandidateEvidence]
    ) -> DecisionResult:
        timestamp = datetime.now()
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        
        # 1. Gate 1 - Validación de Evidencia Mínima
        if not evidence.embeddings or len(evidence.embeddings) == 0:
            return DecisionResult(
                decision_id=decision_id,
                session_id=session_id,
                track_id=track_id,
                camera_id=camera_id,
                decision=DecisionType.AMBIGUOUS,
                person_id=None,
                confidence=0.0,
                best_candidate_score=None,
                second_candidate_score=None,
                margin=None,
                quality_score=0.0,
                consistency_score=0.0,
                positive_match_ratio=0.0,
                temporal_confirmation=1.0,
                reason_codes=["AMBIGUOUS_NO_SAMPLES"],
                timestamp=timestamp
            )

        avg_quality = evidence.average_quality
        internal_consistency = evidence.embedding_consistency

        # Validar calidad mínima de las muestras para tomar una decisión final
        if avg_quality < self.config.min_quality:
            return DecisionResult(
                decision_id=decision_id,
                session_id=session_id,
                track_id=track_id,
                camera_id=camera_id,
                decision=DecisionType.AMBIGUOUS,
                person_id=None,
                confidence=0.0,
                best_candidate_score=None,
                second_candidate_score=None,
                margin=None,
                quality_score=avg_quality,
                consistency_score=internal_consistency,
                positive_match_ratio=0.0,
                temporal_confirmation=1.0,
                reason_codes=["AMBIGUOUS_LOW_QUALITY"],
                timestamp=timestamp
            )

        # 2. Gate 2 - Fuerza del Mejor Candidato
        if not candidates:
            # No hay candidatos vectoriales en absoluto
            return DecisionResult(
                decision_id=decision_id,
                session_id=session_id,
                track_id=track_id,
                camera_id=camera_id,
                decision=DecisionType.UNKNOWN,
                person_id=None,
                confidence=0.0,
                best_candidate_score=None,
                second_candidate_score=None,
                margin=None,
                quality_score=avg_quality,
                consistency_score=internal_consistency,
                positive_match_ratio=0.0,
                temporal_confirmation=1.0,
                reason_codes=["UNKNOWN_NO_CANDIDATES"],
                timestamp=timestamp
            )

        best_cand = candidates[0]
        best_score = best_cand.verification_score

        # Si el mejor candidato es demasiado débil, clasificar como UNKNOWN
        if best_score < self.config.min_verification_score:
            return DecisionResult(
                decision_id=decision_id,
                session_id=session_id,
                track_id=track_id,
                camera_id=camera_id,
                decision=DecisionType.UNKNOWN,
                person_id=None,
                confidence=best_score,
                best_candidate_score=best_score,
                second_candidate_score=candidates[1].verification_score if len(candidates) > 1 else None,
                margin=None,
                quality_score=avg_quality,
                consistency_score=internal_consistency,
                positive_match_ratio=best_cand.positive_match_ratio,
                temporal_confirmation=1.0,
                reason_codes=["UNKNOWN_NO_STRONG_CANDIDATE"],
                timestamp=timestamp
            )

        # 3. Gate 3 - Margen contra el Segundo Candidato
        second_score = 0.0
        if len(candidates) > 1:
            second_score = candidates[1].verification_score
            margin = best_score - second_score
        else:
            margin = best_score  # Si es el único candidato, el margen es alto

        if len(candidates) > 1 and margin < self.config.min_margin:
            return DecisionResult(
                decision_id=decision_id,
                session_id=session_id,
                track_id=track_id,
                camera_id=camera_id,
                decision=DecisionType.AMBIGUOUS,
                person_id=None,
                confidence=best_score,
                best_candidate_score=best_score,
                second_candidate_score=second_score,
                margin=margin,
                quality_score=avg_quality,
                consistency_score=internal_consistency,
                positive_match_ratio=best_cand.positive_match_ratio,
                temporal_confirmation=1.0,
                reason_codes=["AMBIGUOUS_LOW_MARGIN"],
                timestamp=timestamp
            )

        # 4. Gate 4 - Consistencia del Candidato contra su Perfil
        if best_cand.consistency_score < self.config.min_consistency or best_cand.positive_match_ratio < 0.60:
            return DecisionResult(
                decision_id=decision_id,
                session_id=session_id,
                track_id=track_id,
                camera_id=camera_id,
                decision=DecisionType.AMBIGUOUS,
                person_id=None,
                confidence=best_score,
                best_candidate_score=best_score,
                second_candidate_score=second_score,
                margin=margin,
                quality_score=avg_quality,
                consistency_score=best_cand.consistency_score,
                positive_match_ratio=best_cand.positive_match_ratio,
                temporal_confirmation=1.0,
                reason_codes=["AMBIGUOUS_INCONSISTENT_EVIDENCE"],
                timestamp=timestamp
            )

        # 5. Gate 5 - Calidad de la evidencia actual específica para MATCH
        # Exigimos que al menos el mejor frame actual tenga una similitud muy sólida
        if best_cand.best_similarity < 0.70:
            return DecisionResult(
                decision_id=decision_id,
                session_id=session_id,
                track_id=track_id,
                camera_id=camera_id,
                decision=DecisionType.AMBIGUOUS,
                person_id=None,
                confidence=best_score,
                best_candidate_score=best_score,
                second_candidate_score=second_score,
                margin=margin,
                quality_score=avg_quality,
                consistency_score=best_cand.consistency_score,
                positive_match_ratio=best_cand.positive_match_ratio,
                temporal_confirmation=1.0,
                reason_codes=["AMBIGUOUS_LOW_MATCH_QUALITY"],
                timestamp=timestamp
            )

        # 6. Todo verificado con éxito: Confirmar MATCH
        return DecisionResult(
            decision_id=decision_id,
            session_id=session_id,
            track_id=track_id,
            camera_id=camera_id,
            decision=DecisionType.MATCH,
            person_id=best_cand.person_id,
            confidence=best_score,
            best_candidate_score=best_score,
            second_candidate_score=second_score if len(candidates) > 1 else None,
            margin=margin,
            quality_score=avg_quality,
            consistency_score=best_cand.consistency_score,
            positive_match_ratio=best_cand.positive_match_ratio,
            temporal_confirmation=1.0,  # Temporalmente en 1.0 por defecto
            reason_codes=["MATCH_HIGH_CONFIDENCE"],
            timestamp=timestamp
        )
