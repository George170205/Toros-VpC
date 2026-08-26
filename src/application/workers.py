import queue
import threading
import traceback
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional
from src.domain.enums import TrackState, DecisionType
from src.domain.vision_models import FaceSample
from src.domain.identity_models import StoredEmbedding, IdentityEvidence
from src.domain.decision_models import DecisionResult, CandidateEvidence
from src.biometrics import ArcFaceEmbeddingGenerator, EvidenceAggregator, VectorSearchService, SecondaryVerificationService, HierarchicalDecisionEngine
from src.biometrics.exceptions import UnstableEvidenceError, BiometricPipelineError
from src.application.queues import TaskQueue
from src.utils.observability import logger, metrics, CorrelationContext

@dataclass(frozen=True, slots=True)
class RecognitionTask:
    task_id: str
    session_id: str
    track_id: int
    camera_id: str
    samples: list[FaceSample]
    created_at: datetime
    session_revision: int

@dataclass(frozen=True, slots=True)
class RecognitionResult:
    task_id: str
    session_id: str
    track_id: int
    camera_id: str
    evidence: Optional[IdentityEvidence]
    success: bool
    error_code: Optional[str]
    completed_at: datetime
    session_revision: int

@dataclass(frozen=True, slots=True)
class PersistenceCommand:
    command_id: str
    session_id: str
    track_id: int
    camera_id: str
    decision: DecisionResult
    evidence: Optional[IdentityEvidence]
    timestamp: datetime
    session_revision: int


class EmbeddingWorker(threading.Thread):
    """Worker dedicado para inferencia en GPU/CPU de ArcFace y agregación de evidencia."""
    def __init__(
        self,
        model_path: str,
        task_queue: TaskQueue[RecognitionTask],
        result_queue: TaskQueue[RecognitionResult],
        outlier_thresh: float = 0.72,
        min_consistency: float = 0.72
    ):
        super().__init__(daemon=True, name="EmbeddingWorker")
        self.task_queue = task_queue
        self.result_queue = result_queue
        
        # El generator se instancia dentro de la inicialización del worker
        self.generator = ArcFaceEmbeddingGenerator(model_path)
        self.aggregator = EvidenceAggregator(outlier_thresh, min_consistency)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("EmbeddingWorker iniciado.")
        while not self._stop_event.is_set():
            try:
                # Leer tarea con un timeout corto para revisar el stop_event periódicamente
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                ctx = CorrelationContext(
                    camera_id=task.camera_id,
                    session_id=task.session_id,
                    track_id=task.track_id,
                    task_id=task.task_id
                )
                log = logger.with_context(ctx)
                log.debug("recognition.task_started", num_samples=len(task.samples))
                
                t_start = datetime.now()
                success = False
                evidence = None
                error_code = None

                try:
                    # 1. Generar embeddings (inferencia ArcFace)
                    embeddings = self.generator.generate_batch(task.samples)
                    
                    # 2. Agregar evidencia (outliers, consistencia, fusión)
                    evidence = self.aggregator.aggregate(
                        session_id=task.session_id,
                        track_id=task.track_id,
                        camera_id=task.camera_id,
                        embeddings=embeddings
                    )
                    success = True
                    
                    t_cost = (datetime.now() - t_start).total_seconds() * 1000
                    metrics.observe("embedding_latency_ms", t_cost)
                    log.debug("recognition.task_completed", latency_ms=t_cost, consistency=evidence.embedding_consistency)

                except UnstableEvidenceError as ue:
                    error_code = "UNSTABLE_EVIDENCE"
                    log.warning("recognition.task_unstable", error=str(ue))
                except Exception as e:
                    error_code = "GENERATION_ERROR"
                    log.error("recognition.task_failed", error=str(e))
                
                # Push a la cola de resultados
                result = RecognitionResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    track_id=task.track_id,
                    camera_id=task.camera_id,
                    evidence=evidence,
                    success=success,
                    error_code=error_code,
                    completed_at=datetime.now(),
                    session_revision=task.session_revision
                )
                self.result_queue.put(result)
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error("recognition.worker_loop_error", error=str(e))


class IdentityPipelineWorker(threading.Thread):
    """Worker de CPU que consume resultados biométricos, consulta el VectorIndex y ejecuta el Decision Engine."""
    def __init__(
        self,
        result_queue: TaskQueue[RecognitionResult],
        persistence_queue: TaskQueue[PersistenceCommand],
        search_service: VectorSearchService,
        verification_service: SecondaryVerificationService,
        decision_engine: HierarchicalDecisionEngine,
        load_profile_func: Callable[[str], list[StoredEmbedding]],
        session_state_callback: Callable[[str, int, TrackState], None]
    ):
        super().__init__(daemon=True, name="IdentityPipelineWorker")
        self.result_queue = result_queue
        self.persistence_queue = persistence_queue
        
        self.search_service = search_service
        self.verification_service = verification_service
        self.decision_engine = decision_engine
        self.load_profile_func = load_profile_func
        self.session_state_callback = session_state_callback
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("IdentityPipelineWorker iniciado.")
        while not self._stop_event.is_set():
            try:
                try:
                    res = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                ctx = CorrelationContext(
                    camera_id=res.camera_id,
                    session_id=res.session_id,
                    track_id=res.track_id,
                    task_id=res.task_id,
                    evidence_id=res.evidence.evidence_id if res.evidence else None
                )
                log = logger.with_context(ctx)

                # Si no tuvo éxito o la evidencia es inestable, devolvemos a COLLECTING
                if not res.success:
                    if res.error_code == "UNSTABLE_EVIDENCE":
                        self.session_state_callback(res.session_id, res.session_revision, TrackState.COLLECTING)
                        log.debug("decision.deferred_unstable")
                    else:
                        self.session_state_callback(res.session_id, res.session_revision, TrackState.COLLECTING)
                        log.error("decision.error_resettocollecting", error_code=res.error_code)
                    
                    self.result_queue.task_done()
                    continue

                evidence = res.evidence
                t_start = datetime.now()

                # 1. Búsqueda Vectorial de Nivel 1 (FAISS/NumPy)
                matches = self.search_service.search(evidence.aggregated_embedding, top_k=5)
                
                # 2. Verificación Secundaria de Nivel 2
                candidates = self.verification_service.verify_candidates(
                    evidence=evidence,
                    matches=matches,
                    load_profile_embeddings=self.load_profile_func
                )

                # 3. Decision Engine Jerárquico
                decision = self.decision_engine.decide(
                    session_id=res.session_id,
                    track_id=res.track_id,
                    camera_id=res.camera_id,
                    evidence=evidence,
                    candidates=candidates
                )

                t_cost = (datetime.now() - t_start).total_seconds() * 1000
                metrics.observe("decision_latency_ms", t_cost)
                log.info(
                    "decision.engine_executed",
                    decision=decision.decision.name,
                    confidence=decision.confidence,
                    person_id=decision.person_id,
                    reason=decision.reason_codes[0] if decision.reason_codes else ""
                )

                # 4. Encolar comando de persistencia para el PersistenceWorker
                cmd = PersistenceCommand(
                    command_id=f"cmd_{uuid.uuid4().hex[:12]}",
                    session_id=res.session_id,
                    track_id=res.track_id,
                    camera_id=res.camera_id,
                    decision=decision,
                    evidence=evidence,
                    timestamp=datetime.now(),
                    session_revision=res.session_revision
                )
                self.persistence_queue.put(cmd)
                self.result_queue.task_done()

            except Exception as e:
                logger.error("decision.pipeline_worker_loop_error", error=str(e))
