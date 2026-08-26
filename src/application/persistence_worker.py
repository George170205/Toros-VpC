import queue
import threading
from datetime import datetime
from typing import Callable, Optional
from src.domain.enums import TrackState, DecisionType
from src.domain.persistence_models import VisitEvent
from src.application.workers import PersistenceCommand
from src.application.queues import TaskQueue
from src.persistence import UnitOfWork, DatabaseService, IdentityService, VisitService, NewIdentityPolicy, IdentityUpdatePolicy
from src.biometrics import VectorSearchService
from src.utils.observability import logger, metrics, CorrelationContext

class PersistenceWorker(threading.Thread):
    """Worker secuencial de base de datos que ejecuta escrituras transaccionales sobre SQLite e indexa cambios en FAISS."""
    def __init__(
        self,
        persistence_queue: TaskQueue[PersistenceCommand],
        search_service: VectorSearchService,
        db_service: DatabaseService,
        identity_service: IdentityService,
        visit_service: VisitService,
        new_identity_policy: NewIdentityPolicy,
        identity_update_policy: IdentityUpdatePolicy,
        session_state_callback: Callable[[str, int, TrackState, Optional[str]], None],
        live_event_callback: Callable[[dict], None]  # Envía notificaciones de ingresos a FastAPI WebSockets
    ):
        super().__init__(daemon=True, name="PersistenceWorker")
        self.persistence_queue = persistence_queue
        self.search_service = search_service
        self.db_service = db_service
        self.identity_service = identity_service
        self.visit_service = visit_service
        self.new_identity_policy = new_identity_policy
        self.identity_update_policy = identity_update_policy
        self.session_state_callback = session_state_callback
        self.live_event_callback = live_event_callback
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("PersistenceWorker iniciado.")
        while not self._stop_event.is_set():
            try:
                try:
                    cmd = self.persistence_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                ctx = CorrelationContext(
                    camera_id=cmd.camera_id,
                    session_id=cmd.session_id,
                    track_id=cmd.track_id,
                    decision_id=cmd.decision.decision_id
                )
                log = logger.with_context(ctx)
                log.debug("persistence.command_started")
                
                t_start = datetime.now()
                
                # Variables para sincronizar el índice vectorial después del commit
                sync_primary_emb = None
                sync_person_id = None
                
                final_person_code = None
                final_state = TrackState.FINISHED
 
                try:
                    # Iniciar Unidad de Trabajo (Transacción SQLite)
                    with UnitOfWork(self.db_service) as uow:
                        # Asegurar que la cámara existe en la base de datos (para evitar fallos de Foreign Key)
                        uow.conn.execute(
                            """
                            INSERT OR IGNORE INTO cameras (
                                camera_id, name, location, entry_zone, resolution_width, resolution_height, target_fps, active, created_at
                            ) VALUES (?, ?, ?, ?, 640, 480, 30.0, 1, ?)
                            """,
                            (
                                cmd.camera_id,
                                f"Cámara {cmd.camera_id}",
                                f"Acceso {cmd.camera_id}",
                                f"Zona-{cmd.camera_id}",
                                datetime.now().isoformat()
                            )
                        )

                        # 1. Persistir el registro de auditoría de la decisión
                        uow.decisions.save(cmd.decision, self.identity_service.config.configuration_id)
                        
                        # Guardar el log de la sesión
                        uow.session_logs.save_summary(
                            session_id=cmd.session_id,
                            track_id=cmd.track_id,
                            camera_id=cmd.camera_id,
                            first_seen=cmd.timestamp, # Placeholder, simplificado en worker
                            last_seen=datetime.now(),
                            total_frames=cmd.evidence.total_samples if cmd.evidence else 1,
                            valid_samples=cmd.evidence.valid_samples if cmd.evidence else 1,
                            rejected_samples=cmd.evidence.rejected_samples if cmd.evidence else 0,
                            final_state=cmd.decision.decision.name,
                            decision_id=cmd.decision.decision_id
                        )

                        # 2. Evaluar políticas según el tipo de decisión
                        if cmd.decision.decision == DecisionType.MATCH:
                            person_id = cmd.decision.person_id
                            person = uow.persons.get_by_id(person_id)
                            final_person_code = person.public_code if person else "IND_UNKNOWN"

                            # Registrar visita de forma idempotente
                            event = VisitEvent(
                                event_id=f"evt_{cmd.session_id}_{cmd.track_id}",
                                person_id=person_id,
                                session_id=cmd.session_id,
                                track_id=cmd.track_id,
                                camera_id=cmd.camera_id,
                                timestamp=cmd.decision.timestamp,
                                decision_id=cmd.decision.decision_id,
                                confidence=cmd.decision.confidence,
                                entry_zone=uow.cameras.get(cmd.camera_id).entry_zone if uow.cameras.get(cmd.camera_id) else None
                            )
                            self.visit_service.register(uow, event)

                            # Evaluar actualización del perfil biométrico
                            primary_emb = uow.embeddings.get_primary(person_id)
                            secondaries = uow.embeddings.get_secondary(person_id)
                            
                            update_decision = self.identity_update_policy.evaluate(
                                evidence=cmd.evidence,
                                decision=cmd.decision,
                                primary=primary_emb,
                                secondaries=secondaries
                            )

                            if update_decision.should_update:
                                primary_changed, new_emb_id = self.identity_service.update_identity(
                                    uow=uow,
                                    person_id=person_id,
                                    evidence=cmd.evidence,
                                    decision=cmd.decision,
                                    update=update_decision
                                )
                                if primary_changed and new_emb_id:
                                    # Recargar el embedding guardado para sincronizarlo
                                    sync_primary_emb = uow.embeddings.get_primary(person_id)
                                    sync_person_id = person_id

                        elif cmd.decision.decision == DecisionType.UNKNOWN:
                            # Evaluar si califica para crear una nueva identidad
                            # Buscar candidatos para descartar existentes en la política
                            creation_decision = self.new_identity_policy.evaluate(
                                evidence=cmd.evidence,
                                decision=cmd.decision,
                                candidates=[]
                            )

                            if creation_decision.should_create:
                                # Crear identidad transaccionalmente
                                person = self.identity_service.create_identity(
                                    uow=uow,
                                    evidence=cmd.evidence,
                                    decision=cmd.decision,
                                    creation=creation_decision
                                )
                                final_person_code = person.public_code
                                
                                # Registrar primera visita
                                event = VisitEvent(
                                    event_id=f"evt_{cmd.session_id}_{cmd.track_id}",
                                    person_id=person.person_id,
                                    session_id=cmd.session_id,
                                    track_id=cmd.track_id,
                                    camera_id=cmd.camera_id,
                                    timestamp=cmd.decision.timestamp,
                                    decision_id=cmd.decision.decision_id,
                                    confidence=cmd.decision.confidence,
                                    entry_zone=uow.cameras.get(cmd.camera_id).entry_zone if uow.cameras.get(cmd.camera_id) else None
                                )
                                self.visit_service.register(uow, event)

                                # Sincronizar nuevo PRIMARY
                                sync_primary_emb = uow.embeddings.get_primary(person.person_id)
                                sync_person_id = person.person_id
                            else:
                                final_person_code = "DESCARTE_NUEVO"
                                final_state = TrackState.UNKNOWN

                        else:  # AMBIGUOUS
                            final_person_code = "AMBIGUOUS"
                            final_state = TrackState.AMBIGUOUS

                    # --- FIN DE TRANSACCIÓN SQLite (UoW commitea automáticamente aquí) ---
                    
                    # 3. Sincronizar el Índice Vectorial (en memoria) si hubo cambios en PRIMARY
                    if sync_primary_emb and sync_person_id:
                        self.search_service.add_item(sync_primary_emb)
                        log.debug("persistence.index_synchronized", person_id=sync_person_id)

                    # 4. Actualizar estado de la sesión en el coordinador
                    self.session_state_callback(cmd.session_id, cmd.session_revision, final_state, final_person_code)

                    # 5. Notificar a FastAPI (WebSockets) para visualización en tiempo real
                    self.live_event_callback({
                        "session_id": cmd.session_id,
                        "track_id": cmd.track_id,
                        "camera_id": cmd.camera_id,
                        "decision": cmd.decision.decision.name,
                        "person_id": cmd.decision.person_id,
                        "public_code": final_person_code,
                        "confidence": cmd.decision.confidence,
                        "timestamp": cmd.decision.timestamp.isoformat(),
                        "reason": cmd.decision.reason_codes[0] if cmd.decision.reason_codes else ""
                    })

                    t_cost = (datetime.now() - t_start).total_seconds() * 1000
                    metrics.observe("db_write_latency_ms", t_cost)
                    log.debug("persistence.command_completed", latency_ms=t_cost)

                except Exception as e:
                    metrics.increment("transaction_failures")
                    log.error("persistence.transaction_failed", error=str(e))
                    # Retornar la sesión al estado anterior para reintentar o re-recolectar
                    self.session_state_callback(cmd.session_id, cmd.session_revision, TrackState.COLLECTING, None)
                
                self.persistence_queue.task_done()
            except Exception as e:
                logger.error("persistence.worker_loop_error", error=str(e))
