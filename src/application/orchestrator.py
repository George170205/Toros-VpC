import time
import uuid
import threading
import cv2
import base64
from datetime import datetime
from typing import Optional, Callable
from src.domain import (
    Frame, TrackedFace, TrackState, FaceSample,
    DecisionResult, IdentityEvidence, Camera, SystemConfiguration
)
from src.vision import (
    CameraSource, SCRFDFaceDetector, SimpleFaceTracker,
    FaceQualityAnalyzer, SampleCollector, TrackSession
)
from src.biometrics import VectorSearchService, SecondaryVerificationService, HierarchicalDecisionEngine
from src.persistence import DatabaseService, UnitOfWork, IdentityService, VisitService, NewIdentityPolicy, IdentityUpdatePolicy
from src.application.queues import LatestFrameQueue, TaskQueue
from src.application.workers import RecognitionTask, RecognitionResult, PersistenceCommand, EmbeddingWorker, IdentityPipelineWorker
from src.application.persistence_worker import PersistenceWorker
from src.utils.observability import logger, metrics, CorrelationContext

class PipelineOrchestrator:
    """Coordinador central del pipeline de tiempo real y asíncrono biométrico."""
    def __init__(
        self,
        camera_id: str,
        model_det_path: str,
        model_rec_path: str,
        config: SystemConfiguration,
        live_event_callback: Callable[[dict], None],
        frame_callback: Optional[Callable[[str], None]] = None
    ):
        self.camera_id = camera_id
        self.config = config
        self.live_event_callback = live_event_callback
        self.frame_callback = frame_callback
        
        # Servicios e infraestructura de base de datos
        self.db_service = DatabaseService()
        self.search_service = VectorSearchService()
        
        # Sincronizar VectorIndex inicial con base de datos
        self._sync_vector_index()

        # Componentes del pipeline de visión
        self.camera_source = CameraSource(camera_id)
        self.detector = SCRFDFaceDetector(model_det_path, default_threshold=config.min_quality)
        self.tracker = SimpleFaceTracker(iou_threshold=0.35, max_lost_frames=30)
        self.quality_analyzer = FaceQualityAnalyzer(min_quality_threshold=config.min_quality)
        self.sample_collector = SampleCollector(max_samples=config.max_secondary_embeddings, min_valid_samples=3)

        # Componentes del pipeline biométrico
        self.verification_service = SecondaryVerificationService(positive_threshold=0.70)
        self.decision_engine = HierarchicalDecisionEngine(config)
        self.identity_service = IdentityService(config)
        self.visit_service = VisitService()
        self.new_identity_policy = NewIdentityPolicy(config)
        self.identity_update_policy = IdentityUpdatePolicy(config)

        # Colas acotadas
        self.frame_queue = LatestFrameQueue(maxsize=3)
        self.recognition_queue = TaskQueue[RecognitionTask](maxsize=10)
        self.result_queue = TaskQueue[RecognitionResult](maxsize=10)
        self.persistence_queue = TaskQueue[PersistenceCommand](maxsize=15)

        # Sesiones de track en memoria
        self.sessions: dict[int, TrackSession] = {}
        self._sessions_lock = threading.Lock()

        # Workers
        self.embedding_worker = EmbeddingWorker(
            model_path=model_rec_path,
            task_queue=self.recognition_queue,
            result_queue=self.result_queue,
            outlier_thresh=0.72,
            min_consistency=config.min_consistency
        )
        self.pipeline_worker = IdentityPipelineWorker(
            result_queue=self.result_queue,
            persistence_queue=self.persistence_queue,
            search_service=self.search_service,
            verification_service=self.verification_service,
            decision_engine=self.decision_engine,
            load_profile_func=self._load_profile_embeddings,
            session_state_callback=self._update_session_state
        )
        self.persistence_worker = PersistenceWorker(
            persistence_queue=self.persistence_queue,
            search_service=self.search_service,
            db_service=self.db_service,
            identity_service=self.identity_service,
            visit_service=self.visit_service,
            new_identity_policy=self.new_identity_policy,
            identity_update_policy=self.identity_update_policy,
            session_state_callback=self._update_session_state,
            live_event_callback=self._on_live_decision_event
        )

        # Hilos del coordinador
        self._capture_thread = None
        self._vision_thread = None
        self._cleanup_thread = None
        self._stop_event = threading.Event()

    def _sync_vector_index(self) -> None:
        """Carga todos los embeddings PRIMARY activos para inicializar FAISS."""
        try:
            with UnitOfWork(self.db_service) as uow:
                primaries = uow.embeddings.get_all_primaries()
                self.search_service.build_index(primaries)
            logger.info("vector_index.synchronized", total_items=self.search_service.total_items())
        except Exception as e:
            logger.error("vector_index.sync_failed", error=str(e))

    def _load_profile_embeddings(self, person_id: str) -> list[StoredEmbedding]:
        """Callback para que los workers carguen el perfil biométrico de la base de datos."""
        try:
            with UnitOfWork(self.db_service) as uow:
                primary = uow.embeddings.get_primary(person_id)
                secondaries = uow.embeddings.get_secondary(person_id)
                profile = []
                if primary:
                    profile.append(primary)
                profile.extend(secondaries)
                return profile
        except Exception as e:
            logger.error("persistence.load_profile_failed", person_id=person_id, error=str(e))
            return []

    def _update_session_state(self, session_id: str, revision: int, state: TrackState, public_code: str = None) -> None:
        """Callback thread-safe para que los workers actualicen el estado del TrackSession en memoria."""
        with self._sessions_lock:
            # Buscar la sesión por ID
            target_track_id = None
            for tid, sess in self.sessions.items():
                if sess.session_id == session_id:
                    target_track_id = tid
                    break
            
            if target_track_id is None:
                return

            session = self.sessions[target_track_id]
            # Validar número de revisión para evitar sobreescribir con resultados obsoletos
            if session.revision > revision:
                logger.warning(
                    "orchestrator.stale_revision_update_ignored",
                    session_id=session_id,
                    current_rev=session.revision,
                    update_rev=revision
                )
                return

            session.state = state
            
            if state == TrackState.FINISHED:
                # Marcar visita registrada para que no envíe más comandos
                session.visit_registered = True
                session.public_code = public_code
                logger.info("orchestrator.session_finished", session_id=session_id, public_code=public_code)
            elif state == TrackState.COLLECTING:
                # Incrementar intentos de identificación
                session.identification_attempts += 1
                logger.debug("orchestrator.session_reset_to_collecting", session_id=session_id)
            elif state == TrackState.AMBIGUOUS:
                session.public_code = "AMBIGUO"
                logger.info("orchestrator.session_ambiguous", session_id=session_id)
            elif state == TrackState.UNKNOWN:
                session.public_code = public_code if public_code else "DESCONOCIDO"
                logger.info("orchestrator.session_unknown", session_id=session_id)

    def _on_live_decision_event(self, event_data: dict) -> None:
        # Enviar notificación en vivo al servidor FastAPI
        self.live_event_callback(event_data)

    def start(self) -> None:
        """Inicia todos los hilos de captura, procesamiento de visión y workers biométricos."""
        self._stop_event.clear()
        self.camera_source.open()
        
        # Iniciar workers
        self.embedding_worker.start()
        self.pipeline_worker.start()
        self.persistence_worker.start()

        # Iniciar hilos del orquestador
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True, name="VisionCoordinator")
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="CleanupThread")
        
        self._capture_thread.start()
        self._vision_thread.start()
        self._cleanup_thread.start()
        logger.info("PipelineOrchestrator iniciado con éxito.")

    def stop(self) -> None:
        """Ejecuta un apagado controlado (Graceful Shutdown) vaciando las colas y deteniendo los hilos."""
        logger.info("Iniciando apagado controlado del orquestador...")
        self._stop_event.set()
        
        # 1. Detener captura y cerrar cola de frames
        self.frame_queue.close()
        self.camera_source.close()
        
        # 2. Esperar a que terminen hilos de captura y visión
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._vision_thread:
            self._vision_thread.join(timeout=2.0)
            
        # 3. Detener workers y esperar a que drenen
        self.embedding_worker.stop()
        self.pipeline_worker.stop()
        self.persistence_worker.stop()
        
        self.embedding_worker.join(timeout=2.0)
        self.pipeline_worker.join(timeout=2.0)
        self.persistence_worker.join(timeout=2.0)
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)
            
        logger.info("PipelineOrchestrator detenido con éxito.")

    def _capture_loop(self) -> None:
        """Hilo de captura en tiempo real."""
        while not self._stop_event.is_set() and self.camera_source.is_open():
            try:
                frame = self.camera_source.read()
                self.frame_queue.put(frame)
            except Exception as e:
                logger.error("capture.loop_error", error=str(e))
                time.sleep(0.5)

    def _vision_loop(self) -> None:
        """Coordinador del camino en tiempo real (visión, tracking y recolección de calidad)."""
        while not self._stop_event.is_set():
            try:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                t_start = datetime.now()

                # 1. Detección Facial (SCRFD ONNX)
                detections = self.detector.detect(frame)
                metrics.observe("fps_capture", 1.0 / max(0.001, (datetime.now() - frame.timestamp).total_seconds()))

                # 2. Tracking Temporal (SimpleFaceTracker)
                tracked_faces = self.tracker.update(frame, detections)
                
                with self._sessions_lock:
                    for face in tracked_faces:
                        tid = face.track_id
                        
                        # Obtener o crear TrackSession para este track_id
                        if tid not in self.sessions:
                            self.sessions[tid] = TrackSession(
                                session_id=f"sess_{uuid.uuid4().hex[:12]}",
                                track_id=tid,
                                camera_id=frame.camera_id,
                                first_seen=frame.timestamp,
                                last_seen=frame.timestamp,
                                state=TrackState.NEW,
                                samples=[]
                            )
                            logger.info("orchestrator.track_created", track_id=tid, session_id=self.sessions[tid].session_id)
                        
                        session = self.sessions[tid]
                        
                        # Si ya se identificó o finalizó, ignorar procesamiento biométrico adicional
                        if session.state in (TrackState.FINISHED, TrackState.EXPIRED) or session.visit_registered:
                            continue

                        # Si está en medio de una inferencia asíncrona, no recopilar temporalmente
                        if session.state in (TrackState.GENERATING_EMBEDDINGS, TrackState.SEARCHING, TrackState.VERIFYING):
                            continue

                        # 3. Analizar Calidad Facial
                        quality = self.quality_analyzer.analyze(frame, face)
                        
                        # 4. Extraer Muestra y agregar
                        if quality.is_acceptable:
                            # Recortar rostro de la imagen original
                            img = frame.image
                            h, w = img.shape[:2]
                            xmin = max(0, face.bbox[0])
                            ymin = max(0, face.bbox[1])
                            xmax = min(w, face.bbox[2])
                            ymax = min(h, face.bbox[3])
                            face_img = img[ymin:ymax, xmin:xmax]
                            
                            sample = FaceSample(
                                sample_id=f"smpl_{uuid.uuid4().hex[:12]}",
                                track_id=tid,
                                frame_id=frame.frame_id,
                                camera_id=frame.camera_id,
                                timestamp=frame.timestamp,
                                face_image=face_img,
                                quality=quality
                            )
                            
                            # Intentar agregar muestra a la sesión
                            self.sample_collector.add_sample(session, sample)
                            session.state = TrackState.COLLECTING

                        # 5. Evaluar si la sesión alcanzó READY para reconocimiento
                        if self.sample_collector.is_ready(session):
                            session.state = TrackState.GENERATING_EMBEDDINGS
                            
                            # Encolar Tarea de Reconocimiento
                            task = RecognitionTask(
                                task_id=f"tsk_{uuid.uuid4().hex[:12]}",
                                session_id=session.session_id,
                                track_id=tid,
                                camera_id=frame.camera_id,
                                samples=list(session.samples),
                                created_at=datetime.now(),
                                session_revision=session.revision
                            )
                            self.recognition_queue.put(task)
                            logger.debug("orchestrator.task_submitted", session_id=session.session_id, revision=session.revision)

                # 6. Dibujar overlay visual en tiempo real
                annotated_img = frame.image.copy()
                with self._sessions_lock:
                    for face in tracked_faces:
                        tid = face.track_id
                        state_str = "NUEVO"
                        code_str = ""
                        color = (0, 255, 255) # Amarillo
                        
                        if tid in self.sessions:
                            sess = self.sessions[tid]
                            state_str = sess.state.name
                            if sess.state == TrackState.FINISHED:
                                color = (20, 200, 110) # Verde Toros
                                code_str = getattr(sess, 'public_code', '')
                            elif sess.state == TrackState.AMBIGUOUS:
                                color = (20, 160, 240) # Naranja/Amber
                                code_str = "AMBIGUO"
                            elif sess.state == TrackState.UNKNOWN:
                                color = (255, 50, 80) # Rojo/Azul
                                code_str = getattr(sess, 'public_code', 'DESCONOCIDO')
                            else:
                                code_str = f"Muestras: {len(sess.samples)}"

                        # Dibujar Bounding Box
                        cv2.rectangle(
                            annotated_img,
                            (face.bbox[0], face.bbox[1]),
                            (face.bbox[2], face.bbox[3]),
                            color,
                            2
                        )
                        
                        # Dibujar texto
                        label = f"ID: {tid} [{state_str}] {code_str}"
                        cv2.putText(
                            annotated_img,
                            label,
                            (face.bbox[0], face.bbox[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA
                        )

                # Redimensionar para transmisión por WebSocket si es mayor a 640px de ancho (mejora rendimiento y reduce lag de red)
                h_w = annotated_img.shape[:2]
                target_width = 640
                if h_w[1] > target_width:
                    scale = target_width / h_w[1]
                    new_h = int(h_w[0] * scale)
                    annotated_img = cv2.resize(annotated_img, (target_width, new_h))

                # Comprimir a JPEG y codificar a Base64
                _, buffer = cv2.imencode('.jpg', annotated_img)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                # Transmitir por websocket a través del callback
                if self.frame_callback:
                    self.frame_callback(jpg_as_text)

                t_cost = (datetime.now() - t_start).total_seconds() * 1000
                metrics.observe("vision_latency_ms", t_cost)

            except Exception as e:
                logger.error("vision.pipeline_loop_error", error=str(e))
                time.sleep(0.01)

    def _cleanup_loop(self) -> None:
        """Hilo en segundo plano para expirar sesiones de tracks que salieron de la escena."""
        while not self._stop_event.is_set():
            try:
                # Ejecutar limpieza cada 2.5 segundos
                time.sleep(2.5)
                now = datetime.now()
                expired_count = 0
                
                with self._sessions_lock:
                    to_remove = []
                    for tid, session in self.sessions.items():
                        # Si la sesión no se ha actualizado en los últimos 4 segundos, expirar
                        time_inactive = (now - session.last_seen).total_seconds()
                        
                        if time_inactive > 4.0:
                            to_remove.append(tid)
                            if session.state != TrackState.FINISHED and not session.visit_registered:
                                session.state = TrackState.EXPIRED
                                expired_count += 1
                                logger.info("orchestrator.track_expired", track_id=tid, session_id=session.session_id)
                                
                    for tid in to_remove:
                        del self.sessions[tid]
                        
                if expired_count > 0:
                    metrics.increment("track_expired", expired_count)
                    
            except Exception as e:
                logger.error("orchestrator.cleanup_loop_error", error=str(e))
