import sqlite3
import json
import numpy as np
import uuid
from datetime import datetime
from src.domain.enums import PersonStatus, EmbeddingType, DecisionType
from src.domain.vision_models import FacePose
from src.domain.identity_models import PersonIdentity, StoredEmbedding
from src.domain.decision_models import DecisionResult
from src.domain.persistence_models import VisitEvent, Camera, SystemConfiguration
from src.persistence.exceptions import RepositoryError, DuplicateVisitEventError

class PersonRepository:
    """Repositorio SQLite para administrar la entidad de identidades (persons)."""
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_by_id(self, person_id: str) -> PersonIdentity | None:
        try:
            cursor = self.conn.execute(
                "SELECT * FROM persons WHERE person_id = ?", (person_id,)
            )
            row = cursor.fetchone()
            return self._map_row(row) if row else None
        except Exception as e:
            raise RepositoryError(f"Error al obtener persona por ID: {e}") from e

    def get_by_public_code(self, public_code: str) -> PersonIdentity | None:
        try:
            cursor = self.conn.execute(
                "SELECT * FROM persons WHERE public_code = ?", (public_code,)
            )
            row = cursor.fetchone()
            return self._map_row(row) if row else None
        except Exception as e:
            raise RepositoryError(f"Error al obtener persona por código público: {e}") from e

    def create(self, person: PersonIdentity) -> PersonIdentity:
        try:
            self.conn.execute(
                """
                INSERT INTO persons (person_id, public_code, status, primary_embedding_id, first_seen, last_seen, total_visits, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person.person_id,
                    person.public_code,
                    person.status.name,
                    person.primary_embedding_id,
                    person.first_seen.isoformat(),
                    person.last_seen.isoformat(),
                    person.total_visits,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                )
            )
            return person
        except Exception as e:
            raise RepositoryError(f"Error al crear persona: {e}") from e

    def update(self, person: PersonIdentity) -> None:
        try:
            self.conn.execute(
                """
                UPDATE persons
                SET public_code = ?, status = ?, primary_embedding_id = ?, first_seen = ?, last_seen = ?, total_visits = ?, updated_at = ?
                WHERE person_id = ?
                """,
                (
                    person.public_code,
                    person.status.name,
                    person.primary_embedding_id,
                    person.first_seen.isoformat(),
                    person.last_seen.isoformat(),
                    person.total_visits,
                    datetime.now().isoformat(),
                    person.person_id
                )
            )
        except Exception as e:
            raise RepositoryError(f"Error al actualizar persona: {e}") from e

    def increment_visit_count(self, person_id: str, seen_at: datetime) -> None:
        try:
            self.conn.execute(
                """
                UPDATE persons
                SET total_visits = total_visits + 1, last_seen = ?, updated_at = ?
                WHERE person_id = ?
                """,
                (seen_at.isoformat(), datetime.now().isoformat(), person_id)
            )
        except Exception as e:
            raise RepositoryError(f"Error al incrementar contador de visitas: {e}") from e

    def set_status(self, person_id: str, status: PersonStatus) -> None:
        try:
            self.conn.execute(
                "UPDATE persons SET status = ?, updated_at = ? WHERE person_id = ?",
                (status.name, datetime.now().isoformat(), person_id)
            )
        except Exception as e:
            raise RepositoryError(f"Error al cambiar estado de persona: {e}") from e

    def list_active(self) -> list[PersonIdentity]:
        try:
            cursor = self.conn.execute(
                "SELECT * FROM persons WHERE status = ?", (PersonStatus.ACTIVE.name,)
            )
            return [self._map_row(r) for r in cursor.fetchall()]
        except Exception as e:
            raise RepositoryError(f"Error al listar personas activas: {e}") from e

    def get_next_public_code(self) -> str:
        """Genera el código público secuencial IND_000001 de manera dinámica."""
        try:
            cursor = self.conn.execute(
                "SELECT public_code FROM persons ORDER BY public_code DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if not row:
                return "IND_000001"
            
            last_code = row["public_code"]
            num_part = int(last_code.split("_")[1])
            new_num = num_part + 1
            return f"IND_{new_num:06d}"
        except Exception as e:
            raise RepositoryError(f"Error al generar siguiente código público: {e}") from e

    def _map_row(self, row: sqlite3.Row) -> PersonIdentity:
        return PersonIdentity(
            person_id=row["person_id"],
            public_code=row["public_code"],
            status=PersonStatus[row["status"]],
            primary_embedding_id=row["primary_embedding_id"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            total_visits=row["total_visits"]
        )


class EmbeddingRepository:
    """Repositorio SQLite para administrar los embeddings PRIMARY y SECONDARY."""
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_primary(self, person_id: str) -> StoredEmbedding | None:
        try:
            cursor = self.conn.execute(
                "SELECT * FROM embeddings WHERE person_id = ? AND embedding_type = ? AND active = 1 LIMIT 1",
                (person_id, EmbeddingType.PRIMARY.name)
            )
            row = cursor.fetchone()
            return self._map_row(row) if row else None
        except Exception as e:
            raise RepositoryError(f"Error al obtener embedding primario: {e}") from e

    def get_secondary(self, person_id: str, active_only: bool = True) -> list[StoredEmbedding]:
        try:
            query = "SELECT * FROM embeddings WHERE person_id = ? AND embedding_type = ?"
            params = [person_id, EmbeddingType.SECONDARY.name]
            if active_only:
                query += " AND active = 1"
            cursor = self.conn.execute(query, params)
            return [self._map_row(r) for r in cursor.fetchall()]
        except Exception as e:
            raise RepositoryError(f"Error al obtener embeddings secundarios: {e}") from e

    def get_all_primaries(self) -> list[StoredEmbedding]:
        """Recupera todos los embeddings PRIMARY activos para inicializar el VectorIndex."""
        try:
            cursor = self.conn.execute(
                "SELECT * FROM embeddings WHERE embedding_type = ? AND active = 1",
                (EmbeddingType.PRIMARY.name,)
            )
            return [self._map_row(r) for r in cursor.fetchall()]
        except Exception as e:
            raise RepositoryError(f"Error al obtener todos los embeddings primarios: {e}") from e

    def add(self, embedding: StoredEmbedding) -> None:
        try:
            vector_bytes = embedding.vector.astype(np.float32).tobytes()
            
            yaw = embedding.pose.yaw if embedding.pose else None
            pitch = embedding.pose.pitch if embedding.pose else None
            roll = embedding.pose.roll if embedding.pose else None

            self.conn.execute(
                """
                INSERT INTO embeddings (
                    embedding_id, person_id, embedding_type, vector, dimension, 
                    model_name, model_version, quality_score, pose_yaw, pose_pitch, pose_roll, active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    embedding.embedding_id,
                    embedding.person_id,
                    embedding.embedding_type.name,
                    vector_bytes,
                    embedding.dimension,
                    embedding.model_name,
                    embedding.model_version,
                    embedding.quality_score,
                    yaw,
                    pitch,
                    roll,
                    1 if embedding.active else 0,
                    embedding.created_at.isoformat(),
                    embedding.updated_at.isoformat()
                )
            )
        except Exception as e:
            raise RepositoryError(f"Error al insertar embedding: {e}") from e

    def update(self, embedding: StoredEmbedding) -> None:
        try:
            vector_bytes = embedding.vector.astype(np.float32).tobytes()
            yaw = embedding.pose.yaw if embedding.pose else None
            pitch = embedding.pose.pitch if embedding.pose else None
            roll = embedding.pose.roll if embedding.pose else None

            self.conn.execute(
                """
                UPDATE embeddings
                SET person_id = ?, embedding_type = ?, vector = ?, dimension = ?, model_name = ?, model_version = ?, 
                    quality_score = ?, pose_yaw = ?, pose_pitch = ?, pose_roll = ?, active = ?, updated_at = ?
                WHERE embedding_id = ?
                """,
                (
                    embedding.person_id,
                    embedding.embedding_type.name,
                    vector_bytes,
                    embedding.dimension,
                    embedding.model_name,
                    embedding.model_version,
                    embedding.quality_score,
                    yaw,
                    pitch,
                    roll,
                    1 if embedding.active else 0,
                    datetime.now().isoformat(),
                    embedding.embedding_id
                )
            )
        except Exception as e:
            raise RepositoryError(f"Error al actualizar embedding: {e}") from e

    def deactivate(self, embedding_id: str) -> None:
        try:
            self.conn.execute(
                "UPDATE embeddings SET active = 0, updated_at = ? WHERE embedding_id = ?",
                (datetime.now().isoformat(), embedding_id)
            )
        except Exception as e:
            raise RepositoryError(f"Error al desactivar embedding: {e}") from e

    def replace_primary(self, person_id: str, embedding: StoredEmbedding) -> None:
        """Desactiva el primary actual de la persona e inserta uno nuevo en una sola operación transaccional."""
        try:
            self.conn.execute(
                "UPDATE embeddings SET active = 0, updated_at = ? WHERE person_id = ? AND embedding_type = ?",
                (datetime.now().isoformat(), person_id, EmbeddingType.PRIMARY.name)
            )
            self.add(embedding)
        except Exception as e:
            raise RepositoryError(f"Error al reemplazar embedding primario: {e}") from e

    def count_secondary(self, person_id: str) -> int:
        try:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE person_id = ? AND embedding_type = ? AND active = 1",
                (person_id, EmbeddingType.SECONDARY.name)
            )
            return cursor.fetchone()[0]
        except Exception as e:
            raise RepositoryError(f"Error al contar embeddings secundarios: {e}") from e

    def _map_row(self, row: sqlite3.Row) -> StoredEmbedding:
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        pose = None
        if row["pose_yaw"] is not None:
            pose = FacePose(yaw=row["pose_yaw"], pitch=row["pose_pitch"], roll=row["pose_roll"])
            
        return StoredEmbedding(
            embedding_id=row["embedding_id"],
            person_id=row["person_id"],
            embedding_type=EmbeddingType[row["embedding_type"]],
            vector=vector,
            quality_score=row["quality_score"],
            pose=pose,
            model_name=row["model_name"],
            model_version=row["model_version"],
            dimension=row["dimension"],
            active=bool(row["active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )


class VisitRepository:
    """Repositorio SQLite para administrar las visitas confirmadas e idempotencia."""
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def exists_event(self, event_id: str) -> bool:
        try:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM visits WHERE event_id = ?", (event_id,)
            )
            return cursor.fetchone()[0] > 0
        except Exception as e:
            raise RepositoryError(f"Error al verificar existencia de evento de visita: {e}") from e

    def create(self, event: VisitEvent) -> str:
        try:
            if self.exists_event(event.event_id):
                raise DuplicateVisitEventError(f"La visita con event_id {event.event_id} ya fue registrada previamente.")

            self.conn.execute(
                """
                INSERT INTO visits (visit_id, event_id, person_id, camera_id, decision_id, track_id, entry_timestamp, confidence, entry_zone, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"vst_{uuid.uuid4().hex[:12]}" if not hasattr(event, 'visit_id') else f"vst_{uuid.uuid4().hex[:12]}", # Generar ID interno
                    event.event_id,
                    event.person_id,
                    event.camera_id,
                    event.decision_id,
                    event.track_id,
                    event.timestamp.isoformat(),
                    event.confidence,
                    event.entry_zone,
                    datetime.now().isoformat()
                )
            )
            return event.event_id
        except DuplicateVisitEventError as e:
            raise e
        except Exception as e:
            raise RepositoryError(f"Error al registrar visita: {e}") from e

    def list_by_person(self, person_id: str) -> list[VisitEvent]:
        try:
            cursor = self.conn.execute(
                "SELECT * FROM visits WHERE person_id = ? ORDER BY entry_timestamp DESC", (person_id,)
            )
            return [self._map_row(r) for r in cursor.fetchall()]
        except Exception as e:
            raise RepositoryError(f"Error al listar visitas de la persona: {e}") from e

    def count_by_person(self, person_id: str) -> int:
        try:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM visits WHERE person_id = ?", (person_id,)
            )
            return cursor.fetchone()[0]
        except Exception as e:
            raise RepositoryError(f"Error al contar visitas de la persona: {e}") from e

    def _map_row(self, row: sqlite3.Row) -> VisitEvent:
        return VisitEvent(
            event_id=row["event_id"],
            person_id=row["person_id"],
            session_id="",  # No guardado directamente en visits
            track_id=row["track_id"],
            camera_id=row["camera_id"],
            timestamp=datetime.fromisoformat(row["entry_timestamp"]),
            decision_id=row["decision_id"],
            confidence=row["confidence"],
            entry_zone=row["entry_zone"]
        )


class DecisionAuditRepository:
    """Repositorio SQLite para la auditoría de decisiones biométricas."""
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save(self, decision: DecisionResult, configuration_id: str) -> None:
        try:
            reason_json = json.dumps(decision.reason_codes)
            self.conn.execute(
                """
                INSERT INTO identification_decisions (
                    decision_id, track_id, camera_id, selected_person_id, decision_type, decision_confidence, 
                    best_candidate_score, second_candidate_score, margin, average_quality, consistency_score, 
                    positive_match_ratio, temporal_confirmation, reason_codes, decision_engine_version, configuration_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.track_id,
                    decision.camera_id,
                    decision.person_id,
                    decision.decision.name,
                    decision.confidence,
                    decision.best_candidate_score,
                    decision.second_candidate_score,
                    decision.margin,
                    decision.quality_score,
                    decision.consistency_score,
                    decision.positive_match_ratio,
                    decision.temporal_confirmation,
                    reason_json,
                    "1.0.0",
                    configuration_id,
                    decision.timestamp.isoformat()
                )
            )
        except Exception as e:
            raise RepositoryError(f"Error al guardar registro de auditoría de decisión: {e}") from e

    def list_by_session(self, session_id: str) -> list[DecisionResult]:
        # Para simplificar, buscamos decisiones que ocurrieron cerca en base de datos.
        # Las decisiones de auditoría no guardan session_id directamente en SQLite, pero se puede mapear.
        # En v1 retornamos una lista vacía o buscamos por track_id.
        return []

    def get(self, decision_id: str) -> DecisionResult | None:
        try:
            cursor = self.conn.execute(
                "SELECT * FROM identification_decisions WHERE decision_id = ?", (decision_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            
            reasons = json.loads(row["reason_codes"])
            return DecisionResult(
                decision_id=row["decision_id"],
                session_id="",  # No persistido directamente
                track_id=row["track_id"],
                camera_id=row["camera_id"],
                decision=DecisionType[row["decision_type"]],
                person_id=row["selected_person_id"],
                confidence=row["decision_confidence"],
                best_candidate_score=row["best_candidate_score"],
                second_candidate_score=row["second_candidate_score"],
                margin=row["margin"],
                quality_score=row["average_quality"],
                consistency_score=row["consistency_score"],
                positive_match_ratio=row["positive_match_ratio"],
                temporal_confirmation=row["temporal_confirmation"],
                reason_codes=reasons,
                timestamp=datetime.fromisoformat(row["created_at"])
            )
        except Exception as e:
            raise RepositoryError(f"Error al obtener decisión de auditoría: {e}") from e


class CameraRepository:
    """Repositorio SQLite para administrar la configuración lógica de cámaras."""
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, camera_id: str) -> Camera | None:
        try:
            cursor = self.conn.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
            row = cursor.fetchone()
            return self._map_row(row) if row else None
        except Exception as e:
            raise RepositoryError(f"Error al obtener cámara por ID: {e}") from e

    def list_active(self) -> list[Camera]:
        try:
            cursor = self.conn.execute("SELECT * FROM cameras WHERE active = 1")
            return [self._map_row(r) for r in cursor.fetchall()]
        except Exception as e:
            raise RepositoryError(f"Error al listar cámaras activas: {e}") from e

    def save(self, camera: Camera) -> None:
        try:
            self.conn.execute(
                """
                INSERT INTO cameras (camera_id, name, location, entry_zone, resolution_width, resolution_height, target_fps, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    name=excluded.name, location=excluded.location, entry_zone=excluded.entry_zone,
                    resolution_width=excluded.resolution_width, resolution_height=excluded.resolution_height,
                    target_fps=excluded.target_fps, active=excluded.active
                """,
                (
                    camera.camera_id,
                    camera.name,
                    camera.location,
                    camera.entry_zone,
                    camera.resolution_width,
                    camera.resolution_height,
                    camera.target_fps,
                    1 if camera.active else 0,
                    datetime.now().isoformat()
                )
            )
        except Exception as e:
            raise RepositoryError(f"Error al guardar cámara: {e}") from e

    def _map_row(self, row: sqlite3.Row) -> Camera:
        return Camera(
            camera_id=row["camera_id"],
            name=row["name"],
            location=row["location"],
            entry_zone=row["entry_zone"],
            resolution_width=row["resolution_width"],
            resolution_height=row["resolution_height"],
            target_fps=row["target_fps"],
            active=bool(row["active"])
        )


class TrackSessionLogRepository:
    """Repositorio SQLite para persistir resúmenes de TrackSessions finalizados."""
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save_summary(
        self,
        session_id: str,
        track_id: int,
        camera_id: str,
        first_seen: datetime,
        last_seen: datetime,
        total_frames: int,
        valid_samples: int,
        rejected_samples: int,
        final_state: str,
        decision_id: str | None
    ) -> None:
        try:
            self.conn.execute(
                """
                INSERT INTO track_session_logs (
                    session_id, track_id, camera_id, first_seen, last_seen, 
                    total_frames, valid_samples, rejected_samples, final_state, decision_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    track_id,
                    camera_id,
                    first_seen.isoformat(),
                    last_seen.isoformat(),
                    total_frames,
                    valid_samples,
                    rejected_samples,
                    final_state,
                    decision_id,
                    datetime.now().isoformat()
                )
            )
        except Exception as e:
            raise RepositoryError(f"Error al guardar registro de sesión de track: {e}") from e
