import os
import sqlite3
import json
from datetime import datetime
from src.persistence.exceptions import TransactionError

DB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "recurrent_visitors.db"))

class DatabaseService:
    """Servicio que administra la creación, conexión e inicialización del esquema SQLite."""
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        """Obtiene una conexión con soporte de claves foráneas y modo WAL habilitado."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Pragmáticos para optimización y consistencia en SQLite
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            return conn
        except Exception as e:
            raise TransactionError(f"Error al abrir conexión con la base de datos: {e}") from e

    def init_database(self) -> None:
        """Inicializa las tablas e índices si no existen."""
        with self.get_connection() as conn:
            # 1. Tabla: cameras
            conn.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT,
                entry_zone TEXT,
                resolution_width INTEGER,
                resolution_height INTEGER,
                target_fps REAL,
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );
            """)

            # 2. Tabla: system_configurations
            conn.execute("""
            CREATE TABLE IF NOT EXISTS system_configurations (
                configuration_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                active INTEGER DEFAULT 0
            );
            """)

            # 3. Tabla: persons
            conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,
                public_code TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                primary_embedding_id TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                total_visits INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 4. Tabla: embeddings
            conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                embedding_type TEXT NOT NULL,
                vector BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                quality_score REAL NOT NULL,
                pose_yaw REAL,
                pose_pitch REAL,
                pose_roll REAL,
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE
            );
            """)

            # 5. Tabla: identification_decisions
            conn.execute("""
            CREATE TABLE IF NOT EXISTS identification_decisions (
                decision_id TEXT PRIMARY KEY,
                track_id INTEGER NOT NULL,
                camera_id TEXT NOT NULL,
                selected_person_id TEXT,
                decision_type TEXT NOT NULL,
                decision_confidence REAL NOT NULL,
                best_candidate_score REAL,
                second_candidate_score REAL,
                margin REAL,
                average_quality REAL NOT NULL,
                consistency_score REAL NOT NULL,
                positive_match_ratio REAL NOT NULL,
                temporal_confirmation REAL NOT NULL,
                reason_codes TEXT NOT NULL, -- Guardado como JSON serializado
                decision_engine_version TEXT NOT NULL,
                configuration_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (camera_id) REFERENCES cameras (camera_id),
                FOREIGN KEY (selected_person_id) REFERENCES persons (person_id) ON DELETE SET NULL,
                FOREIGN KEY (configuration_id) REFERENCES system_configurations (configuration_id)
            );
            """)

            # 6. Tabla: visits
            conn.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                visit_id TEXT PRIMARY KEY,
                event_id TEXT UNIQUE NOT NULL, -- Clave de idempotencia
                person_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                entry_timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_zone TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE,
                FOREIGN KEY (camera_id) REFERENCES cameras (camera_id),
                FOREIGN KEY (decision_id) REFERENCES identification_decisions (decision_id)
            );
            """)

            # 7. Tabla: track_session_logs
            conn.execute("""
            CREATE TABLE IF NOT EXISTS track_session_logs (
                session_id TEXT PRIMARY KEY,
                track_id INTEGER NOT NULL,
                camera_id TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                total_frames INTEGER NOT NULL,
                valid_samples INTEGER NOT NULL,
                rejected_samples INTEGER NOT NULL,
                final_state TEXT NOT NULL,
                decision_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (camera_id) REFERENCES cameras (camera_id),
                FOREIGN KEY (decision_id) REFERENCES identification_decisions (decision_id)
            );
            """)

            # 8. Índices para acelerar búsquedas
            conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_person ON visits (person_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_time ON visits (entry_timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings (person_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings (embedding_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_created ON identification_decisions (created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_logs_track ON track_session_logs (track_id);")

            # 9. Insertar cámara por defecto e inicializar configuración del sistema si están vacías
            self._init_defaults(conn)
            
            conn.commit()

    def _init_defaults(self, conn: sqlite3.Connection) -> None:
        """Crea la cámara por defecto y la configuración inicial de control si no existen."""
        # Cámara por defecto
        cursor = conn.execute("SELECT COUNT(*) FROM cameras WHERE camera_id = 'CAM_01'")
        if cursor.fetchone()[0] == 0:
            conn.execute("""
            INSERT INTO cameras (camera_id, name, location, entry_zone, resolution_width, resolution_height, target_fps, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ('CAM_01', 'Cámara Principal Acceso', 'Acceso Principal Norte', 'Norte-Entrada', 640, 480, 25.0, 1, datetime.now().isoformat()))

        # Configuración por defecto
        cursor = conn.execute("SELECT COUNT(*) FROM system_configurations WHERE configuration_id = 'config_default_v1'")
        if cursor.fetchone()[0] == 0:
            params = {
                "min_quality": 0.45,
                "top_k": 5,
                "candidate_delta": 0.08,
                "min_verification_score": 0.68,
                "min_margin": 0.08,
                "min_consistency": 0.72,
                "max_secondary_embeddings": 5,
                "primary_update_lambda": 0.85
            }
            conn.execute("""
            INSERT INTO system_configurations (configuration_id, version, parameters_json, created_at, active)
            VALUES (?, ?, ?, ?, ?)
            """, ('config_default_v1', '1.0.0', json.dumps(params), datetime.now().isoformat(), 1))
