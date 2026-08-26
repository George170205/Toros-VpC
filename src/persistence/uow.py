import sqlite3
from src.persistence.db_service import DatabaseService, DB_FILE
from src.persistence.repositories import (
    PersonRepository,
    EmbeddingRepository,
    VisitRepository,
    DecisionAuditRepository,
    CameraRepository,
    TrackSessionLogRepository,
)
from src.persistence.exceptions import TransactionError

class UnitOfWork:
    """Implementa el patrón Unit of Work para garantizar transacciones ACID sobre SQLite."""
    def __init__(self, db_service: DatabaseService = None):
        self.db_service = db_service or DatabaseService()
        self.conn: sqlite3.Connection | None = None
        
        # Repositorios instanciados
        self.persons: PersonRepository | None = None
        self.embeddings: EmbeddingRepository | None = None
        self.visits: VisitRepository | None = None
        self.decisions: DecisionAuditRepository | None = None
        self.cameras: CameraRepository | None = None
        self.session_logs: TrackSessionLogRepository | None = None

    def __enter__(self) -> 'UnitOfWork':
        try:
            self.conn = self.db_service.get_connection()
            # Iniciar transacción
            self.conn.execute("BEGIN TRANSACTION;")
            
            # Instanciar repositorios con la conexión transaccional compartida
            self.persons = PersonRepository(self.conn)
            self.embeddings = EmbeddingRepository(self.conn)
            self.visits = VisitRepository(self.conn)
            self.decisions = DecisionAuditRepository(self.conn)
            self.cameras = CameraRepository(self.conn)
            self.session_logs = TrackSessionLogRepository(self.conn)
            
            return self
        except Exception as e:
            if self.conn:
                self.conn.close()
            raise TransactionError(f"Error al iniciar transacción de Unit of Work: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            try:
                if exc_type is not None:
                    # En caso de error, deshacer cambios (rollback)
                    self.conn.rollback()
                else:
                    # Si no hay error, persistir cambios (commit)
                    self.conn.commit()
            except Exception as e:
                raise TransactionError(f"Error al cerrar la transacción (commit/rollback): {e}") from e
            finally:
                self.conn.close()
                self.conn = None
                self._clear_repos()

    def commit(self) -> None:
        """Fuerza un commit explícito dentro del bloque."""
        if self.conn:
            try:
                self.conn.commit()
                # Iniciar otra transacción inmediatamente para continuar el bloque de forma segura
                self.conn.execute("BEGIN TRANSACTION;")
            except Exception as e:
                raise TransactionError(f"Fallo al ejecutar commit explícito: {e}") from e

    def rollback(self) -> None:
        """Fuerza un rollback explícito dentro del bloque."""
        if self.conn:
            try:
                self.conn.rollback()
                self.conn.execute("BEGIN TRANSACTION;")
            except Exception as e:
                raise TransactionError(f"Fallo al ejecutar rollback explícito: {e}") from e

    def _clear_repos(self) -> None:
        self.persons = None
        self.embeddings = None
        self.visits = None
        self.decisions = None
        self.cameras = None
        self.session_logs = None
