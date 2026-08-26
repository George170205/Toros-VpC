class PersistenceError(Exception):
    """Clase base para todos los errores de la capa de persistencia."""
    pass

class RepositoryError(PersistenceError):
    """Error al ejecutar operaciones de repositorio en la base de datos."""
    pass

class TransactionError(PersistenceError):
    """Error durante la gestión de transacciones (commit/rollback)."""
    pass

class DuplicateVisitEventError(PersistenceError):
    """Se disparó al intentar registrar un evento de visita ya existente (idempotencia)."""
    pass

class PersonNotFoundError(PersistenceError):
    """No se encontró la persona con el identificador provisto."""
    pass

class EmbeddingNotFoundError(PersistenceError):
    """No se encontró el embedding provisto."""
    pass
