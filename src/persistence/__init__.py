from src.persistence.exceptions import (
    PersistenceError,
    RepositoryError,
    TransactionError,
    DuplicateVisitEventError,
    PersonNotFoundError,
    EmbeddingNotFoundError,
)

from src.persistence.db_service import DatabaseService, DB_FILE
from src.persistence.repositories import (
    PersonRepository,
    EmbeddingRepository,
    VisitRepository,
    DecisionAuditRepository,
    CameraRepository,
    TrackSessionLogRepository,
)
from src.persistence.uow import UnitOfWork
from src.persistence.policies import NewIdentityPolicy, IdentityUpdatePolicy
from src.persistence.services import IdentityService, VisitService
