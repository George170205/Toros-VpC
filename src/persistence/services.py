import uuid
from datetime import datetime
from src.domain.enums import PersonStatus, EmbeddingType
from src.domain.identity_models import PersonIdentity, StoredEmbedding
from src.domain.decision_models import DecisionResult, IdentityUpdateDecision, NewIdentityDecision
from src.domain.persistence_models import VisitEvent, SystemConfiguration
from src.domain.vision_models import FacePose
from src.persistence.uow import UnitOfWork
from src.persistence.policies import IdentityUpdatePolicy

class IdentityService:
    """Servicio de aplicación para orquestar la creación y actualización transaccional de identidades biométricas."""
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.update_policy = IdentityUpdatePolicy(config)

    def create_identity(
        self,
        uow: UnitOfWork,
        evidence: IdentityEvidence,
        decision: DecisionResult,
        creation: NewIdentityDecision
    ) -> PersonIdentity:
        # Generar identificadores únicos
        person_id = f"usr_{uuid.uuid4().hex[:12]}"
        primary_emb_id = f"emb_{uuid.uuid4().hex[:12]}"
        
        # Obtener siguiente código seudónimo secuencial
        public_code = uow.persons.get_next_public_code()

        # 1. Crear entidad PersonIdentity
        person = PersonIdentity(
            person_id=person_id,
            public_code=public_code,
            status=PersonStatus.ACTIVE,
            primary_embedding_id=primary_emb_id,
            first_seen=decision.timestamp,
            last_seen=decision.timestamp,
            total_visits=0  # Se incrementa al registrar la visita
        )

        # 2. Crear StoredEmbedding PRIMARY
        primary_emb = StoredEmbedding(
            embedding_id=primary_emb_id,
            person_id=person_id,
            embedding_type=EmbeddingType.PRIMARY,
            vector=creation.primary_embedding,
            quality_score=creation.quality_score,
            pose=FacePose(yaw=0.0, pitch=0.0, roll=0.0),  # Pose agregada neutralizada
            model_name="ArcFace",
            model_version="buffalo_l_r50",
            dimension=512,
            active=True,
            created_at=decision.timestamp,
            updated_at=decision.timestamp
        )

        # Guardar en base de datos
        uow.persons.create(person)
        uow.embeddings.add(primary_emb)

        # 3. Guardar StoredEmbeddings SECONDARY iniciales
        for idx, vec in enumerate(creation.secondary_embeddings):
            sec_id = f"emb_{uuid.uuid4().hex[:12]}"
            # Mapear metadatos de calidad de la muestra correspondiente
            matching_sample = evidence.embeddings[idx] if idx < len(evidence.embeddings) else None
            q_score = matching_sample.quality_score if matching_sample else creation.quality_score
            pose = matching_sample.pose if matching_sample else None
            
            sec_emb = StoredEmbedding(
                embedding_id=sec_id,
                person_id=person_id,
                embedding_type=EmbeddingType.SECONDARY,
                vector=vec,
                quality_score=q_score,
                pose=pose,
                model_name="ArcFace",
                model_version="buffalo_l_r50",
                dimension=512,
                active=True,
                created_at=decision.timestamp,
                updated_at=decision.timestamp
            )
            uow.embeddings.add(sec_emb)

        return person

    def update_identity(
        self,
        uow: UnitOfWork,
        person_id: str,
        evidence: IdentityEvidence,
        decision: DecisionResult,
        update: IdentityUpdateDecision
    ) -> tuple[bool, str | None]:
        """Aplica las modificaciones decididas por la política al perfil (PRIMARY y/o SECONDARY)."""
        primary_changed = False
        new_primary_id = None
        
        # 1. Actualizar el PRIMARY si aplica
        if update.update_primary:
            old_primary = uow.embeddings.get_primary(person_id)
            if old_primary:
                new_vec = self.update_policy.calculate_updated_primary(old_primary.vector, evidence.aggregated_embedding)
                new_primary_id = f"emb_{uuid.uuid4().hex[:12]}"
                
                new_primary = StoredEmbedding(
                    embedding_id=new_primary_id,
                    person_id=person_id,
                    embedding_type=EmbeddingType.PRIMARY,
                    vector=new_vec,
                    quality_score=evidence.average_quality,
                    pose=FacePose(yaw=0.0, pitch=0.0, roll=0.0),
                    model_name="ArcFace",
                    model_version="buffalo_l_r50",
                    dimension=512,
                    active=True,
                    created_at=decision.timestamp,
                    updated_at=decision.timestamp
                )
                
                # Reemplazar transaccionalmente en la base de datos
                uow.embeddings.replace_primary(person_id, new_primary)
                
                # Actualizar el puntero en la entidad PersonIdentity
                person = uow.persons.get_by_id(person_id)
                if person:
                    person.primary_embedding_id = new_primary_id
                    uow.persons.update(person)
                
                primary_changed = True

        # 2. Agregar/Reemplazar SECONDARY si aplica
        if update.add_secondary:
            # Desactivar anterior si fue un reemplazo
            if update.replace_secondary_id:
                uow.embeddings.deactivate(update.replace_secondary_id)

            # Tomar la muestra actual de mejor calidad
            best_idx = int(np.argmax([e.quality_score for e in evidence.embeddings]))
            best_sample = evidence.embeddings[best_idx]
            
            new_sec_id = f"emb_{uuid.uuid4().hex[:12]}"
            new_secondary = StoredEmbedding(
                embedding_id=new_sec_id,
                person_id=person_id,
                embedding_type=EmbeddingType.SECONDARY,
                vector=best_sample.embedding,
                quality_score=best_sample.quality_score,
                pose=best_sample.pose,
                model_name="ArcFace",
                model_version="buffalo_l_r50",
                dimension=512,
                active=True,
                created_at=decision.timestamp,
                updated_at=decision.timestamp
            )
            uow.embeddings.add(new_secondary)

        return primary_changed, new_primary_id


class VisitService:
    """Servicio de aplicación para registrar visitas de forma idempotente."""
    def __init__(self):
        pass

    def register(self, uow: UnitOfWork, event: VisitEvent) -> str:
        # Cláusula de Idempotencia: Verificar si la visita ya está registrada
        if uow.visits.exists_event(event.event_id):
            # Retornar el event_id directamente sin duplicar el registro
            return event.event_id

        # 1. Crear el registro de visita
        uow.visits.create(event)
        
        # 2. Incrementar el contador denormalizado de visitas del PersonIdentity
        uow.persons.increment_visit_count(event.person_id, event.timestamp)
        
        return event.event_id
