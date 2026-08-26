import os
import sys
import numpy as np
from datetime import datetime

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain import (
    PersonStatus, EmbeddingType, DecisionType, StoredEmbedding,
    PersonIdentity, VisitEvent, SystemConfiguration, FacePose,
    EmbeddingEvidence, CandidateMatch, IdentityEvidence
)
from src.biometrics import VectorSearchService, EvidenceAggregator, HierarchicalDecisionEngine, SecondaryVerificationService
from src.persistence import DatabaseService, UnitOfWork, NewIdentityPolicy, IdentityUpdatePolicy

def test_vector_search():
    print("=== Test 1: VectorSearchService ===")
    search_svc = VectorSearchService(dimension=512)
    print(f"Buscador Vectorial inicializado. FAISS disponible: {search_svc.has_faiss}")

    # Crear algunos embeddings sintéticos
    emb1 = StoredEmbedding(
        embedding_id="emb_01",
        person_id="person_A",
        embedding_type=EmbeddingType.PRIMARY,
        vector=np.ones(512, dtype=np.float32) / np.linalg.norm(np.ones(512)),
        quality_score=0.8,
        pose=None,
        model_name="ArcFace",
        model_version="v1",
        dimension=512,
        active=True,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    emb2 = StoredEmbedding(
        embedding_id="emb_02",
        person_id="person_B",
        embedding_type=EmbeddingType.PRIMARY,
        vector=-np.ones(512, dtype=np.float32) / np.linalg.norm(-np.ones(512)),
        quality_score=0.85,
        pose=None,
        model_name="ArcFace",
        model_version="v1",
        dimension=512,
        active=True,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    search_svc.build_index([emb1, emb2])
    print(f"Elementos en el índice: {search_svc.total_items()}")
    assert search_svc.total_items() == 2

    # Buscar usando vector similar a emb1
    query = np.ones(512, dtype=np.float32) / np.linalg.norm(np.ones(512))
    matches = search_svc.search(query, top_k=2)
    
    print(f"Top 1 match: {matches[0].person_id} con score {matches[0].primary_similarity:.4f}")
    assert matches[0].person_id == "person_A"
    assert matches[0].primary_similarity > 0.99
    print("Test 1 completado con éxito!")


def test_evidence_aggregator():
    print("\n=== Test 2: EvidenceAggregator ===")
    agg = EvidenceAggregator(outlier_threshold=0.70, min_consistency=0.70)
    
    # Crear 8 muestras con embeddings similares y 1 outlier
    emb_base = np.ones(512, dtype=np.float32) / np.linalg.norm(np.ones(512))
    
    evs = []
    # 8 similares
    for i in range(8):
        # Pequeña perturbación
        v = emb_base + np.random.normal(0, 0.01, 512).astype(np.float32)
        v = v / np.linalg.norm(v)
        evs.append(EmbeddingEvidence(
            sample_id=f"sample_{i}",
            track_id=1,
            embedding=v,
            quality_score=0.8,
            pose=None,
            weight=0.0,
            timestamp=datetime.now()
        ))
        
    # 1 outlier (opuesto)
    v_outlier = -emb_base
    evs.append(EmbeddingEvidence(
        sample_id="sample_outlier",
        track_id=1,
        embedding=v_outlier,
        quality_score=0.9,
        pose=None,
        weight=0.0,
        timestamp=datetime.now()
    ))

    result = agg.aggregate("session_test", 1, "CAM_01", evs)
    print(f"Total muestras: {result.total_samples}, Válidas (Estables): {result.valid_samples}, Rechazadas (Outliers): {result.rejected_samples}")
    print(f"Consistencia interna de muestras estables: {result.embedding_consistency:.4f}")
    assert result.rejected_samples == 1
    assert result.valid_samples == 8
    print("Test 2 completado con éxito!")


def test_database_persistence():
    print("\n=== Test 3: Database & UoW & Repositories ===")
    test_db_path = os.path.abspath("data/recurrent_visitors_test.db")
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

    db_svc = DatabaseService(db_path=test_db_path)
    print("Base de datos de prueba inicializada.")

    # Usar UnitOfWork para transaccionalidad
    with UnitOfWork(db_svc) as uow:
        # Crear una persona
        person = PersonIdentity(
            person_id="usr_test_01",
            public_code="IND_000001",
            status=PersonStatus.ACTIVE,
            primary_embedding_id="emb_test_01",
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            total_visits=0
        )
        uow.persons.create(person)
        
        # Crear su embedding primario
        emb = StoredEmbedding(
            embedding_id="emb_test_01",
            person_id="usr_test_01",
            embedding_type=EmbeddingType.PRIMARY,
            vector=np.ones(512, dtype=np.float32) / np.linalg.norm(np.ones(512)),
            quality_score=0.85,
            pose=FacePose(yaw=0.0, pitch=0.0, roll=0.0),
            model_name="ArcFace",
            model_version="v1",
            dimension=512,
            active=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        uow.embeddings.add(emb)
        
        # Incrementar visitas
        uow.persons.increment_visit_count("usr_test_01", datetime.now())

    # Verificar que los datos se guardaron fuera de la transacción
    with UnitOfWork(db_svc) as uow:
        p = uow.persons.get_by_id("usr_test_01")
        assert p is not None
        assert p.public_code == "IND_000001"
        assert p.total_visits == 1
        
        e = uow.embeddings.get_primary("usr_test_01")
        assert e is not None
        assert e.quality_score == 0.85
        assert len(e.vector) == 512
        print(f"Recuperada persona {p.public_code} con total visitas: {p.total_visits}")

    # Limpieza del archivo de prueba
    try:
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    except Exception as e:
        print(f"Advertencia: no se pudo eliminar el archivo de prueba en Windows (bloqueado): {e}")
    print("Test 3 completado con éxito!")


def test_decision_engine_and_policies():
    print("\n=== Test 4: Decision Engine & Policies ===")
    config = SystemConfiguration(
        configuration_id="config_test",
        version="1.0.0",
        min_quality=0.45,
        top_k=5,
        candidate_delta=0.08,
        min_verification_score=0.68,
        min_margin=0.08,
        min_consistency=0.72,
        max_secondary_embeddings=5,
        primary_update_lambda=0.85,
        created_at=datetime.now(),
        active=True
    )

    engine = HierarchicalDecisionEngine(config)
    
    # Caso 1: Evidencia vacía
    evidence = IdentityEvidence(
        evidence_id="ev_01",
        session_id="sess_01",
        track_id=1,
        camera_id="CAM_01",
        embeddings=[],
        aggregated_embedding=np.zeros(512),
        total_samples=0,
        valid_samples=0,
        rejected_samples=0,
        average_quality=0.0,
        embedding_consistency=0.0,
        aggregation_weights=[],
        created_at=datetime.now()
    )
    
    dec = engine.decide("sess_01", 1, "CAM_01", evidence, [])
    print(f"Decisión para evidencia vacía: {dec.decision.name} (Esperado: AMBIGUOUS), Razones: {dec.reason_codes}")
    assert dec.decision == DecisionType.AMBIGUOUS
    assert "AMBIGUOUS_NO_SAMPLES" in dec.reason_codes

    # Caso 2: Match de alta confianza
    emb = np.ones(512, dtype=np.float32) / np.linalg.norm(np.ones(512))
    evidence_match = IdentityEvidence(
        evidence_id="ev_02",
        session_id="sess_02",
        track_id=2,
        camera_id="CAM_01",
        embeddings=[EmbeddingEvidence(
            sample_id="s1", track_id=2, embedding=emb, quality_score=0.8, pose=None, weight=1.0, timestamp=datetime.now()
        )],
        aggregated_embedding=emb,
        total_samples=1,
        valid_samples=1,
        rejected_samples=0,
        average_quality=0.8,
        embedding_consistency=1.0,
        aggregation_weights=[1.0],
        created_at=datetime.now()
    )

    cand = CandidateMatch(
        person_id="person_match",
        rank=1,
        primary_similarity=0.90,
        index_score=0.90,
        retrieved_at=datetime.now()
    )
    
    # Construir un CandidateEvidence ficticio para el test
    from src.domain.decision_models import SimilarityMatrix
    from src.domain.decision_models import CandidateEvidence
    
    cand_evidence = CandidateEvidence(
        person_id="person_match",
        rank=1,
        primary_similarity=0.90,
        similarity_matrix=SimilarityMatrix(values=np.array([[0.90]]), current_embedding_ids=["s1"], candidate_embedding_ids=["c1"]),
        best_similarity=0.90,
        best_per_sample=[0.90],
        mean_top_similarity=0.90,
        median_top_similarity=0.90,
        consistency_score=1.0,
        positive_match_ratio=1.0,
        average_sample_quality=0.8,
        verification_score=0.88 # Supera el min_verification_score de 0.68
    )

    dec_match = engine.decide("sess_02", 2, "CAM_01", evidence_match, [cand_evidence])
    print(f"Decisión para Match de alta confianza: {dec_match.decision.name} (Esperado: MATCH), Persona: {dec_match.person_id}, Razones: {dec_match.reason_codes}")
    assert dec_match.decision == DecisionType.MATCH
    assert dec_match.person_id == "person_match"
    print("Test 4 completado con éxito!")


if __name__ == "__main__":
    print("==============================================")
    print("Iniciando Pruebas de Núcleo de Reconocimiento")
    print("==============================================")
    
    test_vector_search()
    test_evidence_aggregator()
    test_database_persistence()
    test_decision_engine_and_policies()
    
    print("\n==============================================")
    print("¡Todas las pruebas pasaron exitosamente!")
    print("==============================================")
