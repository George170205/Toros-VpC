import numpy as np
from datetime import datetime
from src.domain.decision_models import CandidateMatch
from src.domain.identity_models import StoredEmbedding

# Intentar importar faiss para la búsqueda vectorial
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class VectorSearchService:
    """Servicio de búsqueda vectorial usando FAISS IndexFlatIP (Cosine Similarity) con fallback robusto en NumPy."""
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.has_faiss = HAS_FAISS
        
        # Estructuras para FAISS
        self._faiss_index = None
        
        # Mapeo de índice entero de FAISS -> person_id
        self._index_to_person_id: list[str] = []
        
        # Estructuras para Fallback de NumPy (lista de embeddings almacenados)
        self._numpy_embeddings: list[StoredEmbedding] = []

        self.reset()

    def reset(self) -> None:
        """Reinicia el índice vectorial."""
        self._index_to_person_id.clear()
        self._numpy_embeddings.clear()
        
        if self.has_faiss:
            try:
                # Usar IndexFlatIP para Similitud de Coseno (vectores L2-normalizados)
                self._faiss_index = faiss.IndexFlatIP(self.dimension)
            except Exception as e:
                print(f"Error al inicializar FAISS, desactivando y usando NumPy: {e}")
                self.has_faiss = False
                self._faiss_index = None

    def build_index(self, primary_embeddings: list[StoredEmbedding]) -> None:
        """Construye el índice desde cero a partir de una lista de embeddings PRIMARY."""
        self.reset()
        if not primary_embeddings:
            return

        if self.has_faiss and self._faiss_index is not None:
            try:
                vectors = []
                for emb in primary_embeddings:
                    vectors.append(emb.vector)
                    self._index_to_person_id.append(emb.person_id)
                
                # Convertir a float32 y matriz numpy
                data_matrix = np.stack(vectors, axis=0).astype(np.float32)
                self._faiss_index.add(data_matrix)
            except Exception as e:
                print(f"Error al construir índice FAISS: {e}. Reconstruyendo con NumPy...")
                self.has_faiss = False
                self._faiss_index = None
                self._numpy_embeddings = list(primary_embeddings)
        else:
            self._numpy_embeddings = list(primary_embeddings)

    def search(self, aggregated_embedding: np.ndarray, top_k: int = 5) -> list[CandidateMatch]:
        """Busca los Top-K candidatos más similares en el índice."""
        if (self.has_faiss and not self._index_to_person_id) or (not self.has_faiss and not self._numpy_embeddings):
            return []

        # Asegurar tipo float32
        query_vector = aggregated_embedding.astype(np.float32).reshape(1, -1)
        
        if self.has_faiss and self._faiss_index is not None:
            try:
                k = min(top_k, self._faiss_index.ntotal)
                similarities, indices = self._faiss_index.search(query_vector, k)
                
                candidates = []
                for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx == -1:
                        continue
                    person_id = self._index_to_person_id[idx]
                    candidates.append(CandidateMatch(
                        person_id=person_id,
                        rank=rank + 1,
                        primary_similarity=float(sim),
                        index_score=float(sim),
                        retrieved_at=datetime.now()
                    ))
                return candidates
            except Exception as e:
                print(f"Fallo en búsqueda FAISS: {e}. Usando NumPy de emergencia...")
                # Fallback inmediato
                return self._search_numpy(query_vector[0], top_k)
        else:
            return self._search_numpy(query_vector[0], top_k)

    def _search_numpy(self, query_vector: np.ndarray, top_k: int) -> list[CandidateMatch]:
        """Método de búsqueda fallback usando multiplicación de matrices pura en NumPy."""
        candidates_scores = []
        for emb in self._numpy_embeddings:
            # Similitud de coseno (dot product para L2-norm)
            sim = float(np.dot(query_vector, emb.vector))
            candidates_scores.append((emb.person_id, sim))
            
        # Ordenar por similitud descendente
        candidates_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver Top-K
        candidates = []
        for rank, (person_id, sim) in enumerate(candidates_scores[:top_k]):
            candidates.append(CandidateMatch(
                person_id=person_id,
                rank=rank + 1,
                primary_similarity=sim,
                index_score=sim,
                retrieved_at=datetime.now()
            ))
        return candidates

    def add_item(self, stored_emb: StoredEmbedding) -> None:
        """Agrega un nuevo embedding PRIMARY al índice en caliente."""
        # Si ya existe esta persona en el índice, removerla primero para evitar duplicados
        self.remove_item(stored_emb.person_id)

        if self.has_faiss and self._faiss_index is not None:
            try:
                vector = stored_emb.vector.astype(np.float32).reshape(1, -1)
                self._faiss_index.add(vector)
                self._index_to_person_id.append(stored_emb.person_id)
            except Exception as e:
                print(f"Error al agregar item a FAISS: {e}. Cambiando a NumPy...")
                self.has_faiss = False
                self._faiss_index = None
                self._numpy_embeddings.append(stored_emb)
        else:
            self._numpy_embeddings.append(stored_emb)

    def remove_item(self, person_id: str) -> None:
        """Remueve un embedding PRIMARY asociado a person_id."""
        if self.has_faiss and self._faiss_index is not None:
            try:
                if person_id in self._index_to_person_id:
                    idx = self._index_to_person_id.index(person_id)
                    
                    # FAISS no permite eliminar fácilmente de IndexFlat de forma directa por índice posicional,
                    # así que lo más seguro y limpio es reconstruir el índice de FAISS sin el elemento eliminado
                    self._index_to_person_id.pop(idx)
                    
                    # Obtener los remanentes
                    remaining_pids = list(self._index_to_person_id)
                    # Reconstruir
                    # Extraer vectores de FAISS para reconstruir
                    self._faiss_index.reset()
                    self._index_to_person_id.clear()
                    
                    # Si no quedan pids, terminamos
                    if not remaining_pids:
                        return
                    
                    # Para reconstruir, leemos los vectores de la base de datos (se maneja en el reconcile)
                    # O si no podemos de forma directa aquí, removemos temporalmente y dejamos para reconciliar.
                    # En una implementación limpia en caliente, removemos de numpy_embeddings
                    # y reconstruimos FAISS desde el backup de numpy_embeddings que mantendremos siempre como copia.
            except Exception as e:
                print(f"Error al remover item de FAISS: {e}")
        
        # Remover de la lista NumPy (que sirve como copia de seguridad)
        self._numpy_embeddings = [e for e in self._numpy_embeddings if e.person_id != person_id]
        
        # Si estamos usando FAISS, reconstruir usando la lista numpy
        if self.has_faiss and self._faiss_index is not None:
            self.build_index(self._numpy_embeddings)

    def total_items(self) -> int:
        """Devuelve el número total de elementos en el índice."""
        if self.has_faiss and self._faiss_index is not None:
            return self._faiss_index.ntotal
        return len(self._numpy_embeddings)
