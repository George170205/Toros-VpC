# Resumen técnico consolidado del proyecto

Sistema de identificación recurrente de visitantes mediante visión por computadora

Documento actualizado con diseño interno y diagramas ASCII


## 1. Objetivo del sistema

Desarrollar un sistema de visión por computadora capaz de registrar cuántas veces una misma persona visita un estadio mediante identificadores seudónimos persistentes como IND_000001. El objetivo inicial es reconocer recurrencia, no necesariamente conocer el nombre real.


## 2. Escenario físico y prototipo

La captura se realizará en un punto controlado de acceso. El prototipo inicial utilizará webcam, computadora y procesamiento local. La distancia inicial de referencia es aproximadamente 2 a 2.5 metros, pendiente de validación. Para ingresos nocturnos se contempla iluminación artificial controlada.


## 3. Tecnologías y componentes base

Python; OpenCV para captura y procesamiento; SCRFD para detección facial; ByteTrack para tracking; ArcFace para embeddings; InsightFace como framework de referencia; similitud coseno; SQLite para el prototipo; y FAISS o HNSW como alternativas de búsqueda vectorial a evaluar.


## 4. Pipeline general

Webcam → OpenCV → Detección facial → Tracking temporal → Control de calidad → Selección de muestras → ArcFace → Embeddings → Agregación de evidencia → Búsqueda vectorial → TOP-K → Verificación secundaria → Decision Engine → Registro.


## 5. Separación de identidades

Track_ID representa una trayectoria temporal dentro de una cámara. Person_ID representa una identidad persistente. Una misma persona puede recibir diferentes Track_ID en visitas distintas y mantener el mismo Person_ID.


## 6. TrackSession y Sample Collector

Cada Track_ID crea una sesión temporal con first_seen, last_seen, total_frames, valid_frames, muestras seleccionadas, estado e intentos de identificación. El Sample Collector conserva un número limitado de muestras de calidad y diversidad suficientes.


## 7. Control de calidad

Cada rostro se evalúa por calidad general, nitidez, iluminación, tamaño, pose, visibilidad y oclusiones. Las muestras insuficientes se descartan y los thresholds se determinarán experimentalmente.


## 8. Estrategia de múltiples muestras

No se procesa necesariamente cada frame. Se seleccionan muestras representativas para reducir costo computacional y mejorar la robustez. La sesión puede pasar a READY por evidencia suficiente o por timeout si existe evidencia mínima.


## 9. Máquina de estados

NEW → COLLECTING → READY → GENERATING_EMBEDDINGS → VALIDATING_EVIDENCE → STABLE/UNSTABLE → SEARCHING → VERIFYING → MATCH/AMBIGUOUS/UNKNOWN. La evidencia inestable puede regresar a COLLECTING.


## 10. Evidence Aggregator

Recibe múltiples embeddings de una misma TrackSession. Normaliza, mide consistencia interna, detecta outliers, calcula pesos, genera un embedding agregado y conserva evidencia individual para verificación.


## 11. Función compuesta de ponderación

Se adopta una función configurable: S = αQ + βSh + γB + δP + εF + ζD, donde Q es calidad general, Sh nitidez, B iluminación, P pose, F tamaño/calidad geométrica y D diversidad temporal. Los coeficientes se calibrarán experimentalmente y permanecerán configurables.


## 12. Agregación ponderada

Los pesos relativos se obtienen conceptualmente mediante w_i = S_i / ΣS. El embedding agregado es E_agg = normalize(Σ(w_i · E_i)). Esto evita que muestras de baja calidad influyan igual que las mejores.


## 13. Consistencia y outliers

Se compara cada embedding con el grupo de la misma TrackSession. Una muestra claramente inconsistente puede descartarse para evitar contaminar la representación agregada.


## 14. Representación de identidad

Cada PersonIdentity tendrá un embedding principal y un conjunto limitado de embeddings secundarios. El principal sirve para recuperación rápida y los secundarios para verificación detallada y conservación de diversidad.


## 15. Gestión de embeddings secundarios

Se establece un máximo configurable. Un nuevo embedding solo se conserva si aporta calidad o diversidad; si no hay espacio, puede reemplazar una representación débil o redundante.


## 16. Búsqueda en dos niveles

Nivel 1: el embedding agregado consulta el índice de embeddings principales y recupera TOP-K Person_ID. Nivel 2: únicamente esos candidatos se verifican contra sus embeddings secundarios.


## 17. CandidateEvidence

La evidencia por candidato incluirá person_id, rank, primary_similarity, similitudes individuales, máximo, media, mediana, consistency_score y final_candidate_score.


## 18. Decision Engine

La decisión no dependerá de una sola similitud. Considerará calidad, múltiples frames, consistencia, TOP-K, margen entre primer y segundo candidato y evidencia secundaria. Los estados principales son MATCH, AMBIGUOUS y UNKNOWN.


## 19. Registro de visitas

Una coincidencia confirmada genera un evento con fecha, hora, cámara y confianza. Tracking, zona o línea virtual y lógica de estado/cooldown evitarán múltiples conteos de la misma presencia.


## 20. Base de datos y búsqueda

SQLite será la base inicial para Persons, Visits y Embeddings. La base administra entidades y eventos; el índice vectorial administra recuperación por similitud. Esta separación permite sustituir tecnologías sin rediseñar la lógica.


## 21. Optimización y concurrencia

Se evitará generar embeddings en cada frame. La arquitectura podrá usar colas entre captura, detección, reconocimiento, decisión y persistencia. Multiprocessing, threading, batching y GPU se evaluarán mediante profiling.


## 22. Arquitectura modular

Componentes conceptuales: CameraManager, FaceDetector, Tracker, TrackSessionManager, FaceQualityAnalyzer, SampleCollector, EmbeddingGenerator, EvidenceAggregator, VectorSearchService, DecisionEngine, DatabaseService y Tests.


## 23. Métricas y experimentación

Se medirán True Positive, False Positive, False Negative, FAR, FRR, FPS, latencia, throughput y consumo de recursos. Los thresholds se calibrarán con comparaciones genuinas e impostoras.


## 24. Seguridad y privacidad

El prototipo deberá usar participantes voluntarios y entorno controlado. Se contemplan minimización de datos, identificadores seudónimos, control de acceso, protección de embeddings, retención, eliminación, auditoría y legislación aplicable.


## 25. Diagramas de arquitectura y flujo interno


### 25.1 Pipeline general

```
CAMERA
   |
   v
DETECTION
   |
   v
TRACKING
   |
   v
QUALITY ANALYSIS
   |
   v
SAMPLE COLLECTION
   |
   v
BEST FACE SAMPLES
   |
   v
ARCFACE
   |
   v
MULTIPLE EMBEDDINGS
   |
   v
EVIDENCE AGGREGATOR
   |
   v
PRIMARY VECTOR INDEX
   |
   v
TOP-K CANDIDATES
   |
   v
SECONDARY VERIFICATION
   |
   v
CANDIDATE EVIDENCE
   |
   v
DECISION ENGINE
   |
   v
PERSISTENCE
```


### 25.2 Evidence Aggregator

```
MULTIPLE EMBEDDINGS
        |
        v
+-------------------+
| NORMALIZATION     |
+---------+---------+
          |
          v
+-------------------+
| CONSISTENCY CHECK |
+---------+---------+
          |
     +----+----+
     |         |
  STABLE    OUTLIER
     |         |
     |      DISCARD
     v
+-------------------+
| COMPOSITE WEIGHTS |
+---------+---------+
          |
          v
+-------------------+
| WEIGHTED          |
| AGGREGATION       |
+---------+---------+
          |
          v
   AGGREGATED EMBEDDING
          |
          v
    IDENTITY EVIDENCE
```


### 25.3 Identidad persistente

```
PersonIdentity
|
+-- person_id
|
+-- PRIMARY EMBEDDING
|    +-- vector
|    +-- quality_score
|    +-- created_at
|    +-- updated_at
|
+-- SECONDARY EMBEDDINGS[]
|    +-- embedding_id
|    +-- vector
|    +-- quality_score
|    +-- pose_metadata
|    +-- created_at
|
+-- first_seen
+-- last_seen
+-- total_visits
```


### 25.4 Búsqueda en dos niveles

```
IdentityEvidence
       |
       v
AggregatedEmbedding
       |
       v
+--------------------------+
| PRIMARY VECTOR INDEX     |
| Fast retrieval           |
+------------+-------------+
             |
             v
           TOP-K
             |
             v
 Candidate Person_IDs
             |
             v
+--------------------------+
| SECONDARY VERIFICATION   |
| Detailed comparison      |
+------------+-------------+
             |
             v
   CandidateEvidence
             |
             v
      DECISION ENGINE
```


### 25.5 Gestión de embeddings secundarios

```
NEW EMBEDDING
      |
      v
+---------------------------+
| Adds quality or diversity?|
+-------------+-------------+
              |
         +----+----+
         |         |
        NO        YES
         |         |
      DISCARD      v
              +-------------------+
              | Space available?  |
              +----+----------+---+
                   |          |
                  YES         NO
                   |          |
                 STORE    REPLACE
                          weakest /
                          redundant
```


## 26. Parámetros pendientes de validación experimental

| Parámetro | Estado |
|---|---|
| Distancia final de cámara | Pendiente de experimentación |
| Resolución | Pendiente de experimentación |
| FPS objetivo | Pendiente de experimentación |
| Tamaño mínimo de rostro | Pendiente de experimentación |
| Threshold de similitud | Pendiente de experimentación |
| Threshold de calidad | Pendiente de experimentación |
| Número óptimo de muestras | Pendiente de experimentación |
| TOP-K óptimo | Pendiente de experimentación |
| Margen entre candidatos | Pendiente de experimentación |
| Número máximo de secundarios | Pendiente de experimentación |
| Coeficientes α, β, γ, δ, ε, ζ | Pendiente de experimentación |
| FAISS vs HNSW | Pendiente de experimentación |
| CPU vs GPU | Pendiente de experimentación |
| Hardware definitivo | Pendiente de experimentación |


## 27. Próximo diseño

El siguiente órgano interno será el Vector Search Service: estructura del índice principal, relación vector-Person_ID, recuperación TOP-K, candidatos insuficientes, recuperación de embeddings secundarios y construcción formal de CandidateEvidence.


## 28. Secondary Verification Service

Este servicio actúa como puente entre la recuperación vectorial y la decisión final. Recibe IdentityEvidence y CandidateMatch[], recupera el embedding principal y los embeddings secundarios de cada candidato, construye una matriz de similitud, extrae métricas robustas y produce CandidateEvidence[]. No decide la identidad; únicamente cuantifica la fuerza y consistencia de la evidencia biométrica.

```
IdentityEvidence + CandidateMatch[]
                |
                v
+--------------------------------+
| SECONDARY VERIFICATION SERVICE |
+---------------+----------------+
                |
                v
       CandidateEvidence[]
                |
                v
         DECISION ENGINE
```


### 28.1 Matriz de similitud

Para M embeddings actuales y N embeddings del perfil candidato se construye una matriz M x N mediante similitud coseno. El embedding principal también participa en la verificación.

```
              Candidate IND_0014
           P      S1     S2     S3
       +------+------+------+------+
 C1    | .93  | .91  | .89  | .90  |
 C2    | .92  | .90  | .88  | .91  |
 C3    | .90  | .89  | .87  | .88  |
       +------+------+------+------+
```


### 28.2 Métricas de verificación

best_similarity = max(M) - Mayor similitud individual observada; sirve como evidencia auxiliar, nunca como criterio único de MATCH.

best_per_sample[i] = max(M[i, :]) - Mejor correspondencia histórica para cada muestra actual.

mean_top_similarity = mean(best_per_sample) - Promedio de las mejores correspondencias por muestra.

median_top_similarity = median(best_per_sample) - Medida robusta frente a valores extremos.

consistency_score = f(varianza, desviación estándar, rango) - Mide el acuerdo entre las muestras actuales. Se normalizará a [0,1] y su fórmula exacta se calibrará.

positive_match_ratio = positive_samples / total_samples - Fracción de muestras que superan un umbral configurable de evidencia positiva.

primary_similarity = sim(E_aggregated, E_primary) - Conserva la evidencia de la recuperación principal.

margin = best_candidate_score - second_best_score - Separación entre los dos candidatos finales más fuertes; un margen pequeño indica ambigüedad.


### 28.3 Verification Score compuesto

Se deja preparada una función compuesta y configurable para resumir la verificación. No sustituye las barreras del Decision Engine.

V = aP + bM + cMd + dC + eR + fQ

P = primary_similarity; M = mean_top_similarity; Md = median_top_similarity; C = consistency_score; R = positive_match_ratio; Q = sample_quality. Los coeficientes a, b, c, d, e, f se normalizan para sumar 1 y se calibrarán experimentalmente.


### 28.4 CandidateEvidence

```
CandidateEvidence
|
+-- person_id
+-- rank
+-- primary_similarity
+-- similarity_matrix
+-- best_similarity
+-- best_per_sample[]
+-- mean_top_similarity
+-- median_top_similarity
+-- consistency_score
+-- positive_match_ratio
+-- average_sample_quality
+-- verification_score
+-- metadata
```

```
CandidateMatch + IdentityEvidence
             |
             v
+--------------------------+
| LOAD CANDIDATE PROFILE   |
+------------+-------------+
             v
+--------------------------+
| PRIMARY + SECONDARIES    |
+------------+-------------+
             v
+--------------------------+
| SIMILARITY MATRIX        |
+------------+-------------+
             v
+--------------------------+
| FEATURE EXTRACTION       |
| best / mean / median     |
| consistency / ratio      |
+------------+-------------+
             v
+--------------------------+
| VERIFICATION SCORE       |
+------------+-------------+
             v
      CandidateEvidence
```


## 29. Decision Engine jerárquico

Se adopta como base un motor de decisión jerárquico por barreras de seguridad. El objetivo es reducir falsos positivos evitando que una única puntuación determine la identidad. La función compuesta de VerificationScore resume evidencia, mientras el Decision Engine aplica reglas independientes de seguridad y contexto temporal.

```
CandidateEvidence[]
        +
 IdentityEvidence
        +
   TrackSession
        |
        v
+----------------------+
| 1. EVIDENCE VALIDATION|
+----------+-----------+
           v
+----------------------+
| 2. CANDIDATE STRENGTH|
+----------+-----------+
           v
+----------------------+
| 3. MARGIN CHECK      |
+----------+-----------+
           v
+----------------------+
| 4. CONSISTENCY CHECK |
+----------+-----------+
           v
+----------------------+
| 5. QUALITY CHECK     |
+----------+-----------+
           v
+----------------------+
| 6. TEMPORAL CONFIRM. |
+----------+-----------+
           v
    +------+------+
    |      |      |
 MATCH AMBIGUOUS UNKNOWN
```


### 29.1 Gate 1 - Validación mínima de evidencia

Verifica número suficiente de muestras, calidad mínima, consistencia interna, ausencia de errores graves y validez del embedding agregado. Si la evidencia es insuficiente, el resultado es AMBIGUOUS y la sesión continúa recolectando cuando sea posible.


### 29.2 Gate 2 - Fuerza del mejor candidato

Selecciona el candidato con mayor verification_score y exige que supere MIN_VERIFICATION_SCORE. El valor será configurable y experimental. Si ningún candidato es suficientemente fuerte, se clasifica como UNKNOWN.


### 29.3 Gate 3 - Margen contra el segundo candidato

Calcula margin = best_score - second_best_score. Si el margen es menor que MIN_CANDIDATE_MARGIN, la decisión se considera AMBIGUOUS aunque ambos scores sean altos.


### 29.4 Gate 4 - Consistencia del candidato

Evalúa consistency_score, positive_match_ratio, mean_top_similarity y median_top_similarity. Un score global alto acompañado de evidencia secundaria contradictoria no debe producir MATCH.


### 29.5 Gate 5 - Calidad de evidencia actual

Evalúa la calidad de las muestras que originaron la identificación. Una similitud alta con evidencia de baja calidad se trata con mayor cautela y puede resultar en AMBIGUOUS.


### 29.6 Gate 6 - Confirmación temporal

Aprovecha el video para comprobar que el mismo Person_ID se mantiene en ventanas temporales sucesivas. La confirmación temporal requerida será configurable. Resultados contradictorios fuerzan recolección adicional o AMBIGUOUS.

```
Window A -> IND_0014
Window B -> IND_0014
Window C -> IND_0014
          |
          v
 TEMPORAL CONSISTENCY HIGH

Window A -> IND_0014
Window B -> IND_0087
Window C -> IND_0014
          |
          v
   COLLECT MORE / AMBIGUOUS
```


### 29.7 Regla jerárquica conceptual

```
if evidence_invalid:
    decision = AMBIGUOUS
elif best_candidate_too_weak:
    decision = UNKNOWN
elif margin_too_small:
    decision = AMBIGUOUS
elif candidate_inconsistent:
    decision = AMBIGUOUS
elif sample_quality_too_low:
    decision = AMBIGUOUS
elif temporal_confirmation_insufficient:
    decision = AMBIGUOUS
else:
    decision = MATCH
```


### 29.8 Creación segura de una nueva identidad

UNKNOWN no implica crear automáticamente un Person_ID. La creación de una nueva identidad requiere evidencia actual de buena calidad, consistencia interna suficiente y descarte razonable de candidatos existentes. Si la evidencia es pobre, se registra el resultado sin crear un perfil biométrico nuevo.

```
UNKNOWN
   |
   v
+---------------------------+
| Good current evidence?    |--NO--> DO NOT CREATE
+-------------+-------------+
              | YES
              v
+---------------------------+
| Internally consistent?    |--NO--> DO NOT CREATE
+-------------+-------------+
              | YES
              v
+---------------------------+
| Existing candidates       |
| reasonably discarded?     |--NO--> AMBIGUOUS
+-------------+-------------+
              | YES
              v
       CREATE NEW IDENTITY
```


### 29.9 DecisionResult

```
DecisionResult
|
+-- decision
|    +-- MATCH
|    +-- AMBIGUOUS
|    +-- UNKNOWN
|
+-- person_id (nullable)
+-- decision_confidence
+-- best_candidate_score
+-- second_candidate_score
+-- margin
+-- quality_score
+-- consistency_score
+-- positive_match_ratio
+-- temporal_confirmation
+-- reason_codes[]
+-- timestamp
```

Reason codes propuestos: MATCH_HIGH_CONFIDENCE, AMBIGUOUS_LOW_MARGIN, AMBIGUOUS_LOW_QUALITY, AMBIGUOUS_INCONSISTENT_EVIDENCE, AMBIGUOUS_TEMPORAL_CONFLICT, UNKNOWN_NO_STRONG_CANDIDATE, UNKNOWN_INSUFFICIENT_ENROLLMENT_EVIDENCE. Estos códigos permiten auditoría, pruebas y análisis de errores.


### 29.10 Principio de diseño

No se utilizará un único confidence como criterio mágico. verification_score, margin, consistency, quality y temporal_confirmation permanecen disponibles como señales independientes. decision_confidence puede calcularse para interfaz y logging, pero no sustituye las reglas jerárquicas.


## 30. Flujo interno consolidado actualizado

```
CAMERA
  |
  v
DETECTION (SCRFD)
  |
  v
TRACKING (ByteTrack)
  |
  v
TRACK SESSION
  |
  v
QUALITY ANALYSIS
  |
  v
SAMPLE COLLECTOR
  |
  v
BEST FACE SAMPLES
  |
  v
ARCFACE
  |
  v
MULTIPLE EMBEDDINGS
  |
  v
COMPOSITE WEIGHT STRATEGY
  |
  v
OUTLIER / CONSISTENCY CHECK
  |
  v
WEIGHTED AGGREGATION
  |
  v
IDENTITY EVIDENCE
  |
  v
PRIMARY VECTOR INDEX
  |
  v
TOP-K + CANDIDATE WINDOW
  |
  v
SECONDARY VERIFICATION
  |
  v
CANDIDATE EVIDENCE[]
  |
  v
HIERARCHICAL DECISION ENGINE
  |
  +------------+------------+
  |            |            |
MATCH      AMBIGUOUS      UNKNOWN
  |            |            |
  v            v            v
REGISTER    COLLECT      SAFE NEW-ID
 VISIT       MORE         EVALUATION
```


## 31. Parámetros nuevos pendientes de calibración

| Parámetro | Estado |
|---|---|
| Coeficientes a,b,c,d,e,f del VerificationScore | Configurable; pendiente de calibración experimental |
| Umbral de positive match | Configurable; pendiente de calibración experimental |
| Fórmula/normalización de consistency_score | Configurable; pendiente de calibración experimental |
| MIN_VERIFICATION_SCORE | Configurable; pendiente de calibración experimental |
| MIN_CANDIDATE_MARGIN | Configurable; pendiente de calibración experimental |
| Calidad mínima para decisión | Configurable; pendiente de calibración experimental |
| Número de ventanas temporales | Configurable; pendiente de calibración experimental |
| Regla de temporal_confirmation | Configurable; pendiente de calibración experimental |
| Criterios mínimos para crear nueva identidad | Configurable; pendiente de calibración experimental |


## 32. Estado del diseño interno

Con esta actualización quedan definidos a nivel arquitectónico el flujo de evidencia, búsqueda primaria, verificación secundaria y motor jerárquico de decisión. Permanecen pendientes la calibración numérica, el esquema SQL definitivo, contratos/interfaces de implementación, concurrencia detallada, logging, manejo de errores y pruebas.


## 31. Persistencia posterior a la decisión y modelo de datos v1

Esta sección formaliza qué ocurre después del Decision Engine y traduce el flujo operativo del sistema a una estrategia de persistencia consistente, auditable y reconstruible. La decisión de diseño central es que la base de datos será la fuente de verdad del sistema, mientras que el índice vectorial será una estructura derivada orientada exclusivamente a acelerar búsquedas.


### 31.1 Flujo posterior al Decision Engine

El resultado del Decision Engine puede ser MATCH, AMBIGUOUS o UNKNOWN. Cada estado activa una política diferente y evita que la persistencia mezcle reconocimiento, actualización de identidad y registro de visita en una sola operación monolítica.

```
                DecisionResult
                      |
          +-----------+-----------+
          |           |           |
        MATCH      AMBIGUOUS    UNKNOWN
          |           |           |
          v           v           v
+----------------+  NO FINAL  +-------------------+
| IdentityUpdate |  PERSIST   | NewIdentityPolicy |
| Policy         |            +---------+---------+
+-------+--------+                      |
        |                          +-----+-----+
        v                          |           |
 Register Visit                  REJECT      CREATE
        |                          |           |
        v                          |           v
 Persist Decision                 |      NEW IDENTITY
        |                          |           |
        v                          |           v
 Maybe Update Profile             |      REGISTER VISIT
        |                          |           |
        v                          |           v
 Sync Vector Index <--------------+----- SYNC INDEX
```


### 31.2 Caso MATCH: registro y actualización segura

Un MATCH confirmado implica dos responsabilidades separadas: registrar una visita y evaluar si la nueva evidencia biométrica merece modificar el perfil histórico. La identidad no debe actualizarse automáticamente por el mero hecho de existir una coincidencia, ya que esto podría contaminar progresivamente el perfil con evidencia dudosa.

```
MATCH
  |
  +--> REGISTER VISIT
  |
  +--> IDENTITY UPDATE POLICY
          |
          v
   High confidence?
          |
   High sample quality?
          |
   Evidence consistent?
          |
   Adds quality/diversity?
      +---+---+
      |       |
     NO      YES
      |       |
  KEEP       UPDATE
 PROFILE     PROFILE
```


### 31.3 IdentityUpdateDecision

La decisión de actualizar un perfil debe estar representada explícitamente para facilitar auditoría y pruebas.

```
IdentityUpdateDecision
|
+-- should_update
+-- update_primary
+-- add_secondary
+-- replace_secondary_id
+-- confidence
+-- reason_codes[]
```

Ejemplos de reason_codes: PROFILE_UPDATE_HIGH_QUALITY, SECONDARY_ADDED_FOR_DIVERSITY, SECONDARY_REPLACED_LOW_QUALITY, NO_UPDATE_REDUNDANT_SAMPLE y NO_UPDATE_LOW_CONFIDENCE.


### 31.4 Actualización conservadora del embedding principal

El embedding principal es la representación utilizada en la primera fase de recuperación y, por tanto, debe ser estable. Se propone una política conservadora: solo podrá modificarse con un MATCH de muy alta confianza, evidencia de alta calidad, fuerte consistencia con el perfil histórico y una mejora demostrable.

Como alternativa al reemplazo brusco, se deja preparada una actualización suave:

P_new = normalize(λ · P_old + (1 - λ) · E_current)

donde P_old es el embedding principal histórico, E_current es el embedding agregado de la visita actual y 0 < λ < 1. Un λ alto conserva mayor peso histórico y reduce cambios abruptos. El valor definitivo de λ será experimental y versionado en configuración.


### 31.5 Gestión de embeddings secundarios

Los embeddings secundarios pueden actualizarse de forma más flexible que el principal, siempre preservando un número máximo configurable y buscando diversidad útil. Una nueva muestra redundante se descarta; una muestra de alta calidad que represente una pose, iluminación o condición útil puede añadirse o reemplazar una secundaria más débil.

```
NEW SECONDARY CANDIDATE
        |
        v
   QUALITY CHECK
        |
        v
SIMILARITY TO EXISTING
        |
   +----+--------------------+
   |                         |
REDUNDANT                 DIVERSE / USEFUL
   |                         |
DISCARD                 Space available?
                         +----+----+
                         |         |
                        YES       NO
                         |         |
                        ADD     REPLACE
                                weakest /
                                redundant
```


### 31.6 Caso UNKNOWN: NewIdentityPolicy

UNKNOWN no equivale automáticamente a crear una nueva persona. Antes de crear una identidad deben cumplirse condiciones mínimas: evidencia de calidad, consistencia interna, ausencia de candidatos razonablemente cercanos y confirmación de que la TrackSession representa un evento de entrada válido.

```
UNKNOWN
   |
   v
High-quality evidence?
   |
Internally consistent?
   |
No plausible existing candidate?
   |
Valid track + entry event?
   |
 +--+--+
 |     |
NO    YES
 |     |
NO    CREATE
ID    IDENTITY
```


### 31.7 NewIdentityDecision

```
NewIdentityDecision
|
+-- should_create
+-- initial_primary_embedding
+-- initial_secondary_embeddings[]
+-- quality_score
+-- evidence_id
+-- reason_codes[]
```


### 31.8 Creación transaccional de una identidad

La creación de una nueva persona debe ser consistente. No es aceptable crear la entidad sin sus embeddings o registrar una visita que apunte a una identidad incompleta. Por ello las operaciones persistentes se ejecutarán dentro de una transacción.

```
UNKNOWN VALIDATED
      |
      v
Generate Person_ID
      |
      v
BEGIN TRANSACTION
      |
      +--> INSERT person
      +--> INSERT primary embedding
      +--> INSERT secondary embeddings
      +--> INSERT identification decision
      +--> INSERT first visit
      |
      v
COMMIT
      |
      v
SYNC PRIMARY VECTOR INDEX
```


### 31.9 Transacciones y consistencia

La base de datos será la fuente de verdad. Las mutaciones relacionadas con una decisión se agruparán en transacciones: crear o actualizar Person, insertar embeddings, registrar Visit y persistir la decisión de auditoría. Ante un error previo al COMMIT se ejecutará ROLLBACK. El índice vectorial se sincronizará después del commit y podrá reconstruirse desde la base si se pierde o corrompe.


### 31.10 VisitService e idempotencia

El registro de visita debe ser idempotente. El mismo evento de entrada no debe generar dos visitas aunque el mensaje se procese dos veces. Se utilizará un event_id único, generado al confirmar el evento de entrada, y una restricción UNIQUE en base de datos.

```
VisitEvent
|
+-- event_id (UNIQUE)
+-- person_id
+-- track_id
+-- camera_id
+-- timestamp
+-- decision_id
+-- confidence
+-- entry_zone
+-- metadata
```


### 31.11 Auditoría de decisiones

Se adopta formalmente una entidad de auditoría para calibración, debugging y trazabilidad. Esta tabla permitirá responder por qué el sistema tomó una decisión y analizar posteriormente falsos positivos, falsos negativos, casos ambiguos y efecto de diferentes configuraciones.

```
IdentificationDecisionRecord
|
+-- decision_id
+-- track_id
+-- camera_id
+-- selected_person_id (nullable)
+-- decision_type
+-- decision_confidence
+-- best_candidate_score
+-- second_candidate_score
+-- margin
+-- average_quality
+-- consistency_score
+-- positive_match_ratio
+-- temporal_confirmation
+-- reason_codes[]
+-- decision_engine_version
+-- configuration_id
+-- created_at
```


## 32. Modelo relacional definitivo v1

El esquema v1 se compone de siete entidades persistentes principales. Una octava entidad, face_samples, será opcional y solo se habilitará en escenarios de evaluación/debug con consentimiento y política de retención.


### 32.1 Entidades núcleo

- persons
- embeddings
- visits
- identification_decisions
- cameras
- system_configurations
- track_session_logs
- face_samples (opcional, modo evaluación/debug)

### 32.2 persons

```
persons
+-------------------------+
| person_id PK            |
| public_code UNIQUE      |
| first_seen              |
| last_seen               |
| total_visits            |
| status                  |
| created_at              |
| updated_at              |
+-------------------------+
```

Se recomienda separar el identificador interno (UUID) del código legible IND_000014. El UUID protege las relaciones internas frente a cambios de formato, mientras que public_code facilita depuración y visualización. Status podrá admitir ACTIVE, INACTIVE, MERGED, DELETED y REVIEW para manejar el ciclo de vida de identidades y posibles duplicados.


### 32.3 embeddings

```
embeddings
+-----------------------------+
| embedding_id PK             |
| person_id FK                |
| embedding_type              |
| vector BLOB                 |
| dimension                   |
| dtype                       |
| model_name                  |
| model_version               |
| normalization              |
| quality_score               |
| pose_yaw                    |
| pose_pitch                  |
| pose_roll                   |
| sharpness_score             |
| brightness_score            |
| diversity_score             |
| active                      |
| created_at                  |
| updated_at                  |
+-----------------------------+
```

embedding_type será PRIMARY o SECONDARY. La lógica deberá garantizar un solo PRIMARY activo por persona. En SQLite se almacenará inicialmente el vector como BLOB float32 por ser más compacto y eficiente que JSON. Se guardarán dimensión, dtype, modelo y versión para impedir mezclar embeddings incompatibles si el modelo cambia en el futuro.


### 32.4 visits

```
visits
+-------------------------+
| visit_id PK             |
| event_id UNIQUE         |
| person_id FK            |
| camera_id FK            |
| decision_id FK          |
| track_id                |
| entry_timestamp         |
| confidence              |
| entry_zone              |
| created_at              |
+-------------------------+
```

visits es la fuente histórica de recurrencia. persons.total_visits se mantiene como una denormalización útil para lectura rápida, pero debe actualizarse en la misma transacción que la visita para conservar consistencia.


### 32.5 identification_decisions

```
identification_decisions
+-----------------------------------+
| decision_id PK                    |
| track_id                          |
| camera_id FK                      |
| selected_person_id FK nullable    |
| decision_type                     |
| decision_confidence               |
| best_candidate_score              |
| second_candidate_score            |
| margin                            |
| average_quality                   |
| consistency_score                 |
| positive_match_ratio              |
| temporal_confirmation             |
| reason_codes                      |
| decision_engine_version           |
| configuration_id FK               |
| created_at                        |
+-----------------------------------+
```

selected_person_id será nullable para AMBIGUOUS y UNKNOWN. reason_codes podrá persistirse como JSON serializado en SQLite y evolucionar a JSONB si en una fase posterior se migra a PostgreSQL.


### 32.6 cameras

```
cameras
+----------------------+
| camera_id PK         |
| name                 |
| location             |
| entry_zone           |
| resolution_width     |
| resolution_height    |
| fps_target           |
| status               |
| created_at           |
+----------------------+
```

Aunque el prototipo use una webcam, la entidad cameras evita hardcodear una cámara única y prepara la arquitectura para múltiples accesos.


### 32.7 track_session_logs

```
track_session_logs
+--------------------------+
| session_id PK            |
| track_id                 |
| camera_id FK             |
| first_seen               |
| last_seen                |
| total_frames             |
| valid_samples            |
| rejected_samples         |
| final_state              |
| decision_id FK nullable  |
+--------------------------+
```

La TrackSession completa vive en memoria para evitar escrituras por frame. Al terminar, se persiste solo un resumen útil para debugging, profiling y análisis de sesiones fallidas.


### 32.8 system_configurations

```
system_configurations
+-------------------------+
| configuration_id PK     |
| version                 |
| parameters_json         |
| created_at              |
| active                  |
+-------------------------+
```

Esta entidad versiona parámetros como MIN_QUALITY, TOP_K, CANDIDATE_DELTA, MIN_VERIFICATION_SCORE, MIN_MARGIN, MIN_CONSISTENCY, λ de actualización y coeficientes de ponderación. Cada decisión almacena configuration_id, lo cual permite reproducir y comparar experimentos.


### 32.9 face_samples opcional

Por defecto no se almacenarán imágenes faciales. En modo de evaluación/debug autorizado puede existir una tabla temporal con sample_id, track_session_id, file_reference, quality_score y retention_until. Esta información deberá estar sujeta a consentimiento y política de eliminación.


## 33. Relaciones, índices y fuente de verdad


### 33.1 Relaciones principales

```
                         +------------------------+
                         | SYSTEM_CONFIGURATIONS  |
                         +-----------+------------+
                                     |
                                     v
                         +------------------------+
                         | IDENTIFICATION_DECISIONS|
                         +----+---------------+---+
                              |               |
                              v               v
                         +---------+      +---------+
                         | CAMERAS |      | PERSONS |
                         +----+----+      +----+----+
                              |                |
                              |                +-------------------+
                              |                                    |
                              v                                    v
                         +---------+                         +------------+
                         | VISITS  |                         | EMBEDDINGS |
                         +---------+                         +------------+

TRACK_SESSION_LOGS ----------------> IDENTIFICATION_DECISIONS
```

Cardinalidades principales: PERSONS 1:N VISITS; PERSONS 1:N EMBEDDINGS; CAMERAS 1:N VISITS; CAMERAS 1:N IDENTIFICATION_DECISIONS; PERSONS 0..1:N IDENTIFICATION_DECISIONS. Una decisión puede no seleccionar persona.


### 33.2 Índices SQL iniciales

- INDEX visits(person_id)
- INDEX visits(entry_timestamp)
- INDEX embeddings(person_id)
- INDEX embeddings(embedding_type)
- INDEX identification_decisions(camera_id)
- INDEX identification_decisions(created_at)
- INDEX identification_decisions(decision_type)
- UNIQUE visits(event_id)
- UNIQUE persons(public_code)
Los índices priorizan las consultas operativas más frecuentes: historial por persona, visitas por fecha, embeddings de una identidad y análisis de decisiones por cámara, fecha o tipo.


### 33.3 Base de datos vs índice vectorial

```
                 DATABASE
              SOURCE OF TRUTH
                    |
                    v
              PersonIdentity
                    |
                    v
             IndexSyncService
                    |
                    v
              VECTOR INDEX
              FAST RETRIEVAL
```

El índice vectorial no será fuente de verdad. Si se elimina o corrompe, el sistema podrá reconstruirlo leyendo los PRIMARY embeddings activos de la base de datos.


## 34. Flujos transaccionales


### 34.1 MATCH

```
DecisionEngine
      |
      v
MATCH person_id
      |
      v
BEGIN TRANSACTION
      |
      +--> INSERT identification_decision
      +--> INSERT visit
      +--> UPDATE persons.last_seen
      +--> UPDATE persons.total_visits
      +--> optional embedding updates
      |
      v
COMMIT
      |
      v
IndexSyncService
```


### 34.2 UNKNOWN validado

```
UNKNOWN
   |
   v
NewIdentityPolicy
   |
CREATE = TRUE
   |
   v
BEGIN TRANSACTION
   |
   +--> INSERT person
   +--> INSERT primary embedding
   +--> INSERT secondary embeddings
   +--> INSERT identification_decision
   +--> INSERT first visit
   |
   v
COMMIT
   |
   v
ADD PRIMARY TO VECTOR INDEX
```


### 34.3 AMBIGUOUS

```
AMBIGUOUS
    |
    v
INSERT identification_decision
    |
    +--> NO VISIT
    +--> NO NEW PERSON
    +--> NO PROFILE UPDATE
    |
    v
COLLECT MORE EVIDENCE
    |
    v
NEW DecisionResult
```

Puede existir más de una decisión asociada a una misma sesión. Esto permite analizar cómo evoluciona la evidencia hasta llegar a una resolución final o a una sesión expirada.


## 35. Qué vive en memoria y qué persiste

| Dato/objeto | Memoria | Persistencia |
|---|---|---|
| Frame | Sí | No |
| FaceDetection | Sí | No |
| SimilarityMatrix | Sí | No |
| TrackSession completa | Sí | No |
| TrackSession resumen | No | Sí |
| IdentityEvidence temporal | Sí | No |
| CandidateEvidence completo | Sí | Solo métricas resumidas |
| PersonIdentity | Cache opcional | Sí |
| Embeddings activos | Cache/índice | Sí |
| Visit | No | Sí |
| IdentificationDecision | No | Sí |
| Configuration | Cache | Sí |

Esta separación evita almacenar información de alto volumen o poco valor histórico, reduce superficie de privacidad y conserva únicamente datos necesarios para operación, calibración y auditoría.


## 36. Justificación consolidada de las decisiones de persistencia

Base de datos como fuente de verdad: Permite reconstruir el índice vectorial, evita inconsistencias entre motores y centraliza la auditoría.

Índice vectorial derivado: Optimiza búsqueda sin convertir una estructura de rendimiento en fuente autoritativa.

Auditoría de decisiones: Permite calibrar thresholds, investigar errores y reproducir decisiones con configuration_id y versionado.

Configuración versionada: Evita perder trazabilidad cuando cambien thresholds, pesos o reglas del Decision Engine.

UUID interno + código legible: Separa integridad referencial de presentación y evita acoplar la BD al formato IND_xxxxxx.

BLOB float32 para embeddings: Reduce espacio y costo de serialización frente a JSON en SQLite.

model_version en embeddings: Evita comparar accidentalmente representaciones generadas por modelos incompatibles.

TrackSession principalmente en memoria: Evita escrituras constantes durante video en tiempo real y conserva solo un resumen útil.

event_id único: Garantiza idempotencia y evita doble conteo de visitas.

Transacciones: Aseguran que persona, visita, decisión y embeddings no queden en estados parciales.

No persistir matrices ni todos los frames: Reduce almacenamiento, complejidad y exposición de datos biométricos innecesarios.

face_samples opcional: Permite debugging científico sin convertir la retención de imágenes en requisito operativo.


## 37. Estado actualizado y siguiente etapa

Con esta actualización quedan definidos el flujo posterior al Decision Engine, las políticas de actualización y creación de identidades, la idempotencia del registro de visitas, la auditoría, el esquema relacional v1, las relaciones, índices, transacciones, versionado y separación memoria/persistencia.

El siguiente bloque de diseño será formalizar los contratos exactos entre módulos y las clases Python: dataclasses, interfaces, tipos de retorno, errores, estados, eventos de queue y dependencias entre servicios. Esto permitirá transformar el diseño conceptual en una especificación de implementación directamente programable.


## 31. Modelos de dominio y contratos de datos v1

Esta sección traduce la arquitectura conceptual a objetos de software concretos. El objetivo es que cada módulo intercambie estructuras tipadas, predecibles y auditables, evitando diccionarios informales, dependencias implícitas y acoplamiento entre implementaciones. Los ejemplos se plantean con Python dataclasses, type hints, enums y NumPy.


### 31.1 Principios de diseño

- Separar datos de dominio de implementaciones concretas (OpenCV, SCRFD, ByteTrack, ArcFace, FAISS/HNSW).
- Usar tipos explícitos para que cada etapa conozca exactamente qué recibe y qué produce.
- Mantener inmutables los resultados de cálculo siempre que sea posible.
- Permitir mutabilidad únicamente en objetos que representan estado vivo, principalmente TrackSession.
- Separar objetos efímeros en memoria de entidades persistentes en base de datos.
- Versionar configuración y modelos para reproducibilidad experimental.
```
Camera -> Frame -> FaceDetection -> TrackedFace -> QualityResult
      -> FaceSample -> TrackSession -> EmbeddingEvidence[]
      -> IdentityEvidence -> CandidateMatch[] -> CandidateEvidence[]
      -> DecisionResult -> Persistencia / Actualizacion
```


### 31.2 Enums base

Los estados y categorías se representan con enums para evitar strings inconsistentes o valores mágicos dispersos en el código.

```
from enum import Enum, auto

class TrackState(Enum):
    NEW = auto()
    COLLECTING = auto()
    READY = auto()
    GENERATING_EMBEDDINGS = auto()
    VALIDATING_EVIDENCE = auto()
    SEARCHING = auto()
    VERIFYING = auto()
    IDENTIFIED = auto()
    AMBIGUOUS = auto()
    UNKNOWN = auto()
    EXPIRED = auto()
    FINISHED = auto()

class DecisionType(Enum):
    MATCH = auto()
    AMBIGUOUS = auto()
    UNKNOWN = auto()

class EmbeddingType(Enum):
    PRIMARY = auto()
    SECONDARY = auto()

class PersonStatus(Enum):
    ACTIVE = auto()
    INACTIVE = auto()
    REVIEW = auto()
    MERGED = auto()
    DELETED = auto()
```


### 31.3 Frame

Representa una captura cruda de una cámara. No contiene información de reconocimiento ni tracking; su responsabilidad es transportar imagen, tiempo y origen.

```
@dataclass(slots=True)
class Frame:
    frame_id: int
    camera_id: str
    timestamp: datetime
    image: np.ndarray
```


### 31.4 FaceDetection

Resultado de SCRFD. Describe un rostro localizado dentro de un frame sin asumir identidad.

```
@dataclass(frozen=True, slots=True)
class FaceDetection:
    detection_id: str
    frame_id: int
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    confidence: float
```


### 31.5 TrackedFace

Resultado de asociar detección y tracking. Introduce Track_ID como identidad temporal dentro de la cámara.

```
@dataclass(frozen=True, slots=True)
class TrackedFace:
    track_id: int
    frame_id: int
    camera_id: str
    timestamp: datetime
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    detection_confidence: float
```


### 31.6 FacePose y QualityResult

FacePose encapsula yaw, pitch y roll. QualityResult concentra todos los indicadores de calidad y evita transportar múltiples parámetros sueltos entre módulos.

```
@dataclass(frozen=True, slots=True)
class FacePose:
    yaw: float
    pitch: float
    roll: float

@dataclass(frozen=True, slots=True)
class QualityResult:
    overall_score: float
    sharpness_score: float
    brightness_score: float
    size_score: float
    pose_score: float
    visibility_score: float
    pose: FacePose
    is_acceptable: bool
```

Justificación: la decisión de aceptar una muestra debe ser reproducible. Conservar los sub-scores permite depurar por qué una muestra fue rechazada y recalibrar pesos más adelante.


### 31.7 FaceSample

Representa una muestra facial ya aprobada por el módulo de calidad. La imagen recortada existe en memoria para ArcFace; no implica persistencia permanente.

```
@dataclass(slots=True)
class FaceSample:
    sample_id: str
    track_id: int
    frame_id: int
    camera_id: str
    timestamp: datetime
    face_image: np.ndarray
    quality: QualityResult
```


### 31.8 TrackSession

Es el principal objeto mutable del pipeline. Modela el ciclo de vida de una persona mientras permanece dentro de la zona de captura.

```
@dataclass(slots=True)
class TrackSession:
    session_id: str
    track_id: int
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    state: TrackState = TrackState.NEW
    total_frames: int = 0
    valid_frames: int = 0
    rejected_frames: int = 0
    samples: list[FaceSample] = field(default_factory=list)
    identification_attempts: int = 0
    visit_registered: bool = False
```

```
TrackSession
    |
    +-- NEW
    +-- COLLECTING
    +-- READY
    +-- GENERATING_EMBEDDINGS
    +-- VALIDATING_EVIDENCE
    +-- SEARCHING
    +-- VERIFYING
    +-- IDENTIFIED / AMBIGUOUS / UNKNOWN
    +-- FINISHED
```

Regla de encapsulación: TrackSession debe modificarse a través de TrackSessionManager. Evitar cambios directos desde múltiples módulos reduce transiciones inválidas y condiciones de carrera.


### 31.9 EmbeddingEvidence

Representa un embedding individual generado a partir de una muestra, conservando contexto de calidad y pose.

```
@dataclass(frozen=True, slots=True)
class EmbeddingEvidence:
    sample_id: str
    track_id: int
    embedding: np.ndarray
    quality_score: float
    pose: FacePose
    weight: float = 0.0
    timestamp: datetime | None = None
```

Los embeddings se normalizarán para que ||E|| ≈ 1 cuando el modelo y la métrica seleccionada lo requieran. Esto facilita el uso coherente de similitud coseno.


### 31.10 IdentityEvidence

Salida del EvidenceAggregator. Es el expediente biométrico temporal de una TrackSession listo para búsqueda vectorial.

```
@dataclass(frozen=True, slots=True)
class IdentityEvidence:
    evidence_id: str
    session_id: str
    track_id: int
    camera_id: str
    embeddings: list[EmbeddingEvidence]
    aggregated_embedding: np.ndarray
    total_samples: int
    valid_samples: int
    rejected_samples: int
    average_quality: float
    embedding_consistency: float
    aggregation_weights: list[float]
    created_at: datetime
```


### 31.11 CandidateMatch y SimilarityMatrix

CandidateMatch representa recuperación rápida desde el índice principal. SimilarityMatrix encapsula la matriz utilizada durante la verificación secundaria.

```
@dataclass(frozen=True, slots=True)
class CandidateMatch:
    person_id: str
    rank: int
    primary_similarity: float
    index_score: float
    retrieved_at: datetime

@dataclass(frozen=True, slots=True)
class SimilarityMatrix:
    values: np.ndarray
    current_embedding_ids: list[str]
    candidate_embedding_ids: list[str]
```

```
                Candidate profile
              P      S1      S2      S3
          +------+------+------+------ +
Current C1| .93 | .91 | .89 | .90 |
        C2| .92 | .90 | .88 | .91 |
        C3| .90 | .89 | .87 | .88 |
          +------+------+------+------ +
```


### 31.12 CandidateEvidence

Es el contrato final entre SecondaryVerificationService y DecisionEngine. Contiene evidencia ya resumida y calculada, evitando que el DecisionEngine tenga que realizar comparaciones vectoriales.

```
@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    person_id: str
    rank: int
    primary_similarity: float
    similarity_matrix: SimilarityMatrix
    best_similarity: float
    best_per_sample: list[float]
    mean_top_similarity: float
    median_top_similarity: float
    consistency_score: float
    positive_match_ratio: float
    average_sample_quality: float
    verification_score: float
```


### 31.13 DecisionResult

Salida inmutable del DecisionEngine. Incluye no solo la decisión, sino evidencia resumida y códigos de razón para auditoría y calibración.

```
@dataclass(frozen=True, slots=True)
class DecisionResult:
    decision_id: str
    session_id: str
    track_id: int
    camera_id: str
    decision: DecisionType
    person_id: str | None
    confidence: float
    best_candidate_score: float | None
    second_candidate_score: float | None
    margin: float | None
    quality_score: float
    consistency_score: float
    positive_match_ratio: float
    temporal_confirmation: float
    reason_codes: list[str]
    timestamp: datetime
```


### 31.14 IdentityUpdateDecision y NewIdentityDecision

Estas estructuras separan la decisión biométrica de las políticas de persistencia. MATCH no implica necesariamente actualizar el perfil, y UNKNOWN no implica necesariamente crear una identidad.

```
@dataclass(frozen=True, slots=True)
class IdentityUpdateDecision:
    should_update: bool
    update_primary: bool
    add_secondary: bool
    replace_secondary_id: str | None
    confidence: float
    reason_codes: list[str]

@dataclass(frozen=True, slots=True)
class NewIdentityDecision:
    should_create: bool
    primary_embedding: np.ndarray | None
    secondary_embeddings: list[np.ndarray]
    quality_score: float
    evidence_id: str
    reason_codes: list[str]
```


### 31.15 VisitEvent

Evento idempotente enviado a VisitService cuando una identidad confirmada cruza la zona de entrada.

```
@dataclass(frozen=True, slots=True)
class VisitEvent:
    event_id: str
    person_id: str
    session_id: str
    track_id: int
    camera_id: str
    timestamp: datetime
    decision_id: str
    confidence: float
    entry_zone: str | None = None
```

event_id funciona como clave de idempotencia: si el mismo evento se procesa dos veces, la base de datos debe conservar una sola visita.


### 31.16 PersonIdentity y StoredEmbedding

PersonIdentity mantiene metadatos ligeros de la persona. Los vectores no se cargan automáticamente dentro de esta entidad; se obtienen mediante EmbeddingRepository para evitar cargas innecesarias de memoria.

```
@dataclass(slots=True)
class PersonIdentity:
    person_id: str
    public_code: str
    status: PersonStatus
    primary_embedding_id: str | None
    first_seen: datetime
    last_seen: datetime
    total_visits: int

@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    embedding_id: str
    person_id: str
    embedding_type: EmbeddingType
    vector: np.ndarray
    quality_score: float
    pose: FacePose | None
    model_name: str
    model_version: str
    dimension: int
    active: bool
    created_at: datetime
    updated_at: datetime
```


### 31.17 Camera y SystemConfiguration

La cámara y la configuración operativa también son modelos explícitos. Esto elimina valores hardcodeados y permite reproducir experimentos.

```
@dataclass(frozen=True, slots=True)
class Camera:
    camera_id: str
    name: str
    location: str | None
    entry_zone: str | None
    resolution_width: int
    resolution_height: int
    target_fps: float
    active: bool

@dataclass(frozen=True, slots=True)
class SystemConfiguration:
    configuration_id: str
    version: str
    min_quality: float
    top_k: int
    candidate_delta: float
    min_verification_score: float
    min_margin: float
    min_consistency: float
    max_secondary_embeddings: int
    primary_update_lambda: float
    created_at: datetime
    active: bool
```


### 31.18 Pesos configurables

Se modelan por separado los pesos de agregación de muestras y los pesos del VerificationScore. De esta forma pueden calibrarse independientemente.

```
@dataclass(frozen=True, slots=True)
class AggregationWeights:
    quality: float
    sharpness: float
    brightness: float
    pose: float
    face_size: float
    diversity: float

@dataclass(frozen=True, slots=True)
class VerificationWeights:
    primary_similarity: float
    mean_top_similarity: float
    median_top_similarity: float
    consistency: float
    positive_ratio: float
    sample_quality: float
```

Las fórmulas definidas anteriormente permanecen:

```
SampleScore:
S = alpha*Q + beta*Sh + gamma*B + delta*P + epsilon*F + zeta*D

Aggregation weight:
w_i = S_i / sum(S)

Aggregated embedding:
E_agg = normalize(sum(w_i * E_i))

Verification score:
V = alpha*P + beta*M + gamma*Md + delta*C + epsilon*R + zeta*Q
```


### 31.19 Mutabilidad e inmutabilidad

Los resultados calculados deben preferirse inmutables para impedir modificaciones silenciosas entre módulos. TrackSession permanece mutable porque representa estado vivo.

| Objeto | Mutabilidad recomendada | Justificación |
|---|---|---|
| Frame | Mutable/efímero | Transporta una matriz de imagen que no se persiste. |
| FaceDetection | Inmutable | Resultado de inferencia. |
| QualityResult | Inmutable | Resultado de cálculo reproducible. |
| EmbeddingEvidence | Inmutable | Evidencia biométrica individual. |
| IdentityEvidence | Inmutable | Resumen de evidencia de sesión. |
| CandidateMatch | Inmutable | Resultado del índice. |
| CandidateEvidence | Inmutable | Entrada del DecisionEngine. |
| DecisionResult | Inmutable | Resultado auditable. |
| TrackSession | Mutable controlada | Máquina de estados en ejecución. |
| PersonIdentity | Mutable mediante servicio | Metadatos persistentes actualizables. |


### 31.20 Objetos en memoria vs persistentes

| Principalmente en memoria | Persistentes |
|---|---|
| Frame | PersonIdentity |
| FaceDetection | StoredEmbedding |
| TrackedFace | Visit / VisitEvent materializado |
| QualityResult | IdentificationDecision audit |
| FaceSample | Camera |
| TrackSession | SystemConfiguration |
| EmbeddingEvidence | TrackSession summary opcional |
| IdentityEvidence | FaceSample solo en modo debug autorizado |
| SimilarityMatrix |  |
| CandidateMatch / CandidateEvidence |  |

Justificación: persistir únicamente información necesaria para operación, auditoría y calibración reduce almacenamiento, superficie de riesgo y complejidad de privacidad.


### 31.21 Organización de paquetes de dominio

Para el prototipo se recomienda una agrupación moderada, evitando tanto un único archivo gigante como decenas de archivos prematuros.

```
domain/
|-- enums.py
|-- vision_models.py
|-- identity_models.py
|-- decision_models.py
`-- persistence_models.py
```

Cuando la base de código crezca, estos archivos pueden dividirse por entidad sin modificar los contratos públicos.


### 31.22 Mapa de contratos de datos

```
+----------------+
| CameraManager  |
+-------+--------+
        | Frame
        v
+----------------+
| FaceDetector   |
+-------+--------+
        | FaceDetection[]
        v
+----------------+
| FaceTracker    |
+-------+--------+
        | TrackedFace[]
        v
+----------------------+
| TrackSessionManager  |
+----------+-----------+
           | TrackSession
           v
+----------------------+
| FaceQualityAnalyzer  |
+----------+-----------+
           | QualityResult / FaceSample
           v
+----------------------+
| EmbeddingGenerator   |
+----------+-----------+
           | EmbeddingEvidence[]
           v
+----------------------+
| EvidenceAggregator   |
+----------+-----------+
           | IdentityEvidence
           v
+----------------------+
| VectorSearchService  |
+----------+-----------+
           | CandidateMatch[]
           v
+------------------------------+
| SecondaryVerificationService |
+--------------+---------------+
               | CandidateEvidence[]
               v
+----------------------+
| DecisionEngine       |
+----------+-----------+
           | DecisionResult
           v
+----------------------+
| Identity / Visit /   |
| Persistence Services |
+----------------------+
```


### 31.23 Estado de esta capa

| Elemento | Estado |
|---|---|
| Enums principales | Definidos |
| Frame / detección / tracking | Definidos |
| QualityResult / FaceSample | Definidos |
| TrackSession | Definido |
| EmbeddingEvidence / IdentityEvidence | Definidos |
| CandidateMatch / CandidateEvidence | Definidos |
| DecisionResult | Definido |
| IdentityUpdateDecision / NewIdentityDecision | Definidos |
| VisitEvent | Definido |
| PersonIdentity / StoredEmbedding | Definidos |
| Camera / SystemConfiguration | Definidos |
| Pesos configurables | Definidos; valores pendientes de calibración |
| Interfaces/Protocols entre módulos | Siguiente etapa |


### 31.24 Siguiente etapa: interfaces y Protocols

Con los modelos de dominio definidos, el siguiente paso es formalizar las interfaces de cada servicio. Cada módulo deberá declarar métodos, entradas, salidas y posibles errores. Por ejemplo: FaceDetector.detect(frame) -> list[FaceDetection], VectorSearchService.search(evidence) -> list[CandidateMatch] y DecisionEngine.decide(evidence, candidates) -> DecisionResult. Esto permitirá sustituir implementaciones (por ejemplo FAISS por HNSW) sin modificar consumidores.


## 32. Interfaces y contratos entre módulos v1

Esta sección formaliza los contratos del pipeline de visión y del primer tramo del núcleo biométrico. Cada interfaz define responsabilidad, entrada, salida y errores, mientras las implementaciones concretas quedan desacopladas. Esta separación permite sustituir OpenCV, SCRFD, ByteTrack, ArcFace u otras tecnologías sin reescribir la lógica de negocio.


### 32.1 Regla arquitectónica

```
DOMINIO / CONTRATOS
        |
        +--> CameraSource
        +--> FaceDetector
        +--> FaceTracker
        +--> FaceQualityAnalyzer
        +--> FaceSampleExtractor
        +--> SampleCollector
        +--> TrackSessionManager
        +--> EmbeddingGenerator
        +--> WeightStrategy
        +--> EmbeddingOutlierDetector
        +--> EvidenceAggregator

IMPLEMENTACIONES
        |
        +--> OpenCVCameraSource
        +--> SCRFDFaceDetector
        +--> ByteTrackFaceTracker
        +--> ArcFaceEmbeddingGenerator
        +--> CompositeWeightStrategy
```

Justificación: el resto del sistema depende de capacidades, no de librerías específicas. Esto mejora testabilidad, mantenibilidad y evolución tecnológica.


### 32.2 CameraSource

Responsabilidad: producir frames desde una fuente de video sin conocer detección, tracking ni reconocimiento.

```
class CameraSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> Frame: ...
    def close(self) -> None: ...
    def is_open(self) -> bool: ...
```

| Entrada | Salida | Errores |
|---|---|---|
| Configuración/fuente de cámara | Frame | CameraOpenError; FrameReadError; CameraDisconnectedError |


### 32.3 FaceDetector

Responsabilidad: localizar rostros y landmarks. No asigna identidad persistente ni Track_ID.

```
class FaceDetector(Protocol):
    def detect(self, frame: Frame) -> list[FaceDetection]: ...
```

| Entrada | Salida | Errores |
|---|---|---|
| Frame | list[FaceDetection] | DetectionError; InvalidFrameError |


### 32.4 FaceTracker

Responsabilidad: asociar detecciones entre frames y crear identidad temporal. El Track_ID pertenece al seguimiento, no a la identidad biométrica.

```
class FaceTracker(Protocol):
    def update(
        self, frame: Frame, detections: list[FaceDetection]
    ) -> list[TrackedFace]: ...
    def reset(self) -> None: ...
```


### 32.5 TrackSessionManager

Responsabilidad: ser propietario del estado mutable de TrackSession y controlar transiciones. Evita modificaciones arbitrarias de estado desde otros módulos.

```
class TrackSessionManager(Protocol):
    def get_or_create(self, face: TrackedFace) -> TrackSession: ...
    def update_seen(self, session: TrackSession, face: TrackedFace) -> None: ...
    def mark_ready(self, session: TrackSession) -> None: ...
    def mark_finished(self, session: TrackSession) -> None: ...
    def expire_inactive_sessions(self, now: datetime) -> list[TrackSession]: ...
```


### 32.6 FaceQualityAnalyzer

Responsabilidad: medir la utilidad de una observación para reconocimiento. Evalúa nitidez, iluminación, tamaño, pose, visibilidad y, cuando exista, oclusión.

```
class FaceQualityAnalyzer(Protocol):
    def analyze(
        self, frame: Frame, face: TrackedFace
    ) -> QualityResult: ...
```


### 32.7 FaceSampleExtractor

Se separa evaluación de extracción: el analizador responde si la observación es adecuada; el extractor produce el recorte/alineación facial que consumirá ArcFace.

```
class FaceSampleExtractor(Protocol):
    def extract(
        self, frame: Frame, face: TrackedFace, quality: QualityResult
    ) -> FaceSample: ...
```


### 32.8 SampleCollector

Responsabilidad: conservar un conjunto limitado de muestras útiles, diversas y no redundantes dentro de la sesión.

```
class SampleCollector(Protocol):
    def add_sample(self, session: TrackSession, sample: FaceSample) -> None: ...
    def is_ready(self, session: TrackSession) -> bool: ...
    def get_best_samples(self, session: TrackSession) -> list[FaceSample]: ...
```

```
CameraSource -> Frame -> FaceDetector -> FaceDetection[]
     -> FaceTracker -> TrackedFace[] -> TrackSessionManager
     -> FaceQualityAnalyzer -> QualityResult
     -> FaceSampleExtractor -> FaceSample
     -> SampleCollector -> Best Face Samples
```


### 32.9 Excepciones del pipeline de visión

```
class VisionPipelineError(Exception): pass
class CameraError(VisionPipelineError): pass
class CameraOpenError(CameraError): pass
class FrameReadError(CameraError): pass
class CameraDisconnectedError(CameraError): pass
class InvalidFrameError(VisionPipelineError): pass
class DetectionError(VisionPipelineError): pass
class TrackingError(VisionPipelineError): pass
class QualityAnalysisError(VisionPipelineError): pass
class SampleCollectionError(VisionPipelineError): pass
```

La jerarquía permite capturar fallos globales del pipeline o reaccionar de forma específica a desconexiones, frames inválidos o errores de inferencia.


### 32.10 Límites de responsabilidad

- CameraSource no detecta rostros.
- FaceDetector no conoce Person_ID ni consulta la base de datos.
- FaceTracker solo genera identidad temporal.
- FaceQualityAnalyzer evalúa, pero no persiste ni genera embeddings.
- SampleCollector no ejecuta ArcFace.
- TrackSessionManager controla estado, pero no decide MATCH.

## 33. Contratos del núcleo biométrico - tramo de generación y agregación

Cuando una TrackSession alcanza READY, el sistema deja de depender principalmente de píxeles y comienza a trabajar con evidencia vectorial. Esta frontera debe soportar procesamiento por lotes, GPU y ejecución asíncrona.


### 33.1 EmbeddingGenerator

```
class EmbeddingGenerator(Protocol):
    def generate(self, sample: FaceSample) -> EmbeddingEvidence: ...
    def generate_batch(
        self, samples: list[FaceSample]
    ) -> list[EmbeddingEvidence]: ...
```

Implementación inicial: ArcFaceEmbeddingGenerator. generate_batch() se incluye desde v1 para permitir batching en GPU sin alterar contratos futuros.


### 33.2 Invariantes del embedding

- Dimensión igual a la esperada por la versión del modelo.
- Todos los componentes deben ser finitos.
- Normalización L2 cuando el modelo/configuración así lo requiera.
- model_name, model_version, dimension y normalización deben acompañar la evidencia.
```
||E||_2 = sqrt(sum(E_i^2)) ~= 1
```


### 33.3 WeightStrategy y AggregationContext

```
class WeightStrategy(Protocol):
    def calculate(
        self, sample: FaceSample, context: AggregationContext
    ) -> float: ...

@dataclass(frozen=True, slots=True)
class AggregationContext:
    sample_count: int
    temporal_span_ms: float
    mean_quality: float
    pose_diversity: float
    temporal_diversity: float
```

El contexto permite que el peso represente no solo calidad individual sino también diversidad respecto al conjunto observado.


### 33.4 CompositeWeightStrategy

```
S_i = alpha*Q_i + beta*Sh_i + gamma*B_i
    + delta*P_i + epsilon*F_i + zeta*D_i

alpha + beta + gamma + delta + epsilon + zeta ~= 1

w_i = S_i / sum_j(S_j)
```

Q=calidad general; Sh=nitidez; B=iluminación; P=pose; F=tamaño/calidad geométrica; D=diversidad. Los coeficientes permanecen versionados y se calibran experimentalmente.


### 33.5 EmbeddingOutlierDetector

```
class EmbeddingOutlierDetector(Protocol):
    def detect(
        self, embeddings: list[EmbeddingEvidence]
    ) -> set[str]: ...
```

Devuelve sample_id atípicos. La detección evita que una observación errónea contamine el embedding agregado y, posteriormente, una identidad persistente.


### 33.6 EvidenceAggregator

```
class EvidenceAggregator(Protocol):
    def aggregate(
        self,
        session: TrackSession,
        samples: list[FaceSample],
        embeddings: list[EmbeddingEvidence]
    ) -> IdentityEvidence: ...
```

```
EmbeddingEvidence[]
       |
       v
VALIDATION
       |
       v
NORMALIZATION
       |
       v
INTERNAL SIMILARITY
       |
       v
OUTLIER DETECTION
       |
       v
COMPOSITE WEIGHTS
       |
       v
WEIGHTED AGGREGATION
       |
       v
IdentityEvidence
```

```
sim(E_i, E_j) = E_i^T E_j       # embeddings L2-normalized

E_agg = normalize(sum_i(w_i * E_i))
```


### 33.7 Excepciones biométricas

```
class BiometricPipelineError(Exception): pass
class EmbeddingGenerationError(BiometricPipelineError): pass
class InvalidEmbeddingError(BiometricPipelineError): pass
class InsufficientEvidenceError(BiometricPipelineError): pass
class EvidenceAggregationError(BiometricPipelineError): pass
class UnstableEvidenceError(BiometricPipelineError): pass
```

UnstableEvidenceError puede representar una condición recuperable: la sesión puede volver a COLLECTING en lugar de tratarla como un fallo fatal.


## 34. Contratos de concurrencia iniciales

La captura no debe quedar bloqueada por ArcFace. Cuando una sesión alcanza READY se crea una tarea inmutable y se envía a una cola acotada de reconocimiento.

```
@dataclass(frozen=True, slots=True)
class RecognitionTask:
    task_id: str
    session_id: str
    track_id: int
    camera_id: str
    samples: tuple[FaceSample, ...]
    created_at: datetime

@dataclass(frozen=True, slots=True)
class RecognitionResult:
    task_id: str
    session_id: str
    identity_evidence: IdentityEvidence | None
    success: bool
    error_code: str | None
    completed_at: datetime
```

```
CAMERA LOOP
    |
    v
Detection / Tracking / Samples
    |
    +-- session not ready --> continue capture
    |
    +-- session READY
            |
            v
      RecognitionTask
            |
            v
   +-------------------+
   | BOUNDED QUEUE     |
   +---------+---------+
             |
             v
   +-------------------+
   | ARCFACE WORKER    |  GPU-bound when GPU is available
   +---------+---------+
             |
             v
      EvidenceAggregator  CPU / NumPy
             |
             v
       IdentityEvidence
```


### 34.1 Backpressure

La cola debe ser acotada (MAX_QUEUE_SIZE). Una cola infinita no incrementa capacidad: transforma una sobrecarga temporal en consumo creciente de RAM y latencia. La política concreta se decidirá mediante benchmarks.

- Evitar tareas duplicadas para una misma sesión.
- Priorizar evidencia de mayor calidad cuando sea necesario.
- Descartar tareas obsoletas antes que acumular trabajo sin utilidad.
- Registrar telemetría de saturación, profundidad de cola y latencia.
- No fijar aún cantidad de workers ni tamaño de batch: deben derivarse del hardware y profiling.

### 34.2 Mapa de costo computacional

| Componente | Naturaleza prevista | Observación |
|---|---|---|
| Camera/Detection/Tracking | CPU/GPU según backend | Debe mantener latencia baja. |
| EmbeddingGenerator / ArcFace | GPU-bound preferente | Candidato principal para batching. |
| WeightStrategy | CPU ligero | Costo despreciable frente a inferencia. |
| EvidenceAggregator | CPU / NumPy | Matrices pequeñas por TrackSession. |
| OutlierDetector | CPU | Trabaja sobre pocos embeddings. |


## 35. Pipeline contractual consolidado hasta esta versión

```
CameraSource
    | Frame
    v
FaceDetector
    | FaceDetection[]
    v
FaceTracker
    | TrackedFace[]
    v
TrackSessionManager
    | TrackSession
    v
FaceQualityAnalyzer
    | QualityResult
    v
FaceSampleExtractor
    | FaceSample
    v
SampleCollector
    | FaceSample[]
    v
RecognitionTask / Bounded Queue
    |
    v
EmbeddingGenerator (ArcFace)
    | EmbeddingEvidence[]
    v
EmbeddingOutlierDetector + WeightStrategy
    |
    v
EvidenceAggregator
    | IdentityEvidence
    v
[ SIGUIENTE BLOQUE ]
VectorIndex -> VectorSearchService -> CandidateFilter
-> SecondaryVerificationService -> DecisionEngine
```

Con estas interfaces, el pipeline de visión y la generación/agregación biométrica ya tienen contratos explícitos. El siguiente bloque formalizará búsqueda vectorial, filtrado de candidatos, verificación secundaria y decisión final, reutilizando los modelos de dominio definidos en la sección anterior.


## Persistencia y servicios de aplicación - contratos v1

Esta sección completa los contratos del pipeline con la capa responsable de persistir identidades, embeddings, visitas, decisiones de auditoría y configuración. La intención es mantener el dominio desacoplado de SQLite/PostgreSQL y hacer que las operaciones críticas puedan ejecutarse de forma transaccional, auditable e idempotente.


### 1. Principio de separación

Los repositorios exponen acceso a datos; los servicios de aplicación orquestan casos de uso; las políticas determinan si una acción es segura; y el índice vectorial permanece como una estructura derivada. La base de datos continúa siendo la fuente de verdad.

```
Domain / Application
      |
      v
+-------------------+
| Repositories      |  <- contracts
+---------+---------+
          |
          v
+-------------------+
| SQLite/PostgreSQL |  <- implementations
+-------------------+

Database = source of truth
Vector index = derived search structure
```


### 2. PersonRepository

Responsabilidad: persistir y recuperar identidades persistentes sin exponer SQL al resto de la aplicación.

```
class PersonRepository(Protocol):
    def get_by_id(self, person_id: str) -> PersonIdentity | None: ...
    def get_by_public_code(self, public_code: str) -> PersonIdentity | None: ...
    def create(self, person: PersonIdentity) -> PersonIdentity: ...
    def update(self, person: PersonIdentity) -> None: ...
    def increment_visit_count(self, person_id: str, seen_at: datetime) -> None: ...
    def set_status(self, person_id: str, status: PersonStatus) -> None: ...
    def list_active(self) -> list[PersonIdentity]: ...
```

Justificación: PersonIdentity contiene información de dominio liviana; los embeddings se mantienen fuera para no cargar vectores grandes en cada lectura de persona.


### 3. EmbeddingRepository

Responsabilidad: administrar el embedding PRIMARY y los SECONDARY asociados a cada identidad.

```
class EmbeddingRepository(Protocol):
    def get_primary(self, person_id: str) -> StoredEmbedding | None: ...
    def get_secondary(self, person_id: str, active_only: bool = True) -> list[StoredEmbedding]: ...
    def add(self, embedding: StoredEmbedding) -> None: ...
    def update(self, embedding: StoredEmbedding) -> None: ...
    def deactivate(self, embedding_id: str) -> None: ...
    def replace_primary(self, person_id: str, embedding: StoredEmbedding) -> None: ...
    def count_secondary(self, person_id: str) -> int: ...
```

La operación replace_primary debe garantizar un único PRIMARY activo por persona. La implementación deberá proteger esta invariancia mediante transacción y, cuando sea posible, constraints o índices únicos parciales.


### 4. VisitRepository

Responsabilidad: registrar visitas confirmadas y garantizar idempotencia mediante event_id único.

```
class VisitRepository(Protocol):
    def exists_event(self, event_id: str) -> bool: ...
    def create(self, event: VisitEvent) -> str: ...
    def list_by_person(self, person_id: str) -> list[VisitEvent]: ...
    def count_by_person(self, person_id: str) -> int: ...
```

Justificación: una sesión puede emitir el mismo evento más de una vez por reintentos; la unicidad de event_id impide registrar dos visitas para el mismo evento lógico.


### 5. DecisionAuditRepository

Responsabilidad: persistir el resultado del Decision Engine y sus métricas para debugging, trazabilidad y calibración posterior.

```
class DecisionAuditRepository(Protocol):
    def save(self, decision: DecisionResult, configuration_id: str) -> None: ...
    def list_by_session(self, session_id: str) -> list[DecisionResult]: ...
    def list_by_type(self, decision: DecisionType) -> list[DecisionResult]: ...
    def get(self, decision_id: str) -> DecisionResult | None: ...
```

Los registros de auditoría permitirán analizar márgenes, consistencia, calidad y reason_codes frente al ground truth experimental. Deben incluir configuration_id y versiones del motor para reproducir decisiones históricas.


### 6. ConfigurationRepository

Responsabilidad: administrar configuraciones versionadas y permitir conocer exactamente con qué parámetros se ejecutó cada decisión.

```
class ConfigurationRepository(Protocol):
    def get_active(self) -> SystemConfiguration: ...
    def get_by_id(self, configuration_id: str) -> SystemConfiguration | None: ...
    def save(self, configuration: SystemConfiguration) -> None: ...
    def activate(self, configuration_id: str) -> None: ...
```

Una configuración activa incluirá thresholds, TOP-K, Candidate Delta, pesos de agregación, pesos de verificación, límites de secundarios, lambda de actualización y demás parámetros calibrables.


### 7. CameraRepository

Responsabilidad: persistir la configuración lógica de cámaras para evitar hardcodear identificadores y parámetros de captura.

```
class CameraRepository(Protocol):
    def get(self, camera_id: str) -> Camera | None: ...
    def list_active(self) -> list[Camera]: ...
    def save(self, camera: Camera) -> None: ...
    def update(self, camera: Camera) -> None: ...
```


### 8. TrackSessionLogRepository

Recomendado para desarrollo y evaluación. Persiste únicamente un resumen de la TrackSession, nunca los frames completos por defecto.

```
class TrackSessionLogRepository(Protocol):
    def save_summary(self, session: TrackSession, decision_id: str | None) -> None: ...
    def get_by_session_id(self, session_id: str) -> TrackSession | None: ...
```


### 9. UnitOfWork / Transaction Boundary

Para operaciones que modifican varias tablas se introduce un contrato de unidad de trabajo. Su objetivo es evitar estados parciales.

```
class UnitOfWork(Protocol):
    persons: PersonRepository
    embeddings: EmbeddingRepository
    visits: VisitRepository
    decisions: DecisionAuditRepository

    def __enter__(self) -> "UnitOfWork": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
```

```
BEGIN TRANSACTION
        |
        +--> person
        +--> embeddings
        +--> visit
        +--> decision audit
        |
       COMMIT
        |
        v
Vector index synchronization
```

El índice se sincroniza después del commit porque la base es la fuente de verdad. Si la sincronización falla, puede reintentarse o reconstruirse sin perder integridad transaccional.


### 10. IdentityUpdatePolicy

Responsabilidad: decidir si una evidencia MATCH puede modificar el perfil biométrico existente.

```
class IdentityUpdatePolicy(Protocol):
    def evaluate(
        self,
        person: PersonIdentity,
        evidence: IdentityEvidence,
        decision: DecisionResult,
        primary: StoredEmbedding,
        secondary: list[StoredEmbedding],
    ) -> IdentityUpdateDecision: ...
```

La política es deliberadamente conservadora: un MATCH no implica actualización automática. Deben cumplirse confianza alta, calidad alta, consistencia y aportación real de calidad o diversidad.

```
MATCH
  |
  v
High confidence?
  | yes
  v
High quality?
  | yes
  v
Consistent with history?
  | yes
  v
Adds quality/diversity?
  |
  +-- no --> NO UPDATE
  |
  +-- yes --> UPDATE PROFILE
```


### 11. Actualización suave del PRIMARY

Cuando la política autorice actualizar el embedding principal, se recomienda una actualización suave en lugar de reemplazo brusco:

```
P_new = normalize( lambda * P_old + (1 - lambda) * E_current )

0 < lambda < 1
```

Un lambda alto preserva más información histórica; su valor se determinará experimentalmente. Esta estrategia reduce deriva abrupta del perfil.


### 12. SecondaryEmbeddingPolicy

Los secundarios deben representar diversidad útil. La política decide agregar, reemplazar o descartar una nueva representación.

```
class SecondaryEmbeddingPolicy(Protocol):
    def evaluate(
        self,
        candidate: StoredEmbedding,
        existing: list[StoredEmbedding],
        max_embeddings: int,
    ) -> IdentityUpdateDecision: ...
```

```
NEW SECONDARY
      |
      v
Useful quality/diversity?
   /      \
 NO       YES
 |         |
DROP    capacity?
          /   \
        YES   NO
         |     |
        ADD  REPLACE weakest/redundant
```


### 13. NewIdentityPolicy

Responsabilidad: determinar si un UNKNOWN representa evidencia suficiente para crear una nueva identidad. UNKNOWN por sí solo nunca crea una persona.

```
class NewIdentityPolicy(Protocol):
    def evaluate(
        self,
        evidence: IdentityEvidence,
        decision: DecisionResult,
        candidates: list[CandidateEvidence],
    ) -> NewIdentityDecision: ...
```

```
UNKNOWN
   |
   v
High-quality evidence?
   | yes
   v
Internally consistent?
   | yes
   v
Existing candidates rejected reliably?
   | yes
   v
Valid entry event?
   |
   +--> CREATE NEW IDENTITY

Any failure -> do not create / collect more / audit
```

Justificación: evita que un rostro malo o una falla técnica cree identidades duplicadas.


### 14. IdentityService

Responsabilidad: orquestar la creación y actualización de identidades. No calcula similitudes ni decide MATCH; consume DecisionResult y políticas ya evaluadas.

```
class IdentityService(Protocol):
    def create_identity(
        self,
        evidence: IdentityEvidence,
        decision: DecisionResult,
        creation: NewIdentityDecision,
    ) -> PersonIdentity: ...

    def update_identity(
        self,
        person_id: str,
        evidence: IdentityEvidence,
        decision: DecisionResult,
        update: IdentityUpdateDecision,
    ) -> None: ...
```


### 15. VisitService

Responsabilidad: convertir un MATCH o una nueva identidad validada en una visita persistente e idempotente.

```
class VisitService(Protocol):
    def register(self, event: VisitEvent) -> str: ...
```

```
VisitEvent
   |
   v
event_id already exists?
   |
 +-- yes --> return existing / IGNORE DUPLICATE
 |
 +-- no  --> INSERT visit + update person counters
```


### 16. DecisionAuditService

Responsabilidad: asegurar que decisiones relevantes, incluidas AMBIGUOUS y UNKNOWN, queden disponibles para análisis, incluso cuando no generen visita.

```
class DecisionAuditService(Protocol):
    def record(
        self,
        decision: DecisionResult,
        configuration_id: str,
        engine_version: str,
    ) -> None: ...
```


### 17. IndexSyncService

Responsabilidad: mantener el índice principal sincronizado con los PRIMARY persistidos después de commits exitosos.

```
class IndexSyncService(Protocol):
    def add_person(self, person_id: str) -> None: ...
    def update_person(self, person_id: str) -> None: ...
    def remove_person(self, person_id: str) -> None: ...
    def reconcile(self) -> IndexHealth: ...
```

```
DATABASE COMMIT
      |
      v
IndexSyncService
      |
      +--> add/update/remove PRIMARY
      |
      v
VectorIndex

Failure -> audit + retry/reconcile
Database remains authoritative
```


### 18. ApplicationIdentityOrchestrator

Para evitar que main.py conozca demasiados servicios, se recomienda un orquestador de aplicación que consuma DecisionResult y ejecute el caso correspondiente.

```
class IdentityDecisionHandler(Protocol):
    def handle(
        self,
        session: TrackSession,
        evidence: IdentityEvidence,
        candidates: list[CandidateEvidence],
        decision: DecisionResult,
    ) -> None: ...
```

```
DecisionResult
      |
 +----+---------+
 |              |
MATCH       AMBIGUOUS       UNKNOWN
 |              |              |
 v              v              v
Audit        Audit only    NewIdentityPolicy
 |                             |
Visit                      create?
 |                         /    \
UpdatePolicy              no    yes
 |                         |      |
Maybe update             audit  create person
 |                                |
Index sync                       visit
                                  |
                              index sync
```


### 19. Errores de persistencia y aplicación

```
class PersistenceError(Exception): pass
class RepositoryError(PersistenceError): pass
class TransactionError(PersistenceError): pass
class DuplicateVisitEventError(PersistenceError): pass
class PersonNotFoundError(PersistenceError): pass
class EmbeddingNotFoundError(PersistenceError): pass

class ApplicationServiceError(Exception): pass
class IdentityCreationError(ApplicationServiceError): pass
class IdentityUpdateError(ApplicationServiceError): pass
class VisitRegistrationError(ApplicationServiceError): pass
class IndexSynchronizationError(ApplicationServiceError): pass
```

Los errores técnicos no deben convertirse automáticamente en UNKNOWN. UNKNOWN es una conclusión biométrica; un fallo de repositorio o índice es un fallo operativo y debe registrarse/tratarse por separado.


### 20. Contrato de idempotencia

Toda operación que pueda reintentarse debe usar identificadores estables. Para visitas se utiliza event_id; para tareas biométricas task_id; para decisiones decision_id. El mismo identificador lógico repetido no debe producir duplicación persistente.


### 21. Flujo transaccional MATCH

```
Decision = MATCH
      |
      v
BEGIN UoW
      |
      +--> save audit decision
      +--> insert visit if event_id new
      +--> update last_seen / total_visits
      +--> evaluate and persist profile update if approved
      |
    COMMIT
      |
      v
IndexSyncService (only if PRIMARY changed)
```


### 22. Flujo transaccional UNKNOWN validado

```
Decision = UNKNOWN
      |
      v
NewIdentityPolicy
      |
  should_create?
   /        \
 no        yes
 |          |
Audit    BEGIN UoW
            |
            +--> create person UUID + public_code
            +--> save PRIMARY
            +--> save SECONDARY[]
            +--> save decision audit
            +--> register first visit
            |
          COMMIT
            |
            v
        IndexSyncService.add_person()
```


### 23. Flujo AMBIGUOUS

```
AMBIGUOUS
    |
    +--> persist decision audit
    +--> no visit yet
    +--> no new identity
    +--> no profile mutation
    +--> session may collect more evidence
```


### 24. Contratos principales - resumen

| Contrato | Entrada | Salida / efecto |
|---|---|---|
| PersonRepository | PersonIdentity / person_id | CRUD de identidad |
| EmbeddingRepository | StoredEmbedding / person_id | PRIMARY/SECONDARY persistidos |
| VisitRepository | VisitEvent | Visita idempotente |
| DecisionAuditRepository | DecisionResult | Registro de auditoría |
| ConfigurationRepository | SystemConfiguration | Configuración versionada |
| UnitOfWork | Repositorios | Commit / rollback |
| IdentityUpdatePolicy | MATCH + perfil + evidencia | IdentityUpdateDecision |
| NewIdentityPolicy | UNKNOWN + evidencia | NewIdentityDecision |
| IdentityService | Decisiones de política | Crear/actualizar identidad |
| VisitService | VisitEvent | Registrar visita |
| DecisionAuditService | DecisionResult | Auditar |
| IndexSyncService | person_id | Sincronizar índice |
| IdentityDecisionHandler | DecisionResult + contexto | Orquestar caso de uso |


### 25. Arquitectura de extremo a extremo actualizada

```
CAMERA
  |
  v
DETECTION -> TRACKING -> QUALITY -> SAMPLE COLLECTION
  |
  v
RecognitionTask -> ArcFace -> EmbeddingEvidence[]
  |
  v
EvidenceAggregator -> IdentityEvidence
  |
  v
VectorSearchService -> CandidateFilter -> CandidateMatch[]
  |
  v
SecondaryVerificationService -> CandidateEvidence[]
  |
  v
DecisionEngine -> DecisionResult
  |
  v
IdentityDecisionHandler
  |
  +--> MATCH ------> Visit + optional profile update
  |
  +--> AMBIGUOUS --> Audit + collect more
  |
  +--> UNKNOWN ----> NewIdentityPolicy -> maybe create identity + first visit
  |
  v
UnitOfWork / Repositories
  |
  v
DATABASE (source of truth)
  |
  v
IndexSyncService -> VectorIndex
```


### 26. Qué queda después de esta capa

Con los contratos anteriores, el flujo de negocio queda definido de extremo a extremo. Antes de comenzar implementación completa quedan cinco bloques de diseño transversales y una fase de construcción:

1. 1. Orquestación y concurrencia definitiva: workers, colas, tamaños, backpressure, cancelación y prioridades.
1. 2. Observabilidad: logging estructurado, métricas, tracing, health checks y telemetry de saturación.
1. 3. Configuración y composición de dependencias: settings, factories/DI, perfiles CPU/GPU y versiones de modelos.
1. 4. Estrategia de pruebas: unitarias, integración, end-to-end, dataset experimental, ground truth y benchmarks.
1. 5. Seguridad operativa: cifrado, secretos, permisos, retención, anonimización y políticas de acceso.
1. 6. Implementación real del prototipo y calibración de parámetros experimentales.

### 27. Estado arquitectónico después de este bloque

| Área | Estado |
|---|---|
| Pipeline de visión | Definido |
| Modelos de dominio | Definidos |
| Núcleo biométrico | Definido |
| Vector search | Definido |
| Secondary verification | Definido |
| Decision Engine | Definido |
| Persistencia | Definida |
| Servicios de aplicación | Definidos |
| Políticas de identidad | Definidas |
| Auditoría/configuración | Definidas |
| Concurrencia definitiva | Pendiente |
| Observabilidad | Pendiente |
| Testing formal | Pendiente |
| Seguridad operativa | Pendiente |
| Implementación/benchmarks | Pendiente |


### 28. Recomendación de siguiente paso

El siguiente diseño recomendado es la orquestación y concurrencia definitiva. Ya existen RecognitionTask, RecognitionResult y bounded queues a nivel conceptual; ahora debe definirse cuántos workers existen, qué corre en CPU/GPU, cómo se priorizan sesiones, cómo se cancela trabajo obsoleto, cómo se evita saturación y cómo fluye un evento desde cámara hasta persistencia sin bloquear la captura.


## Orquestación y concurrencia v1

Esta sección cierra la arquitectura de ejecución en tiempo real. Su objetivo es definir cómo cooperan los módulos ya contratados sin bloquear la captura, cómo se distribuye trabajo entre CPU y GPU, cómo se limita la presión de carga y cómo se preservan orden, idempotencia y consistencia cuando existen varias tareas simultáneas.


### 1. Principios de orquestación

- Separar el camino de video en tiempo real del camino biométrico asíncrono.
- Usar colas acotadas; nunca colas infinitas.
- Mantener un único dueño de cada TrackSession para evitar condiciones de carrera.
- No permitir que múltiples procesos compitan innecesariamente por la misma GPU.
- Evitar mover frames 1080p completos entre procesos cuando basta un recorte facial.
- Tratar la base de datos como fuente de verdad y el índice vectorial como estructura derivada.
- Distinguir fallos técnicos de conclusiones biométricas: un timeout o error de BD no equivale a UNKNOWN.
- Diseñar todas las operaciones reintentables con identificadores estables e idempotencia.

### 2. Dos caminos de ejecución

Se divide el sistema en un camino de baja latencia para mantener la escena actualizada y un camino biométrico asíncrono para operaciones costosas.

```
REAL-TIME PATH                         BIOMETRIC PATH
----------------                         --------------
Camera -> Frame                          RecognitionTask
   |                                         |
   v                                         v
Detection -> Tracking                  ArcFace Worker
   |                                         |
   v                                         v
Quality -> SampleCollector             EvidenceAggregator
   |                                         |
   +---- session READY --------------------->|
                                             v
                                       Vector Search
                                             |
                                             v
                                      Secondary Verify
                                             |
                                             v
                                       DecisionEngine
                                             |
                                             v
                                      Persistence/Index
```

Justificación: la cámara debe seguir capturando y actualizando tracks aunque ArcFace, la búsqueda o la persistencia estén ocupados. Esto impide que una inferencia lenta congele el video.


### 3. Modelo de ejecución recomendado para el prototipo

Para una webcam se recomienda inicialmente un modelo híbrido sencillo:

- 1 hilo de captura por cámara.
- 1 pipeline de detección/tracking asociado a la cámara o a un worker controlado.
- 1 coordinador de TrackSession en el proceso principal.
- 1 worker dedicado de inferencia biométrica (GPU si está disponible).
- 1 pool CPU pequeño para agregación, búsqueda y verificación cuando sea útil.
- 1 escritor de persistencia serializado mientras se utilice SQLite.
```
+------------------+      +----------------------+      +------------------+
| Capture Thread    | ---> | Vision Coordinator   | ---> | RecognitionQueue |
+------------------+      | detect/track/quality |      +--------+---------+
                          +----------------------+               |
                                                               v
                                                     +--------------------+
                                                     | ArcFace GPU Worker |
                                                     +---------+----------+
                                                               |
                                                               v
                                                     +--------------------+
                                                     | Identity Pipeline  |
                                                     | CPU/search/verify  |
                                                     +---------+----------+
                                                               |
                                                               v
                                                     +--------------------+
                                                     | Persistence Writer |
                                                     +--------------------+
```


### 4. Contrato Orchestrator

```
class PipelineOrchestrator(Protocol):
    def start(self) -> None: ...
    def stop(self, graceful: bool = True) -> None: ...
    def submit_camera(self, camera: Camera) -> None: ...
    def health(self) -> "PipelineHealth": ...
```

Responsabilidad: iniciar y detener recursos, coordinar workers y colas, y exponer el estado global. No contiene lógica biométrica ni SQL.


### 5. Contrato TaskQueue

```
T = TypeVar("T")

class TaskQueue(Protocol, Generic[T]):
    def put(self, item: T, block: bool = True) -> None: ...
    def get(self, timeout: float | None = None) -> T: ...
    def try_put(self, item: T) -> bool: ...
    def qsize(self) -> int: ...
    def capacity(self) -> int: ...
    def close(self) -> None: ...
```

La interfaz desacopla la lógica de aplicación de queue.Queue, multiprocessing.Queue u otra implementación futura.


### 6. Colas principales y política de datos

| Cola | Contenido | Productor | Consumidor | Política inicial |
|---|---|---|---|---|
| FrameQueue | Frame o referencia | CameraSource | VisionCoordinator | Bounded; preferir frame reciente |
| RecognitionQueue | RecognitionTask | SampleCollector/Coordinator | EmbeddingWorker | Bounded; deduplicar por session_id |
| RecognitionResultQueue | RecognitionResult | EmbeddingWorker | IdentityPipelineWorker | Bounded; no descartar resultados vigentes |
| PersistenceQueue | PersistenceCommand | Application handler | PersistenceWorker | Bounded; idempotente, reintentos limitados |


### 7. FrameQueue: política latest-frame

En captura en vivo, procesar cada frame atrasado suele ser peor que saltar algunos frames. Si el detector no alcanza el FPS de cámara, la cola debe favorecer actualidad.

```
Camera:  F100 F101 F102 F103 F104 F105
                 \______________/
                    backlog

BAD:  process F100, F101, F102... while reality is already F105
GOOD: keep a small bounded buffer and prefer the newest useful frame
```

Esta política aplica al video, no a eventos biométricos ya confirmados. Una RecognitionTask no debe desaparecer solo porque llegue un frame nuevo.


### 8. RecognitionQueue y deduplicación de sesión

Una TrackSession no debe tener múltiples tareas biométricas equivalentes ejecutándose al mismo tiempo.

```
class RecognitionScheduler(Protocol):
    def submit(self, task: RecognitionTask) -> bool: ...
    def cancel_session(self, session_id: str) -> None: ...
    def is_pending(self, session_id: str) -> bool: ...
```

```
TrackSession READY
      |
      v
pending(session_id)?
   /        yes       no
  |         |
skip     enqueue task
```

Justificación: evita inferencias duplicadas, consumo doble de GPU y decisiones simultáneas sobre la misma sesión.


### 9. Propiedad exclusiva de TrackSession

TrackSession será mutable, pero únicamente TrackSessionManager/VisionCoordinator podrá modificarla. Los workers reciben snapshots inmutables como RecognitionTask.

```
VisionCoordinator ---- owns ----> TrackSession (mutable)
        |
        +---- snapshot ----> RecognitionTask (immutable)
                                  |
                                  v
                                workers

Workers MUST NOT mutate TrackSession directly.
```

Esta regla elimina gran parte de las condiciones de carrera sin introducir locks alrededor de cada atributo.


### 10. Staleness y cancelación

Una tarea puede quedar obsoleta si la sesión termina, es reemplazada por evidencia más reciente o ya fue identificada por otro intento válido.

```
@dataclass(frozen=True, slots=True)
class TaskContext:
    task_id: str
    session_id: str
    session_revision: int
    created_at: datetime
    deadline_at: datetime | None

```

Antes de aplicar un resultado se verifica que session_revision siga vigente. Un resultado antiguo puede registrarse para telemetría, pero no debe sobrescribir una decisión más reciente.


### 11. Prioridad y fairness

Si existe saturación, se recomienda una PriorityRecognitionQueue con una política transparente y configurable. Una función conceptual puede combinar urgencia, calidad y antigüedad:

```
priority = a * entry_urgency + b * evidence_quality + c * age_score
```

Los coeficientes se calibran; age_score aumenta con el tiempo para evitar starvation. Criterio recomendado de precedencia:

- sesiones que ya cruzaron la línea de entrada y tienen deadline cercano;
- sesiones READY con evidencia de alta calidad;
- sesiones más antiguas que aún no fueron procesadas.

### 12. Backpressure

Cada cola tendrá capacidad máxima. Si la tasa de llegada supera la capacidad de servicio, el sistema debe degradarse de forma controlada en lugar de acumular memoria y latencia indefinidamente.

```
arrival rate (lambda) ---> [ bounded queue ] ---> service rate (c * mu)

If lambda >= c*mu for a sustained interval:
    backlog grows -> latency grows -> tasks become stale -> memory pressure

```

Para c workers con tasa media de servicio mu por worker, una referencia de utilización es:

```
rho = lambda / (c * mu)
```

La operación estable requiere rho < 1 de forma sostenida; en la práctica se busca margen por debajo de 1 porque la latencia aumenta rápidamente cerca de saturación.


### 13. Políticas de backpressure por cola

| Cola | Cuando se llena | Justificación |
|---|---|---|
| FrameQueue | Descartar frame antiguo / conservar más reciente | Los frames viejos pierden valor temporal. |
| RecognitionQueue | No duplicar sesiones; rechazar tarea obsoleta; priorizar deadline | La evidencia de entrada sí tiene valor; no se descarta arbitrariamente. |
| ResultQueue | Breve espera y alerta de saturación | Un resultado vigente no debe perderse silenciosamente. |
| PersistenceQueue | Bloqueo acotado + retry; nunca duplicar event_id | La persistencia debe conservar consistencia e idempotencia. |


### 14. Batching del EmbeddingGenerator

El contrato generate_batch() permite agrupar muestras y mejorar utilización de GPU. Se recomienda micro-batching con dos límites: tamaño máximo y tiempo máximo de espera.

```
@dataclass(frozen=True, slots=True)
class BatchPolicy:
    max_batch_size: int
    max_wait_ms: int

```

```
Task A ----Task B -----+--> batch until max_batch_size OR max_wait_ms --> ArcFace
Task C ----/
```

Justificación: esperar indefinidamente para llenar un batch reduce throughput útil por aumentar latencia. El límite temporal permite balancear ambos objetivos.


### 15. GPU Worker dedicado

Para el prototipo se recomienda un único worker dueño del modelo ArcFace/GPU. Crear varios procesos que carguen el mismo modelo puede duplicar VRAM y producir contención.

```
RecognitionQueue
      |
      v
+---------------------------+
| Dedicated EmbeddingWorker |
| owns ArcFace + GPU context |
+-------------+-------------+
              |
              v
     RecognitionResultQueue
```

Si los benchmarks demuestran que una GPU admite concurrencia útil, esta política podrá evolucionar; no se asumirá de antemano.


### 16. CPU pools y GIL

No se usará multiprocessing indiscriminadamente. NumPy/OpenCV y runtimes de inferencia suelen ejecutar partes nativas fuera del GIL; primero se medirá. Para trabajo Python puramente CPU-bound se podrá usar ProcessPool; para I/O y librerías nativas, threads pueden ser suficientes.


### 17. Minimizar copias entre procesos

Un frame 1920x1080 BGR ocupa aproximadamente 1920*1080*3 bytes (~6 MB sin compresión). Copiar frames completos repetidamente entre procesos es costoso.

```
Camera Frame (~6 MB)
      |
      +--> keep local to vision path
      |
      +--> crop only selected face samples
                 |
                 v
          RecognitionTask
```

Para el prototipo se enviarán recortes faciales seleccionados. Shared memory queda como optimización futura si los benchmarks justifican la complejidad.


### 18. IdentityPipelineWorker

```
class IdentityPipelineWorker(Protocol):
    def process(self, result: RecognitionResult) -> DecisionResult: ...
```

Orquesta, en orden: EvidenceAggregator -> VectorSearchService -> CandidateFilter -> SecondaryVerificationService -> DecisionEngine. No persiste directamente; entrega DecisionResult al handler de aplicación.

```
RecognitionResult
      |
      v
EvidenceAggregator
      |
VectorSearchService
      |
CandidateFilter
      |
SecondaryVerificationService
      |
DecisionEngine
      |
DecisionResult
```


### 19. PersistenceWorker y SQLite

Mientras el prototipo utilice SQLite, se recomienda serializar las escrituras mediante un PersistenceWorker/UnitOfWork. SQLite permite buena concurrencia de lectura, pero múltiples escritores aumentan el riesgo de lock contention.

```
class PersistenceWorker(Protocol):
    def submit(self, command: "PersistenceCommand") -> None: ...
    def flush(self, timeout: float | None = None) -> None: ...
```

Se recomienda habilitar WAL cuando corresponda en la implementación y mantener transacciones cortas. Una migración futura a PostgreSQL permitirá mayor concurrencia de escritura.


### 20. PersistenceCommand

```
@dataclass(frozen=True, slots=True)
class PersistenceCommand:
    command_id: str
    session_id: str
    decision: DecisionResult
    evidence: IdentityEvidence
    configuration_id: str
    created_at: datetime
```

command_id permite reintentos idempotentes. El worker delega en IdentityDecisionHandler y UnitOfWork; no duplica lógica de dominio.


### 21. RetryPolicy

```
@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    initial_delay_ms: int
    max_delay_ms: int
    backoff_multiplier: float

```

Se reintentan únicamente errores clasificados como transitorios. Errores de validación, incompatibilidad de modelo o reglas de dominio no se solucionan repitiendo la misma operación.

```
failure
  |
  v
transient?
 /      no      yes
|        |
fail   retry with bounded backoff
         |
      max attempts?
       /           no       yes
     |         |
   retry    record failure / alert
```


### 22. Idempotencia extremo a extremo

Los identificadores ya definidos permiten seguridad frente a retries:

- task_id: deduplicación de trabajo biométrico.
- session_id + session_revision: descartar resultados obsoletos.
- decision_id: auditoría de una decisión concreta.
- command_id: deduplicación del comando de persistencia.
- event_id: impedir visitas duplicadas.

### 23. Manejo de errores técnicos vs biométricos

```
Biometric evidence weak  ---> UNKNOWN / AMBIGUOUS (domain result)
Database unavailable       ---> technical failure / retry
Vector index out of sync   ---> technical failure / rebuild
GPU inference failure      ---> technical failure / retry or degrade
Candidate not found        ---> domain/consistency case, audited
```

Justificación: mezclar fallos de infraestructura con UNKNOWN contaminaría las métricas FAR/FRR y podría crear identidades falsas.


### 24. Graceful shutdown

El sistema debe poder detenerse sin perder una visita ya decidida. Orden recomendado:

1. Detener nuevas capturas y nuevas RecognitionTask.

2. Permitir que workers biométricos terminen tareas vigentes dentro de un timeout.

3. Cerrar ResultQueue y drenar resultados pendientes.

4. Vaciar PersistenceQueue mediante flush().

5. Commit/rollback de transacciones abiertas.

6. Persistir métricas/estado final y cerrar índice/modelos/cámara.

```
STOP REQUEST
    |
    v
stop producers
    |
drain biometric work
    |
drain persistence
    |
close DB/index/GPU
    |
STOPPED
```


### 25. Startup y readiness

El servicio no debe declararse READY solo porque el proceso arrancó.

```
START
  |
  +--> load configuration
  +--> open database
  +--> initialize repositories
  +--> initialize/check vector index
  +--> load SCRFD / ArcFace
  +--> start workers
  +--> open cameras
  |
  v
READINESS CHECK
  |
  +-- all critical components healthy --> READY
  +-- otherwise -----------------------> DEGRADED / NOT READY
```


### 26. Estado operativo

```
class ServiceState(Enum):
    STARTING = auto()
    READY = auto()
    DEGRADED = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()

@dataclass(frozen=True, slots=True)
class PipelineHealth:
    state: ServiceState
    frame_queue_depth: int
    recognition_queue_depth: int
    persistence_queue_depth: int
    gpu_worker_ready: bool
    vector_index_ready: bool
    database_ready: bool
```


### 27. Métricas mínimas de concurrencia

- queue_depth por cola y porcentaje de ocupación;
- tasks_enqueued, tasks_completed, tasks_cancelled, tasks_stale;
- batch_size y batch_wait_ms;
- worker_busy_ratio;
- camera_frames_captured y frames_dropped;
- recognition_latency_ms y end_to_end_latency_ms;
- persistence_retry_count y transaction_failure_count;
- GPU utilization/VRAM cuando esté disponible.
Estas métricas serán indispensables para benchmarks y para decidir si aumentar workers, cambiar batch o migrar de SQLite.


### 28. Little's Law como herramienta de diagnóstico

Para un sistema estable, una relación útil es:

```
L = lambda * W
```

donde L es el número medio de elementos en el sistema/cola, lambda la tasa media de llegada y W el tiempo medio. No se utilizará como sustituto de benchmarks, sino como comprobación de coherencia entre throughput, latencia y backlog.


### 29. Escalamiento a múltiples cámaras

La arquitectura permite que cada cámara tenga captura y tracking independientes mientras comparte el núcleo biométrico.

```
Camera 1 -> Vision Pipeline 1 --Camera 2 -> Vision Pipeline 2 ----+--> Shared Recognition Scheduler --> GPU Worker(s)
Camera 3 -> Vision Pipeline 3 ----+              |
Camera 4 -> Vision Pipeline 4 --/                v
                                      Identity/Persistence Pipeline
```

La escala de workers no se fijará por número de cámaras, sino por benchmarks de tasa de llegada, batch, latencia y recursos.


### 30. Flujo concurrente consolidado

```
+-----------+      +----------------------+       +-------------------+
| CAMERA(S) | ---> | VisionCoordinator    | ----> | RecognitionQueue  |
+-----------+      | detect/track/quality |       +---------+---------+
                   +----------------------+                 |
                                                            v
                                                   +------------------+
                                                   | Batch Scheduler  |
                                                   +--------+---------+
                                                            |
                                                            v
                                                   +------------------+
                                                   | ArcFace Worker   |
                                                   | GPU owner        |
                                                   +--------+---------+
                                                            |
                                                            v
                                                   RecognitionResult
                                                            |
                                                            v
                                               +------------------------+
                                               | IdentityPipelineWorker |
                                               | aggregate/search/verify|
                                               | decision               |
                                               +-----------+------------+
                                                           |
                                                           v
                                                 PersistenceCommand
                                                           |
                                                           v
                                               +------------------------+
                                               | PersistenceWorker      |
                                               | UoW + audit + visits   |
                                               +-----------+------------+
                                                           |
                                                DB commit  |  index sync
                                                           v
                                                   +---------------+
                                                   | Vector Index  |
                                                   +---------------+
```


### 31. Estructura de paquetes sugerida para orquestación

```
application/
|-- orchestration/
|   |-- pipeline_orchestrator.py
|   |-- vision_coordinator.py
|   |-- recognition_scheduler.py
|   |-- identity_pipeline_worker.py
|   |-- persistence_worker.py
|   `-- lifecycle.py
|
|-- messaging/
|   |-- task_queue.py
|   |-- priority_queue.py
|   `-- messages.py
|
|-- policies/
|   |-- batch_policy.py
|   |-- retry_policy.py
|   |-- backpressure_policy.py
|   `-- scheduling_policy.py
|
`-- monitoring/
    `-- health_models.py
```


### 32. Decisiones cerradas en Orquestación v1

| Decisión | Estado |
|---|---|
| Separar real-time path y biometric path | Definida |
| Colas acotadas | Definida |
| Latest-frame para backlog de video | Definida |
| Deduplicación de RecognitionTask por sesión | Definida |
| Snapshots inmutables hacia workers | Definida |
| session_revision para resultados obsoletos | Definida |
| Prioridad con fairness/aging | Definida conceptualmente; pesos experimentales |
| Micro-batching ArcFace | Definido; tamaño/espera experimental |
| Un GPU worker inicialmente | Definido para prototipo; sujeto a benchmark |
| Persistencia serializada con SQLite | Definida para prototipo |
| Retry solo para fallos transitorios | Definida |
| Graceful shutdown/startup readiness | Definidos |
| Métricas de cola/worker/latencia | Definidas |
| Shared memory | Pospuesto hasta benchmark |


### 33. Parámetros de concurrencia pendientes de benchmark

- frame_queue_capacity
- recognition_queue_capacity
- persistence_queue_capacity
- max_batch_size
- max_batch_wait_ms
- número de workers CPU
- número de GPU workers
- task deadline
- retry delays
- priority coefficients
- timeouts de graceful shutdown
Estos valores son deliberadamente experimentales: fijarlos sin medir el hardware real produciría una falsa precisión de diseño.


### 34. Revisión de completitud arquitectónica

Con esta sección quedan diseñados el flujo funcional y la arquitectura de ejecución. Antes de implementar el prototipo todavía se recomienda cerrar cuatro bloques documentales transversales:

1. Observabilidad: logging estructurado, métricas, tracing, health/readiness y dashboards de desarrollo.

2. Configuración y composición: settings, factories/DI, perfiles CPU/GPU, model/index/config versions y startup wiring.

3. Estrategia de pruebas y validación: unitarias, integración, E2E, dataset/ground truth, protocolo de benchmark y plan de calibración.

4. Seguridad operativa y privacidad aplicada: cifrado, secretos, permisos, retención, borrado, acceso a auditoría y threat model básico.

Después de cerrar esos cuatro bloques, la estructura del proyecto puede considerarse completa para iniciar implementación. Benchmarks y calibración se ejecutarán después del prototipo, porque dependen del código, hardware y datos reales.


### 35. Secuencia final recomendada del proyecto

```
DOCUMENT DESIGN
   |
   +-- Functional architecture .............. DONE
   +-- Domain models ........................ DONE
   +-- Module contracts ..................... DONE
   +-- Persistence/application services ..... DONE
   +-- Orchestration/concurrency ............ DONE
   |
   +-- Observability ........................ NEXT
   +-- Configuration / dependency wiring .... PENDING
   +-- Test & validation strategy ........... PENDING
   +-- Operational security/privacy ......... PENDING
   |
   v
IMPLEMENT PROTOTYPE
   |
   v
BENCHMARK
   |
   v
CALIBRATE EXPERIMENTALLY
   |
   v
VALIDATE / ITERATE / FINAL REPORT
```


## Observabilidad v1

La observabilidad se incorpora como una capa transversal y obligatoria. Su objetivo no es solamente mostrar logs, sino permitir explicar el comportamiento del sistema, detectar saturación, localizar cuellos de botella, auditar decisiones biométricas y proporcionar evidencia cuantitativa para los benchmarks y la calibración experimental.


### 1. Principios de observabilidad

- Correlación extremo a extremo: todo evento relevante debe poder relacionarse con camera_id, session_id, track_id, task_id, evidence_id y decision_id.
- Separación entre telemetría técnica y auditoría biométrica: los logs operativos no sustituyen a identification_decisions.
- Métricas antes que intuición: el rendimiento se evaluará mediante latencias, throughput, profundidad de colas y utilización.
- Privacidad por diseño: no registrar imágenes faciales, embeddings completos ni datos biométricos sensibles en logs.
- Degradación observable: cuando el sistema entre en backpressure o descarte tareas, el evento debe quedar medido y trazable.

### 2. StructuredLogger

Se define un contrato de logging estructurado para evitar mensajes de texto no correlacionables.

```
class StructuredLogger(Protocol):
    def debug(self, event: str, **fields) -> None: ...
    def info(self, event: str, **fields) -> None: ...
    def warning(self, event: str, **fields) -> None: ...
    def error(self, event: str, **fields) -> None: ...
    def exception(self, event: str, **fields) -> None: ...
```

Cada evento utilizará un nombre estable, por ejemplo recognition.task_enqueued, recognition.task_completed, vector.search_completed o decision.match_confirmed.


### 3. CorrelationContext

```
@dataclass(frozen=True, slots=True)
class CorrelationContext:
    camera_id: str | None = None
    session_id: str | None = None
    track_id: int | None = None
    task_id: str | None = None
    evidence_id: str | None = None
    decision_id: str | None = None
    event_id: str | None = None
```

La justificación es poder reconstruir una trayectoria completa sin registrar información biométrica cruda.


### 4. Eventos mínimos de log

| Área | Eventos recomendados |
|---|---|
| Cámara | camera.opened, camera.read_failed, camera.disconnected |
| Tracking | track.created, track.expired, track.finished |
| Samples | sample.accepted, sample.rejected, session.ready |
| Reconocimiento | recognition.task_enqueued, started, completed, stale_result |
| Vector search | vector.search_completed, candidate_filter.completed |
| Verificación | verification.completed, candidate_profile_invalid |
| Decision Engine | decision.match, ambiguous, unknown |
| Persistencia | transaction.committed, rolled_back, persistence.retry |
| Índice | index.ready, index.out_of_sync, index.rebuilt |
| Concurrencia | queue.high_watermark, task.dropped, worker.failed |


### 5. MetricsCollector

```
class MetricsCollector(Protocol):
    def increment(self, name: str, value: int = 1, **labels) -> None: ...
    def observe(self, name: str, value: float, **labels) -> None: ...
    def gauge(self, name: str, value: float, **labels) -> None: ...
```

Las métricas se agrupan en cuatro familias principales.

| Familia | Métricas |
|---|---|
| Visión | fps_capture, detection_latency_ms, tracking_latency_ms, accepted_sample_ratio |
| Biometría | embedding_latency_ms, batch_size, evidence_consistency, verification_latency_ms |
| Búsqueda/decisión | vector_search_latency_ms, candidate_count, decision_latency_ms, match/ambiguous/unknown count |
| Concurrencia | queue_depth, queue_wait_ms, task_age_ms, dropped_tasks, worker_utilization |
| Persistencia | db_write_latency_ms, transaction_failures, index_sync_lag_ms |
| Recursos | cpu_percent, ram_bytes, gpu_utilization, vram_bytes |


### 6. Latencia extremo a extremo

Se define como métrica principal:

```
T_e2e = t_decision_completed - t_entry_event_detected
```

Y se descompone conceptualmente:

```
T_e2e =
    T_collect
  + T_queue
  + T_embedding
  + T_aggregation
  + T_search
  + T_verification
  + T_decision
  + T_persistence
```

La descomposición permite saber si una latencia alta proviene de visión, cola, GPU, búsqueda o base de datos.


### 7. Tracing

```
EntryEvent
   |
   +-- span: sample_collection
   |
   +-- span: embedding_generation
   |
   +-- span: evidence_aggregation
   |
   +-- span: vector_search
   |
   +-- span: secondary_verification
   |
   +-- span: decision_engine
   |
   +-- span: persistence
```

Para el prototipo puede implementarse con identificadores y timestamps internos; la arquitectura queda preparada para OpenTelemetry u otra solución en una fase posterior.


### 8. Health y readiness

```
@dataclass(frozen=True, slots=True)
class SystemHealth:
    camera_ready: bool
    detector_ready: bool
    tracker_ready: bool
    embedder_ready: bool
    vector_index_ready: bool
    database_ready: bool
    queues_healthy: bool
    degraded: bool
    reasons: tuple[str, ...]
```

Liveness responde si el proceso está vivo. Readiness responde si puede procesar visitantes de forma segura. Un proceso vivo con índice desincronizado no debe considerarse ready.


### 9. Diagrama de observabilidad

```

                         +------------------+
                         | Pipeline runtime |
                         +---------+--------+
                                   |
             +---------------------+----------------------+
             |                     |                      |
             v                     v                      v
      +-------------+       +-------------+        +-------------+
      | Structured  |       | Metrics     |        | Trace /     |
      | Logging     |       | Collector   |        | Correlation |
      +------+------+       +------+------+        +------+------+
             |                     |                      |
             +----------+----------+----------+-----------+
                        |                     |
                        v                     v
                +---------------+      +---------------+
                | Debug / Audit |      | Health /      |
                | Investigation |      | Benchmarks    |
                +---------------+      +---------------+
```


## Configuración y composición de dependencias v1

La configuración se separa del código para que los parámetros experimentales, infraestructura y perfiles de ejecución puedan modificarse sin alterar la lógica del dominio. La composición de dependencias tendrá un único punto de entrada, evitando instanciaciones dispersas.


### 10. ConfigurationProvider

```
class ConfigurationProvider(Protocol):
    def load(self) -> SystemConfiguration: ...
    def validate(self, config: SystemConfiguration) -> None: ...
    def active_version(self) -> str: ...
```

Las configuraciones deberán ser versionadas y auditables.


### 11. Configuración jerárquica

```
SystemConfiguration
|
+-- camera
+-- detection
+-- tracking
+-- quality
+-- recognition
+-- aggregation
+-- vector_search
+-- verification
+-- decision
+-- persistence
+-- concurrency
+-- observability
+-- security
```

La separación por dominios evita un archivo plano con decenas de variables sin contexto.


### 12. Perfiles de ejecución

| Perfil | Objetivo |
|---|---|
| development_cpu | Depuración local sin GPU |
| development_gpu | Desarrollo con aceleración |
| evaluation | Captura intensiva de métricas y auditoría |
| benchmark | Medición controlada con mínima interferencia |
| production_like | Configuración cercana al escenario real |


### 13. AppContainer / composición

```
@dataclass(slots=True)
class AppContainer:
    camera_source: CameraSource
    face_detector: FaceDetector
    face_tracker: FaceTracker
    quality_analyzer: FaceQualityAnalyzer
    sample_collector: SampleCollector

    embedding_generator: EmbeddingGenerator
    evidence_aggregator: EvidenceAggregator

    vector_index: VectorIndex
    vector_search: VectorSearchService
    secondary_verification: SecondaryVerificationService
    decision_engine: DecisionEngine

    identity_service: IdentityService
    visit_service: VisitService
    audit_service: DecisionAuditService

    orchestrator: PipelineOrchestrator
    logger: StructuredLogger
    metrics: MetricsCollector
```


### 14. CompositionRoot

```
def build_application(config: SystemConfiguration) -> AppContainer:
    # 1. infraestructura básica
    # 2. repositorios / UnitOfWork
    # 3. modelos SCRFD / ArcFace
    # 4. índice vectorial
    # 5. servicios
    # 6. workers / queues
    # 7. orchestrator
    # 8. observabilidad
    ...
```

El Composition Root será el único lugar que conoce implementaciones concretas como OpenCVCameraSource, SCRFDFaceDetector, ArcFaceEmbeddingGenerator, FaissVectorIndex y SQLite repositories.


### 15. Orden de arranque

```

LOAD CONFIG
    |
    v
VALIDATE CONFIG
    |
    v
OPEN DATABASE
    |
    v
LOAD MODELS
    |
    v
INITIALIZE / CHECK VECTOR INDEX
    |
    v
CREATE QUEUES + WORKERS
    |
    v
OPEN CAMERA
    |
    v
READINESS CHECK
    |
    v
SYSTEM READY
```


### 16. Shutdown

```

STOP ACCEPTING NEW TRACKS
          |
          v
DRAIN / CANCEL OBSOLETE TASKS
          |
          v
FLUSH PERSISTENCE QUEUE
          |
          v
STOP WORKERS
          |
          v
SAVE / FLUSH INDEX
          |
          v
CLOSE CAMERA + DATABASE
```


### 17. Validación de configuración

- Los pesos de agregación deben sumar aproximadamente 1.
- Los pesos de VerificationScore deben sumar aproximadamente 1.
- top_k > 0 y max_candidates <= top_k.
- Las dimensiones del embedding deben coincidir con el modelo y el índice.
- Los thresholds deben pertenecer a rangos válidos.
- Las colas deben ser acotadas.
- La configuración de modelo y versión debe coincidir con embeddings persistidos.

## Estrategia formal de pruebas y validación v1

Las pruebas se diseñan antes de la implementación para asegurar que el prototipo sea verificable y que la calibración posterior pueda realizarse con datos reproducibles. Se separan correctness, rendimiento y validez biométrica.


### 18. Pirámide de pruebas

```

                +------------------+
                |       E2E        |
                +--------+---------+
                         |
             +-----------+-----------+
             |      Integration      |
             +-----------+-----------+
                         |
        +----------------+----------------+
        |          Unit / Property         |
        +----------------------------------+
```


### 19. Unit tests

| Componente | Pruebas clave |
|---|---|
| Quality | normalización, límites, pose, blur, iluminación |
| WeightStrategy | suma de pesos, monotonicidad, rangos |
| EvidenceAggregator | outliers, normalización, consistencia |
| SimilarityCalculator | simetría, identidad, vectores normalizados |
| CandidateFilter | TOP-K, delta, casos vacíos |
| VerificationScorer | métricas y fórmula |
| DecisionPolicy | cada gate y reason_codes |
| Policies | new identity / update identity |
| Repositories | CRUD, constraints, idempotencia |


### 20. Property-based / invariantes

- ||embedding||2 ≈ 1 después de normalización.
- 0 <= quality_score <= 1.
- 0 <= consistency_score <= 1.
- 0 <= positive_match_ratio <= 1.
- aggregation_weights suman aproximadamente 1.
- Una visita con el mismo event_id no puede duplicarse.
- Una persona no puede tener dos PRIMARY activos.
- CandidateFilter nunca devuelve más candidatos que recibe.

### 21. Integration tests

```

FaceSample[]
   -> EmbeddingGenerator
   -> EvidenceAggregator
   -> VectorSearchService
   -> SecondaryVerification
   -> DecisionEngine
```

Se usarán dobles/mocks para componentes externos cuando sea necesario, pero también habrá integración real con SQLite y con el motor vectorial elegido.


### 22. End-to-end tests

```

VIDEO / CAMERA FIXTURE
        |
        v
FULL PIPELINE
        |
        v
DecisionResult
        |
        v
DATABASE STATE
```

Los E2E validarán que una sesión completa produzca la identidad, visita y auditoría esperadas.


### 23. Dataset experimental y ground truth

El dataset experimental se construirá con participantes voluntarios y sesiones separadas temporalmente. Cada observación deberá poseer un ground truth externo al sistema.

| Elemento | Recomendación |
|---|---|
| Sujeto | ID de evaluación separado de Person_ID del sistema |
| Sesiones | Múltiples días/condiciones |
| Variabilidad | Pose, iluminación, lentes, expresión, distancia |
| Split | Separar calibración y evaluación final |
| Ground truth | Registro manual confiable por sesión |


### 24. Prohibición de leakage

Las muestras utilizadas para calibrar thresholds no deben reutilizarse como conjunto de evaluación final. La evaluación final debe permanecer congelada hasta cerrar los parámetros.


### 25. Métricas biométricas

```
FAR = false_accepts / impostor_attempts
FRR = false_rejects / genuine_attempts
TAR = true_accepts / genuine_attempts
```

También se analizarán ROC/DET y el punto operativo elegido según el costo relativo de falsos positivos y falsos negativos.


### 26. Matriz de decisión

| Ground truth | MATCH correcto | UNKNOWN/AMBIGUOUS | MATCH incorrecto |
|---|---|---|---|
| Persona conocida | True Accept | False Reject / Deferred | False Accept |
| Persona nueva | N/A | Correct reject / New identity | False Accept |


### 27. Benchmarks diseñados desde ahora

| Benchmark | Variables |
|---|---|
| Visión | resolución, FPS, detecciones, latencia |
| Embedding | CPU/GPU, batch_size, latency, throughput |
| Vector search | N identidades, top_k, índice, recall, latency |
| Pipeline | personas/min, T_e2e, queue depth, dropped tasks |
| Persistencia | writes/s, commit latency, contention |
| Escalabilidad | 1, 2, 4 cámaras simuladas |


### 28. Calibración posterior al prototipo

```

IMPLEMENT PROTOTYPE
      |
      v
COLLECT CALIBRATION DATA
      |
      v
FIT / SELECT PARAMETERS
      |
      v
FREEZE CONFIGURATION
      |
      v
RUN HELD-OUT EVALUATION
      |
      v
REPORT FAR / FRR / LATENCY
```

Esto confirma el orden acordado: primero diseño, después prototipo, luego benchmark y finalmente calibración experimental.


## Seguridad operativa y privacidad aplicada v1

La biometría facial se tratará como información sensible. La arquitectura aplica minimización, separación de responsabilidades, control de acceso y retención explícita. Para un despliegue real deberá revisarse la legislación aplicable y obtenerse la base legal correspondiente.


### 29. Threat model básico

| Amenaza | Mitigación |
|---|---|
| Acceso no autorizado a embeddings | RBAC, cifrado, separación de servicios |
| Robo de BD | cifrado en reposo y backups protegidos |
| Logs con biometría | política de logging sin vectores/imágenes |
| Manipulación de índice | DB como source of truth + rebuild + health checks |
| Replay / duplicación de eventos | event_id e idempotencia |
| Contaminación de identidad | IdentityUpdatePolicy conservadora |
| Modelo/config incompatible | versionado y validación de metadata |
| Retención indefinida | política de expiración y borrado |


### 30. Data classification

| Clase | Ejemplos | Tratamiento |
|---|---|---|
| Biométrico sensible | embeddings, muestras faciales | máxima protección |
| Operacional sensible | decisiones, session IDs, visitas | acceso restringido |
| Telemetría | latencias, métricas agregadas | sin biometría |
| Configuración | thresholds, versiones | integridad y auditoría |


### 31. Minimización de datos

- No guardar frames completos por defecto.
- No guardar face crops salvo modo de evaluación autorizado y con retención definida.
- No registrar embeddings completos en logs.
- Persistir métricas resumidas en lugar de matrices de similitud completas.
- Utilizar Person_ID seudónimo y separar cualquier dato identificativo externo si existiera.

### 32. EncryptionProvider

```
class EncryptionProvider(Protocol):
    def encrypt(self, plaintext: bytes, *, context: str) -> bytes: ...
    def decrypt(self, ciphertext: bytes, *, context: str) -> bytes: ...
```

El contrato permite cifrar embeddings o artefactos sensibles sin acoplar el dominio a una biblioteca concreta.


### 33. SecretProvider

```
class SecretProvider(Protocol):
    def get(self, name: str) -> str: ...
```

Secretos, claves o credenciales no deberán residir hardcodeados ni dentro de archivos versionados.


### 34. Access control

| Rol conceptual | Permisos |
|---|---|
| Runtime | lectura/escritura mínima requerida para operar |
| Evaluator | métricas y auditoría de pruebas |
| Administrator | configuración, mantenimiento e índice |
| Research/Debug | acceso temporal explícito a artefactos autorizados |


### 35. RetentionPolicy

```
class RetentionPolicy(Protocol):
    def should_delete(self, artifact_type: str, created_at: datetime) -> bool: ...
    def retention_days(self, artifact_type: str) -> int | None: ...
```

Las políticas concretas se definirán según el escenario real y la normativa. El prototipo mantendrá datos solo durante el periodo necesario para evaluación.


### 36. Borrado y derecho de eliminación

```

DELETE PERSON
    |
    +--> disable/remove embeddings
    +--> remove from vector index
    +--> delete/anonymize linked records per policy
    +--> invalidate caches
    +--> audit deletion
```

El índice vectorial nunca será la única copia: la eliminación se inicia en la fuente de verdad y luego se sincroniza.


### 37. Auditoría de seguridad

- Cambios de configuración.
- Creación, actualización, merge o eliminación de identidades.
- Reconstrucción del índice.
- Accesos administrativos.
- Fallos repetidos de descifrado o integridad.

### 38. Privacy modes

| Modo | Imágenes | Embeddings | Auditoría |
|---|---|---|---|
| Normal | No persistir | Sí, protegidos | Sí |
| Evaluation | Solo muestras autorizadas | Sí | Extendida |
| Benchmark | Fixtures controlados | Según benchmark | Técnica |


### 39. Diagrama de seguridad transversal

```

                   +--------------------------+
                   |      Application         |
                   +------------+-------------+
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
      +-------------+    +-------------+    +-------------+
      | Access      |    | Encryption  |    | Audit       |
      | Control     |    | Provider    |    | Logging     |
      +------+------+    +------+------+    +------+------+
             |                  |                  |
             +------------------+------------------+
                                |
                                v
                     +---------------------+
                     | Sensitive storage   |
                     | DB / embeddings     |
                     +---------------------+
```


## Revisión de completitud arquitectónica

Con los cuatro bloques transversales anteriores, el proyecto dispone de arquitectura funcional, contratos, persistencia, concurrencia, observabilidad, configuración, estrategia de pruebas y seguridad. El diseño queda suficientemente completo para comenzar la implementación sin decisiones estructurales críticas pendientes.

| Bloque | Estado |
|---|---|
| Objetivo y alcance | Definido |
| Adquisición física e iluminación | Definido |
| Pipeline de visión | Definido |
| Tracking / TrackSession | Definido |
| Quality / Sample Collector | Definido |
| ArcFace / embeddings | Definido |
| Evidence Aggregator | Definido |
| Vector search / verification | Definido |
| Decision Engine | Definido |
| Persistencia / auditoría | Definido |
| Servicios y repositorios | Definido |
| Orquestación / concurrencia | Definido |
| Observabilidad | Definido |
| Configuración / DI | Definido |
| Pruebas / validación | Definido |
| Seguridad / privacidad | Definido |
| Valores experimentales | Pendientes por diseño |
| Implementación real | Siguiente fase |
| Benchmarks | Después del prototipo |
| Calibración | Después de benchmarks/dataset |


### 40. Lo que deliberadamente queda pendiente

- Thresholds numéricos de calidad, similitud, margen y consistencia.
- Coeficientes exactos de funciones compuestas.
- top_k, candidate_delta y número óptimo de secundarios.
- queue_size, batch_size, número de workers, deadlines y retries.
- FAISS vs HNSW definitivo.
- CPU/GPU definitivo y hardware de producción.
- Distancia/resolución/FPS finales.
Estos elementos no constituyen huecos arquitectónicos: son parámetros empíricos cuya determinación antes de medir el prototipo sería metodológicamente incorrecta.


### 41. Secuencia final del proyecto

```

DESIGN COMPLETE
      |
      v
IMPLEMENT PROTOTYPE
      |
      v
UNIT + INTEGRATION + E2E
      |
      v
BENCHMARK
      |
      v
COLLECT CALIBRATION DATA
      |
      v
CALIBRATE PARAMETERS
      |
      v
HELD-OUT VALIDATION
      |
      v
FINAL REPORT / ITERATION
```
