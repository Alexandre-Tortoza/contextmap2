# Canonical Pipeline do ContextMap2

Este documento descreve o pipeline end-to-end alvo do ContextMap2, desde uma fonte registrada de sensores até o `ContextMapArtifact` final.

Ele documenta **o canonical pipeline planejado do ContextMap2**. As capabilities são implementadas por milestones independentes, portanto uma etapa descrita aqui pode ainda não estar disponível no código.

Para ownership e dependências, consulte [architecture.md](architecture.md). Para a semântica dos objetos que atravessam módulos, consulte [CONTRACTS.md](CONTRACTS.md). Para persistência e lineage, consulte [ARTIFACTS.md](ARTIFACTS.md).

## Visão geral

O pipeline não é uma única sequência linear. Depois da Ingestion, Percepção Visual e State Estimation podem processar a mesma sequência em branches independentes. A geometria persistente depende do estado estimado. A associação 2D↔3D reúne geometria e evidência visual. A partir daí surgem evidência fundida, entidades, identidades resolvidas, relações e o mapa final.

```mermaid
flowchart TD
    RAW[Raw recorded source]
    ING[Ingestion]
    SEQ[SequenceArtifact]

    VP[Visual Perception]
    PR[PerceptionRunArtifact]

    ST[State Estimation]
    TR[StateEstimationRunArtifact]

    GM[Geometric Mapping]
    GEO[GeometricMapArtifact]

    SA[Sensor Association]
    ASSOC[AssociationRunArtifact]

    P3D[Point Representation, optional]
    PREP[PointRepresentationRunArtifact]

    SF[Semantic Fusion]
    FUS[SemanticFusionRunArtifact]

    SM[Semantic Mapping]
    ENT[Semantic entity artifact]

    ER[Entity Resolution]
    RENT[EntityResolutionRunArtifact]

    SR[Spatial Relations]
    REL[SpatialRelationsRunArtifact]

    CM[Context Map Assembly]
    OUT[ContextMapArtifact]

    RAW --> ING --> SEQ
    SEQ --> VP --> PR
    SEQ --> ST --> TR
    SEQ --> GM
    TR --> GM --> GEO

    PR --> SA
    GEO --> SA
    TR --> SA
    SEQ --> SA
    SA --> ASSOC

    GEO --> P3D
    ASSOC -. optional selection/context .-> P3D
    P3D --> PREP

    ASSOC --> SF
    PREP -. optional evidence .-> SF
    SF --> FUS

    FUS --> SM --> ENT
    ENT --> ER --> RENT
    RENT --> SR --> REL

    GEO --> CM
    RENT --> CM
    REL --> CM
    CM --> OUT
```

## Estado atual da pipeline

O diagrama end-to-end acima é o alvo do canonical pipeline. Na `dev`, o caminho materializado termina hoje em `SemanticFusionRunArtifact` (com a estrutura 3D opcional de `PointRepresentationRunArtifact` como evidência): a Ingestion alimenta Visual Perception e State Estimation, o Geometric Mapping consome a trajetória, o Sensor Association ancora a percepção na geometria, a Point Representation descreve a estrutura 3D local e a Semantic Fusion acumula a evidência multi-vista sem criar identidade de objeto:

```mermaid
flowchart LR
    RAW["Fonte registrada"] --> ING["Ingestion"]
    ING --> SEQ["SequenceArtifact"]
    SEQ --> VP["Visual Perception Core"]
    VP --> PRA["PerceptionRunArtifact"]
    SEQ --> ST["State Estimation"]
    ST --> TRA["StateEstimationRunArtifact"]
    SEQ --> GM["Geometric Mapping"]
    TRA --> GM
    GM --> MAPA["GeometricMapArtifact"]
    PRA --> SA["Sensor Association"]
    MAPA --> SA
    TRA --> SA
    SA --> ASA["SensorAssociationRunArtifact"]
    MAPA --> PTR["Point Representation<br/>(opcional)"]
    PTR --> PTRA["PointRepresentationRunArtifact"]
    ASA --> FUS["Semantic Fusion"]
    PTRA -.-> FUS
    FUS --> FUSA["SemanticFusionRunArtifact"]
    FUSA -. próximo estágio ainda não integrado .-> FUT["Semantic Mapping<br/>+ downstream"]
```

Essa distinção é obrigatória ao ler este documento: seções posteriores descrevem o contrato arquitetural esperado, mas apenas Ingestion, Visual Perception Core, State Estimation, Geometric Mapping, Sensor Association, Point Representation (opcional) e Semantic Fusion possuem implementação consolidada neste ponto.

## Regra fundamental

Cada estágio consome **contratos públicos e artefatos imutáveis**. Um estágio posterior não deve reabrir detalhes privados do backend anterior nem corrigir silenciosamente o resultado upstream.

Exemplos:

- Sensor Association consome `Region2D`, `VisualFeature`, `SemanticClaim`, geometria e pose, mas não importa classes internas de SAM, DINO, Gemini ou FAST-LIO.
- Semantic Fusion consome `SpatialObservation` e evidências referenciadas, mas não executa novamente a VLM para corrigir um label.
- Spatial Relations consome entidades resolvidas, mas não corrige Entity Resolution para conseguir uma relação conveniente.

## 0. Runtime e plano de execução

Antes de executar modelos ou transformações pesadas, `runtime` resolve a configuração do experimento.

```mermaid
flowchart LR
    CFG[Configuração solicitada]
    PRESET[Preset/version]
    BACK[Backends selecionados]
    DAG[Resolved execution DAG]
    VALID[Dependency validation]
    RUN[Execution]

    CFG --> PRESET --> BACK --> DAG --> VALID --> RUN
```

### Responsabilidade

Runtime deve:

- carregar configuração efetiva;
- selecionar implementações concretas;
- construir dependências;
- validar o DAG de execução;
- selecionar explicitamente artifacts/runs upstream;
- ordenar estágios;
- decidir reutilização ou recomputação de artefatos imutáveis;
- registrar o plano resolvido e seus fingerprints.

Runtime não deve conter segmentação, projeção, fusão, entity resolution ou regras de relações.

### Saída

A execução deve registrar um plano reproduzível que identifique:

```text
stage
selected inputs
selected backend
configuration digest
expected output artifact
reuse/recompute decision
dependency edges
```

## 1. Ingestion

Ingestion transforma dados específicos de uma fonte em observações canônicas e em um `SequenceArtifact` independente de ROS, dataset ou formato de gravação.

```mermaid
flowchart TD
    SRC["ROS 1 bag / ROS 2 bag / fonte registrada"] --> AD["SourceAdapter"]
    AD --> OBS["SourceObservation[]"]
    AD --> CAL["CalibrationSet"]
    OBS --> VAL["Validação estrutural"]
    OBS --> SYNC["synchronize()"]
    SYNC --> DIAG["SynchronizationDiagnostics"]
    OBS --> ART["SequenceArtifact"]
    CAL --> ART
    DIAG --> ART
    ART --> SEL["resolve_selection()<br/>seleção / replay"]
    SEL --> DOWN["capabilities downstream"]
```

O adapter decodifica e normaliza a fonte. Sincronização não altera a identidade das observações físicas e não cria truth persistente; ela produz agrupamentos e diagnósticos auditáveis. O artefato persiste as observações normalizadas, calibração/provenance/diagnósticos quando disponíveis e pode ser reaberto sem a fonte ROS original.

### Contratos implementados

- `SourceObservation`, com `ImageObservation`, `LidarObservation`, `ImuObservation` e `ExternalPoseMeasurement`;
- `CalibrationSet`, modelos pinhole/fisheye e `RigidTransform`;
- `ProcessingObservation`, `SynchronizationDiagnostics` e `DroppedEvent`;
- `SequenceSelection` e `selection_identity()`;
- `SequenceArtifactWriter` / `SequenceArtifactReader`;
- `SequenceProvenance`, validação e diagnósticos.

`ExternalPoseMeasurement` continua sendo evidência de entrada. Ela não é automaticamente um `PoseEstimate` do mapa.

### Saída implementada

`SequenceArtifact`, persistido de forma imutável e reabrível.

Detalhes: [documentação de Ingestion](../src/contextmap/ingestion/docs/README.md), especialmente [contracts](../src/contextmap/ingestion/docs/contracts.md), [artifact](../src/contextmap/ingestion/docs/artifact.md), [synchronization](../src/contextmap/ingestion/docs/synchronization.md) e [calibration](../src/contextmap/ingestion/docs/calibration.md).

### Não responsabilidade

Ingestion não faz percepção semântica, state estimation canônica, projeção 2D↔3D, fusão ou criação de entidades.

## 2. Branch de Visual Perception

Visual Perception transforma uma observação física de imagem em evidência visual canônica para um `PerceptionRun`, sem decidir projeção 2D↔3D, identidade persistente de entidade ou fusão multi-view.

### Topologia implementada: `canonical/1`

```mermaid
flowchart LR
    IMG["image_preparation<br/>PreparedImage"]
    REG["region_discovery<br/>Region2D[]"]
    DENSE["dense_feature_extraction<br/>VisualFeature[] DENSE"]
    RFEAT["region_feature_extraction<br/>VisualFeature[] REGION"]
    SCENE["scene_interpretation<br/>SceneContext | None"]
    RINT["region_interpretation<br/>SemanticClaim[]"]
    RESULT["PerceptionResult"]

    IMG --> REG
    IMG --> DENSE
    IMG --> RFEAT
    REG --> RFEAT
    IMG --> SCENE
    IMG --> RINT
    REG --> RINT
    REG --> RESULT
    DENSE --> RESULT
    RFEAT --> RESULT
    SCENE --> RESULT
    RINT --> RESULT
```

`image_preparation` é atualmente um estágio fonte: `PreparedImage` é fornecida externamente ao grafo. `resolve_pipeline()` valida o preset antes de construir backends; os backends são resolvidos uma vez e reutilizados entre observações. `execute_stage_graph()` isola falhas por branch: um estágio `FAILED` faz apenas seus dependentes ficarem `SKIPPED`, enquanto branches independentes continuam contribuindo evidência.

### Contratos implementados

- `PreparedImage`;
- `Region2D`;
- `VisualFeature` com scopes `DENSE`, `GLOBAL` e `REGION`;
- `SemanticClaim`, incluindo hipóteses `PRIMARY` e `ALTERNATIVE`;
- `SceneContext`;
- `SemanticSupport`, produzido por `SemanticScorer` sem mutar a claim;
- `PerceptionRun` e `PerceptionResult`;
- `BackendProvenance`.

`RegionId`, `FeatureId` e `ClaimId` são locais ao `PerceptionResult`. Reprocessar a mesma `SourceObservation` em outro run cria outro `PerceptionResult`; não cria uma nova observação física e não funde resultados anteriores.

### Region Discovery implementado

```mermaid
flowchart LR
    PI["PreparedImage"] --> PASSES["full-frame / tiles / scales"]
    PASSES --> RD["SAM2 | SAM3 | Florence-2"]
    RD --> RC["RegionCandidate[]"]
    RC --> N["normalize / merge"]
    N --> GF["Geometry Freeze"]
    GF --> R2D["Region2D[]"]
    R2D --> RESULT["PerceptionResult"]
```

Region Discovery agora possui adapters concretos para SAM2, SAM3 e Florence-2, além de image preparation auditável, tiling/scale com validação da imagem materializada, remapeamento global, filtros/constraints, merge e diagnostics. O output do estágio continua sendo o `Region2D` canônico; candidatos e decisões intermediárias permanecem evidence/audit da execução. Detalhes: [Region Discovery](../src/contextmap/visual_perception/docs/region-discovery.md) e [protocolo de avaliação](../src/contextmap/evaluation/docs/region-discovery.md).

### Feature Extraction core implementado

```mermaid
flowchart LR
    PI["PreparedImage"] --> FE["FeatureExtractor"]
    REG["Region2D[]"] --> FE
    FE --> VF["VisualFeature[]"]
    VF --> ES["EmbeddingSpace<br/>compatibilidade exata"]
    VF --> STORE["FeatureStore<br/>payload lazy"]
    VF --> DFM["DenseFeatureMap<br/>sampling explícito"]
    DFM --> POOL["pool_region_feature()"]
    REG --> POOL
    DFM -. preset alternativo .-> ENH["FeatureResolutionEnhancement"]
```

Feature Extraction já possui contratos e infraestrutura backend-neutral para os três scopes de `VisualFeature`, identidade de `EmbeddingSpace`, persistência e integridade de payloads, geometria explícita de dense feature maps, pooling mask-aware, diagnostics e avaliação. `dense_feature_extraction` e `region_feature_extraction` já fazem parte de `CANONICAL_PRESET_V1` via o port `FeatureExtractor`.

Isso não equivale a declarar modelos concretos como suportados. DINOv2, DINOv3, CLIP e AlphaCLIP ainda não possuem adapters integrados na `dev`; o core e a CI usam fakes determinísticos para validar os contratos. `feature_resolution_enhancement` é uma capability conhecida pelo DAG, mas permanece opcional, fora do preset canônico e sem backend aprendido integrado.

Detalhes: [Feature Extraction](../src/contextmap/visual_perception/docs/feature-extraction.md), [espaço de embedding](../src/contextmap/visual_perception/docs/embedding_space.md), [feature store](../src/contextmap/visual_perception/docs/feature_store.md) e [protocolo de avaliação](../src/contextmap/evaluation/docs/feature_extraction.md).

### Persistência e leitura multi-run

```mermaid
flowchart LR
    RESULT["PerceptionResult[]"] --> W["PerceptionRunWriter"]
    W --> ART["PerceptionRunArtifact"]
    ART --> R["PerceptionRunReader"]
    R --> SET["PerceptionEvidenceSet"]
    SET -. preserva resultados por run .-> FUT["Semantic Fusion futura"]
```

`PerceptionEvidenceSet` exige seleção explícita de runs e agrupa resultados pela observação física sem escolher label vencedor, combinar confidences ou associar regiões como o mesmo objeto.

### Ainda não materializado na integração canônica

Os seguintes elementos aparecem na arquitetura alvo ou como variation points já definidos, mas ainda não possuem integração concreta na `dev` ou não fazem parte de `CANONICAL_PRESET_V1`:

- adapters concretos DINOv2, DINOv3, CLIP e AlphaCLIP para `FeatureExtractor`;
- backend aprendido de `FeatureResolutionEnhancement` e sua inclusão no preset canônico;
- backends reais de Semantic Interpretation;
- integração de `SemanticScorer` como estágio do DAG;
- semantic refinement;
- integração end-to-end com State Estimation, geometria e Sensor Association.

Implementar um port ou backend não o adiciona automaticamente ao preset. A inclusão exige topologia, inputs/outputs, validação e avaliação explícitas.

Detalhes: [documentação de Visual Perception](../src/contextmap/visual_perception/docs/README.md), [pipeline](../src/contextmap/visual_perception/docs/pipeline.md), [contracts](../src/contextmap/visual_perception/docs/contracts.md), [Region Discovery](../src/contextmap/visual_perception/docs/region-discovery.md), [Feature Extraction](../src/contextmap/visual_perception/docs/feature-extraction.md), [service](../src/contextmap/visual_perception/docs/service.md), [run artifact](../src/contextmap/visual_perception/docs/run_artifact.md) e [evidence set](../src/contextmap/visual_perception/docs/evidence_set.md).

## 3. Branch de State Estimation

State Estimation produz a trajetória canônica usada para colocar observações no frame global do mapa.

```mermaid
flowchart TD
    SEQ[SequenceArtifact]
    PRE[Geometry/frame preflight]
    EXT[External pose backend]
    FL[FAST-LIO backend]
    P[PoseEstimate]
    T[Trajectory]
    L[Pose lookup / interpolation]
    A[StateEstimationRunArtifact]

    SEQ --> PRE
    PRE --> EXT --> P
    PRE --> FL --> P
    P --> T --> L --> A
```

### Entrada

Dependendo do backend:

- external pose measurements;
- LiDAR + IMU;
- calibration/extrinsics requeridos;
- selection temporal.

### Backends

Implementados atrás do port `StateEstimator`:

- `ExternalPose`, valida e normaliza uma pose externa canônica; é o baseline de geometria;
- `FAST-LIO`, produz pose LiDAR-inertial sem vazar tipos do backend, com o processo isolado atrás de um runner. Não há fallback de um backend para o outro. A execução de referência com o FAST-LIO instalado ainda está pendente.

### Contratos

`PoseEstimate` declara explicitamente:

```text
timestamp / clock_id
parent_frame
child_frame
translation_m
orientation (x, y, z, w)
covariance?            # ausente quando o backend não a reporta
validity
provenance
```

`Trajectory` reúne poses com frame de referência, frame do corpo, gaps registrados e provenance; limites de tempo e resumo de qualidade são derivados.

### Time alignment

Downstream solicita pose no timestamp de uma observação através de uma política explícita:

```text
EXACT
NEAREST        (tolerância de distância ao redor da pose)
INTERPOLATED   (tolerância opcional do intervalo entre as vizinhas)
```

Rejeições (fora do intervalo, sem correspondência exata, tolerância excedida, interpolação através de um gap) são dados contáveis, nunca extrapolação silenciosa. Uma pose interpolada preserva os estimates que a originaram e é distinguível de uma pose estimada.

### Preflight

Antes do estimator, valida por capability:

- frames e extrínsecos necessários (frame graph estático com verificação de caminhos redundantes);
- direção do transform e rotação válida;
- clock domains;
- identidade da calibração.

Uma capability posterior sem pré-requisitos (por exemplo, câmera sem modelo de intrínsecos) nunca bloqueia uma execução LiDAR-inertial: aparece como prontidão downstream. Um preflight `BLOCKED` impede que o estimador inicie e que um run seja persistido.

Saída persistida: `StateEstimationRunArtifact`.

Detalhes: [documentação de State Estimation](../src/contextmap/state_estimation/docs/README.md), [contratos](../src/contextmap/state_estimation/docs/contracts.md), [lookup](../src/contextmap/state_estimation/docs/lookup.md), [backends](../src/contextmap/state_estimation/docs/backends.md), [preflight](../src/contextmap/state_estimation/docs/preflight.md), [artifact](../src/contextmap/state_estimation/docs/artifact.md) e [avaliação](../src/contextmap/evaluation/docs/state_estimation.md).

## 4. Geometric Mapping

Geometric Mapping transforma observações geométricas locais em uma geometria persistente no frame global do mapa.

```mermaid
flowchart TD
    S[Selected LiDAR/depth observations]
    D[Motion correction state]
    P[Pose lookup]
    X[Static extrinsic]
    T[Source → map transform]
    A[Persistent accumulation]
    I[Spatial index / bounds]
    M[GeometricMapArtifact]

    S --> D --> T
    P --> T
    X --> T
    T --> A --> I --> M
```

### Entrada

- `SequenceArtifact`;
- selection;
- `StateEstimationRunArtifact`;
- static extrinsics/calibration;
- LiDAR/depth payloads.

### Motion correction

Cada scan precisa declarar estado:

```text
raw / uncorrected
motion-corrected / deskewed
unknown
```

Nunca inferir `deskewed=true` apenas porque FAST-LIO foi usado.

### Transformação source→map

Para um exemplo LiDAR:

```text
P_map = T_map_body(t) · T_body_lidar · P_lidar
```

A cadeia real depende do rig, mas frame source/target e cada transform precisam ser explícitos.

O ponto persistente preserva tanto coordenadas locais quanto globais e lineage do transform.

### Persistência da geometria

O mapa geométrico inicial pode ser point-based. Ele deve fornecer referências estáveis dentro do artefato:

```text
GeometryReference(map_id, geometry_id)
```

Downstream referencia geometria em vez de copiar XYZ repetidamente.

### Spatial access

O contrato deve permitir operações como:

```text
get(GeometryReference)
iter_geometry(...)
query_bounds(Bounds3D)
map_bounds()
```

A implementação concreta do índice é privada.

Saída persistida: `GeometricMapArtifact`.

Detalhes: [documentação de Geometric Mapping](../src/contextmap/geometric_mapping/docs/README.md), [contratos](../src/contextmap/geometric_mapping/docs/contracts.md), [correção de movimento](../src/contextmap/geometric_mapping/docs/motion-correction.md), [inputs](../src/contextmap/geometric_mapping/docs/inputs.md), [transformação](../src/contextmap/geometric_mapping/docs/transformation.md), [acumulação](../src/contextmap/geometric_mapping/docs/accumulation.md), [acesso espacial](../src/contextmap/geometric_mapping/docs/spatial-access.md), [artifact](../src/contextmap/geometric_mapping/docs/artifact.md) e [validação](../src/contextmap/evaluation/docs/geometric_mapping.md).

## 5. Sensor Association

Sensor Association conecta a geometria persistente às evidências visuais de uma observação.

É aqui que o sistema faz a ponte 2D↔3D.

```mermaid
flowchart TD
    PM[P_map]
    POSE[T_map_body at RGB timestamp]
    EXT[body ↔ camera extrinsic]
    CAM[Camera model / intrinsics]
    RAW[Raw camera pixel]
    IMG[PreparedImage transform chain]
    PIX[PreparedImage pixel]
    VIS[Visibility / occlusion]
    REG[Region2D membership]
    DEN[Dense feature sampling]
    SO[SpatialObservation]

    PM --> RAW
    POSE --> RAW
    EXT --> RAW
    CAM --> RAW
    RAW --> IMG --> PIX --> VIS
    VIS --> REG --> SO
    VIS --> DEN --> SO
```

### Projection 3D→2D

A cadeia conceitual é:

```text
P_map
→ P_camera at t_rgb
→ raw camera pixel
→ PreparedImage pixel
```

Ela deve usar:

- trajectory/pose selecionado;
- extrinsic correto;
- camera model correto, incluindo fisheye quando declarado pela calibração;
- transformações de imagem aplicadas por Visual Perception.

Não é permitido assumir que raw image e `PreparedImage` usam as mesmas coordenadas.

### Visibility e occlusion

Pontos projetáveis ainda podem estar atrás de superfícies visíveis. A política de visibility/occlusion decide quais suportes 3D podem contribuir para a evidência daquela imagem.

### Region membership

Para cada ponto visível, verificar membership em `Region2D` congeladas.

Um ponto pode pertencer a zero, uma ou várias regiões. Overlap não é resolvido semanticamente aqui.

### Dense feature sampling

Quando um dense `VisualFeature` está disponível:

```text
GeometryReference
→ prepared-image pixel
→ feature-grid coordinate
→ local visual feature evidence
```

Region-level embeddings continuam associados à região, não são fingidos como medições independentes por ponto.

### Saída

`SpatialObservation` representa evidência visual ancorada em suporte 3D persistente.

Saída persistida: `AssociationRunArtifact`, implementado como `SensorAssociationRunArtifact`.

Detalhes: [documentação de Sensor Association](../src/contextmap/sensor_association/docs/README.md), [contratos](../src/contextmap/sensor_association/docs/contracts.md), [modelos de câmera](../src/contextmap/sensor_association/docs/camera_models.md), [cadeia de projeção](../src/contextmap/sensor_association/docs/projection_chain.md), [visibilidade](../src/contextmap/sensor_association/docs/visibility.md), [pertencimento à máscara](../src/contextmap/sensor_association/docs/membership.md), [amostragem densa](../src/contextmap/sensor_association/docs/dense_sampling.md), [qualidade da observação](../src/contextmap/sensor_association/docs/quality.md), [diagnósticos](../src/contextmap/sensor_association/docs/diagnostics.md), [artifact](../src/contextmap/sensor_association/docs/artifact.md) e [validação](../src/contextmap/evaluation/docs/sensor_association.md).

## 6. Point Representation, opcional

Point Representation descreve estrutura geométrica local sem misturá-la com evidência visual ou semântica.

```mermaid
flowchart LR
    G[GeometryReference]
    N[Neighborhood support]
    E[PointEncoder]
    P[PointRepresentation]
    A[PointRepresentationRunArtifact]

    G --> N --> E --> P --> A
```

### Support extraction

Políticas iniciais podem incluir:

- radius support;
- k-nearest support;
- voxel/cell support quando justificado.

O suporte precisa preservar quais `GeometryReference` participaram.

### Backends

Baseline:

- descriptor geométrico determinístico.

Opcional/experimental:

- PTv3 learned 3D representation.

PTv3 não é requisito automático do canonical pipeline. Seu uso deve ser justificado por avaliação/ablation.

No estado atual, o descritor determinístico está implementado e o PTv3 tem apenas a **fronteira** (`PTv3PointEncoder` atrás de um `PTv3Runtime` injetável, sem torch nem pesos no ambiente de desenvolvimento): nenhuma execução real foi feita e a comparação contra `off` e contra o descritor, com custo e VRAM reais, continua pendente.

### Saída

`PointRepresentation` + `RepresentationSpace`.

Saída persistida: `PointRepresentationRunArtifact`.

Detalhes: [documentação de Point Representation](../src/contextmap/point_representation/docs/README.md), [contratos](../src/contextmap/point_representation/docs/contracts.md), [extração de suporte](../src/contextmap/point_representation/docs/support-extraction.md), [execução](../src/contextmap/point_representation/docs/execution.md), [descritor geométrico](../src/contextmap/point_representation/docs/geometric-descriptor.md), [PTv3](../src/contextmap/point_representation/docs/ptv3.md), [artifact](../src/contextmap/point_representation/docs/artifact.md) e [avaliação](../src/contextmap/evaluation/docs/point_representation.md).

## 7. Semantic Fusion

Semantic Fusion acumula evidência de múltiplas observações físicas sobre suporte espacial compatível, sem ainda criar identidade persistente de objeto.

```mermaid
flowchart TD
    A[Selected AssociationRunArtifacts]
    G[Group by physical SourceObservation]
    S[Build FusionSupport]
    C[Assemble typed evidence channels]
    F[Baseline fusion policy]
    U[Ambiguity / conflict / abstention]
    E[FusedEvidence]

    A --> G --> S --> C --> F --> U --> E
```

### Regra de correlação

Três inferências sobre a mesma imagem são uma observação física com três resultados correlacionados:

```text
frame-0120
├── run A
├── run B
└── run C
```

Isso não equivale a três frames fisicamente distintos.

Fusion deve manter separados:

```text
physical_observation_count
inference_result_count
```

### FusionSupport

Agrupa `SpatialObservation` com suporte 3D suficientemente compatível para acumular evidência.

`FusionSupport` não significa automaticamente "mesmo objeto".

### Canais de evidência

Podem participar, conforme policy/configuração:

- `SemanticClaim`;
- `SemanticSupport`;
- visual feature references;
- visibility/coverage;
- optional `PointRepresentation`;
- temporal metadata.

Os canais mantêm semânticas e escalas separadas.

### Política baseline

A primeira policy deve ser determinística e preservar:

- hypotheses concorrentes;
- primary/alternative roles;
- scored e unscored evidence;
- abstentions;
- conflicts;
- número de observações físicas;
- contribution trace.

Não reduzir destrutivamente tudo a um único label.

### Estado implementado

- **Agrupamento por observação física** (`physical-observation-grouping-v1`) sobre uma seleção **explícita** de runs de percepção; a contagem de frames físicos, de resultados de inferência, de runs e de variantes de backend ficam separadas.
- **`FusionSupport`** por sobreposição de geometria (`geometry-jaccard-support-v1`, limiares declarados e sem valores padrão); observações com pouca geometria vão para uma lista de excluídas explícita.
- **Política baseline** (`baseline-evidence-accumulation-v1`): uma contribuição por observação espacial, hipóteses por chave de label tipográfica, stances (`SUPPORTING`, `CONFLICTING`, `AMBIGUOUS`, `ABSTAINING`) e sinais tipados, sem ranking e sem ponderar por qualidade.
- **Incerteza reportada, nunca resolvida:** contradição, ambiguidade, empate e evidência insuficiente com a evidência exata; abstenção só para labels configurados; refinamentos (`pallet` / `wooden pallet`) **não** são reconhecidos.
- **Canais tipados** (`semantic_claims`, `semantic_scores`, `visual_features`, `observation_quality`, `geometry_support`, `point_representation`) declarados pela política; um canal nunca é ativado só porque há dados.
- **Política opcional ciente de qualidade** (`quality-aware-evidence-accumulation-v1`), com peso inspecionável e fator neutro registrado; **não** é o padrão e nenhuma decisão de adotá-la foi tomada.

### Saída

`FusedEvidence`.

Saída persistida: `SemanticFusionRunArtifact`.

Detalhes: [documentação de Semantic Fusion](../src/contextmap/semantic_fusion/docs/README.md), [contratos](../src/contextmap/semantic_fusion/docs/contracts.md), [agrupamento](../src/contextmap/semantic_fusion/docs/grouping.md), [suporte](../src/contextmap/semantic_fusion/docs/support.md), [acumulação](../src/contextmap/semantic_fusion/docs/accumulation.md), [política ciente de qualidade](../src/contextmap/semantic_fusion/docs/quality-aware.md), [artifact](../src/contextmap/semantic_fusion/docs/artifact.md) e [validação](../src/contextmap/evaluation/docs/semantic_fusion.md).

## 8. Semantic Mapping

Semantic Mapping materializa evidência fundida como entidades semânticas persistentes dentro de um semantic-map artifact.

```mermaid
flowchart TD
    F[FusedEvidence]
    G[EntityGeometry]
    S[EntitySemanticState]
    T[EntityTemporalState]
    E[Evidence linkage]
    ENT[Entity]

    F --> G
    F --> S
    F --> T
    F --> E
    G --> ENT
    S --> ENT
    T --> ENT
    E --> ENT
```

### Entity

Uma entidade conecta:

- geometry support real, por `GeometryReference`;
- centroid/bounds/extent derivados;
- semantic hypotheses e alternatives;
- attributes/properties com provenance;
- conflicts/ambiguity;
- first_seen/last_seen;
- physical observation count;
- inference result count;
- referências para a evidência upstream.

Ela não deve ser reduzida a `{id, label, confidence, xyz}`.

### Saída

Um artifact de entidades semânticas persistidas.

Nesse ponto, entidades podem ainda representar fragmentos/duplicatas do mesmo objeto físico. Essa decisão pertence a Entity Resolution.

## 9. Entity Resolution

Entity Resolution decide quando entidades semânticas de origem devem permanecer distintas, ser agrupadas ou ficar não resolvidas.

```mermaid
flowchart TD
    E[Source Entities]
    C[Candidate retrieval]
    M[Match evidence]
    D[Resolution decision]
    R[ResolvedEntity materialization]

    E --> C --> M --> D --> R
```

### Evidência de matching

A policy pode comparar canais tipados, quando disponíveis:

- geometry overlap/proximity;
- semantic compatibility;
- visual appearance;
- temporal compatibility;
- optional point representations.

Nenhum canal deve ser escondido em um score opaco sem definição.

### Decisões

Estados esperados incluem conceitos equivalentes a:

```text
MATCH
DISTINCT
UNRESOLVED
```

`UNRESOLVED` é válido quando a evidência é insuficiente ou conflitante.

### Materialização

Um `ResolvedEntity` preserva:

- source entity refs;
- merge lineage;
- resolution decisions;
- geometry union/reference;
- semantic alternatives/conflicts;
- temporal state deduplicado por observação física.

As entidades originais permanecem imutáveis.

Saída persistida: `EntityResolutionRunArtifact`.

## 10. Spatial Relations

Spatial Relations descreve relações entre **entidades resolvidas**.

```mermaid
flowchart TD
    E[Resolved entities]
    C[Relation candidates]
    G[Geometric predicates]
    K[Contact/support evidence]
    S[Optional semantic relation evidence]
    D[Relation decision policy]
    R[Relation]

    E --> C
    C --> G --> D
    C --> K --> D
    C --> S --> D
    D --> R
```

### Candidate generation

Reduz pares irrelevantes através de critérios geométricos explícitos, sem decidir que uma relação é verdadeira.

### Relações geométricas

Famílias planejadas incluem, quando geometricamente definíveis:

```text
NEXT_TO
ABOVE / BELOW
IN_FRONT_OF / BEHIND
INSIDE / CONTAINS
INTERSECTS
```

### Contact/support

Quando a geometria suporta a inferência:

```text
TOUCHING
ON_TOP_OF
LEANING_AGAINST
```

Essas relações exigem medições, não apenas labels semânticos.

### Reconciliation

Evidência geométrica e semântica permanecem separadas. Uma policy conservadora produz algo equivalente a:

```text
SUPPORTED
REJECTED
UNRESOLVED
```

Relações inversas e simétricas precisam permanecer consistentes.

### Saída

`Relation` + `RelationEvidence`.

Saída persistida: `SpatialRelationsRunArtifact`.

## 11. Context Map Assembly

A etapa final monta o produto público do repositório.

```mermaid
flowchart TD
    G[GeometricMapArtifact]
    E[EntityResolutionRunArtifact]
    R[SpatialRelationsRunArtifact]
    L[Lineage / metadata]
    C[ContextMap]
    A[ContextMapArtifact]

    G --> C
    E --> C
    R --> C
    L --> C
    C --> A
```

### ContextMap

O contrato de alto nível contém conceitualmente:

```text
ContextMap
├── context_map_id
├── schema_version
├── metadata
├── geometry_ref
├── resolved entities / refs
├── relations / refs
├── indexes / capability metadata
└── provenance / lineage refs
```

### Metadata espacial

O mapa deve declarar explicitamente:

- map frame;
- units;
- coordinate convention;
- origin/anchor semantics;
- bounds;
- temporal/selection extent;
- source sequences;
- capabilities presentes.

Um map frame local de estimator não deve ser confundido com um frame global/geodeticamente alinhado.

### Portabilidade

O `ContextMapArtifact` deve ser legível sem:

- ROS;
- FAST-LIO;
- SAM/DINO/CLIP;
- VLM SDKs;
- CUDA;
- viewer específico.

Consumidores precisam apenas do schema, payloads e dependências contratuais explicitamente referenciadas.

## 12. Artefatos ao longo do pipeline

| Estágio | Artefato principal | Consumo downstream |
| --- | --- | --- |
| Ingestion | `SequenceArtifact` | perception, state estimation, geometry |
| Visual Perception | `PerceptionRunArtifact` | sensor association, evaluation |
| State Estimation | `StateEstimationRunArtifact` | geometry, sensor association |
| Geometric Mapping | `GeometricMapArtifact` | association, point representation, final map |
| Sensor Association | `AssociationRunArtifact` | semantic fusion |
| Point Representation | `PointRepresentationRunArtifact` | optional semantic fusion / resolution evidence |
| Semantic Fusion | `SemanticFusionRunArtifact` | semantic mapping |
| Semantic Mapping | semantic entity artifact | entity resolution |
| Entity Resolution | `EntityResolutionRunArtifact` | spatial relations, final map |
| Spatial Relations | `SpatialRelationsRunArtifact` | final map |
| Context Map Assembly | `ContextMapArtifact` | external consumers |

Todos esses artefatos são tratados como imutáveis. Uma nova execução produz um novo artifact/run identity.

## 13. Lineage ponta a ponta

Uma entidade final deve conseguir explicar sua origem sem copiar todos os payloads upstream.

```mermaid
flowchart RL
    E[ResolvedEntity]
    SE[Source Entity]
    F[FusedEvidence]
    SO[SpatialObservation]
    PR[PerceptionResult]
    R[Region2D / SemanticClaim]
    IMG[SourceObservation RGB]

    EG[EntityGeometry]
    GR[GeometryReference]
    GM[GeometricMap]
    LIDAR[SourceObservation LiDAR]
    POSE[PoseEstimate]

    E --> SE --> F --> SO --> PR --> R --> IMG
    E --> EG --> GR --> GM --> LIDAR
    GM --> POSE
```

Uma relação final deve ser rastreável a:

```text
Relation
→ RelationEvidence
→ ResolvedEntity subject/object
→ source entities
→ geometry + fused evidence
→ source sensor observations
```

## 14. Reuso e recomputação

Como artifacts intermediários são imutáveis, mudar uma parte da pipeline não deve obrigar recomputação de tudo.

Exemplo:

```text
mudar prompt/VLM
    invalidates: perception-dependent downstream
    preserves: state estimation + geometric map when inputs unchanged
```

```text
mudar FAST-LIO configuration
    invalidates: state estimation + geometry + association + downstream
    may preserve: perception over the same canonical RGB selection
```

Reuso só é válido quando upstream identities, schema versions, selections e configuração relevante são compatíveis.

## 15. Validação por estágio

Cada capability possui métricas próprias. O projeto evita um único score opaco de qualidade.

Exemplos:

| Capability | Exemplos de validação |
| --- | --- |
| Ingestion | integridade, sincronização, calibração, indexação |
| Region Discovery | IoU, coverage, over/under-segmentation, runtime |
| Feature Extraction | spatial alignment, payload round-trip, compatibility |
| Semantic Interpretation | conceito, hallucination, ambiguity, parsing |
| State Estimation | ATE/RPE quando referência válida existe, transform consistency |
| Geometric Mapping | source→map accuracy, scan consistency, reproducibility |
| Sensor Association | reprojection error, occlusion, mask membership |
| Semantic Fusion | multi-view consistency, ambiguity retention, repeated-inference regression |
| Entity Resolution | false merge, missed merge, unresolved rate |
| Spatial Relations | precision/recall por predicate, consistency, unresolved rate |
| Artifact | schema/integrity/lineage closure |

Qualidade e performance são relatadas separadamente.

## 16. Invariantes do pipeline

As seguintes regras atravessam todas as etapas:

1. `SourceObservation` representa evidência física, não uma execução de modelo.
2. Reexecutar um frame cria nova inferência, não nova observação física.
3. `Region2D` é geometria 2D local ao resultado de percepção, não Entity ID.
4. Geometria 3D persistente possui coordenada global autoritativa e provenance de transform.
5. Semantic confidence, scorer support e fusion support não são o mesmo conceito.
6. Missing/unscored evidence nunca é convertido silenciosamente para zero ou um.
7. Debug files nunca são contratos públicos downstream.
8. Artifacts persistidos são imutáveis.
9. Runs upstream são selecionados explicitamente, nunca agregados implicitamente porque existem no workspace.
10. Backend-native objects não atravessam fronteiras de capability.
11. Falha de uma etapa deve permanecer atribuível à etapa responsável.
12. O mapa final preserva referências suficientes para auditoria sem exigir que consumidores carreguem modelos de percepção.
