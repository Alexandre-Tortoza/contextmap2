# Artefatos, lineage e reprodutibilidade

Este documento descreve como o ContextMap2 persiste resultados intermediários e o mapa final.

A regra base é simples:

> Cada estágio produz um artefato imutável, versionado e auditável. Reexecutar um estágio gera um novo resultado, não sobrescreve o anterior.

Para o fluxo que produz esses artefatos, consulte [PIPELINE.md](PIPELINE.md). Para os contratos armazenados neles, consulte [CONTRACTS.md](CONTRACTS.md).

## Por que artifacts são parte da arquitetura

O projeto é uma pipeline de pesquisa. É necessário conseguir:

- executar percepção várias vezes sobre os mesmos frames;
- trocar um backend sem destruir o baseline;
- reutilizar state estimation e geometria quando apenas a semântica mudou;
- comparar outputs antigos e novos;
- localizar a origem de um erro downstream;
- abrir resultados sem reinstalar todos os modelos que os produziram.

Artifacts são, portanto, fronteiras de execução e de auditoria.

```mermaid
flowchart LR
    A[Stage N]
    X[Immutable Artifact N]
    B[Stage N+1]
    Y[Immutable Artifact N+1]

    A --> X --> B --> Y
```

## Workspace local

O canonical pipeline usa filesystem local como storage primário. O workspace é organizado por **dataset** e por **run**:

```text
workspace/
├── <dataset>/                       # a sequência física (`inputs.sequence`), por exemplo `corridor-02`
│   └── <run>/                       # um run do runtime: `run-NNNN`
│       ├── effective_config.json    # ┐
│       ├── plan.json                # │ o diário do run, na raiz do run
│       ├── status.json              # │ (ver "Registros de execução do runtime")
│       ├── events.jsonl             # │
│       ├── execution.json           # │
│       ├── run.lock                 # ┘
│       ├── ingestion/               # ┐
│       ├── visual_perception/       # │
│       ├── state_estimation/        # │
│       ├── geometric_mapping/       # │ um diretório por estágio executado,
│       ├── sensor_association/      # │ com o id do estágio da runtime
│       ├── point_representation/    # │ (opcional)
│       ├── semantic_fusion/         # │
│       ├── semantic_mapping/        # │
│       ├── entity_resolution/       # │
│       ├── spatial_relations/       # │
│       └── context_map/             # ┘ o `ContextMapArtifact` (capability `artifact`)
├── experiments/                     # relatórios e manifests de avaliação, fora de qualquer run
└── tmp/                             # conteúdo efêmero
```

Regras do layout:

- **Um diretório por estágio, nomeado pelo id do estágio da runtime.** O diretório é o artifact: `manifest.json`, `outputs/`, `metrics/` e `debug/` ficam direto dentro dele. O estágio final se chama `context_map` mesmo sendo implementado pela capability `artifact`.
- **O diário fica na raiz do run**, ao lado das pastas dos estágios. Os nomes do diário e os ids dos estágios nunca colidem.
- **O run é dono do que executou.** Um estágio reutilizado de um run anterior **não tem pasta** no run novo: ele é **referenciado** por `ArtifactRef` (identidade e hash de conteúdo), nunca copiado. A sequência ingerida pode ter dezenas de GB; copiar seria um erro, e a referência é o que torna a cadeia de reuso auditável.
- **Não há contadores nem registros por capability.** Não existem `runs/<capability>/<sequência>/run-NNNN__...` nem `runs.json`: o único índice de runs é a própria listagem dos diretórios `<dataset>/run-NNNN` (`Runtime.list_runs()`).
- **Um artifact nunca é modificado depois de finalizado**, e um run reexecutado é outro `run-NNNN`.
- `tmp/` nunca é dependência contratual de um artifact válido, e o debug de um artifact nunca é dependência de outro estágio.

Remote storage, S3, MinIO, database ou distributed registry não são requisitos do canonical pipeline.

### Contrato dos writers de estágio

Todo writer de artifact de capability segue o mesmo contrato; o runtime só decide **onde** o estágio grava e o writer nunca calcula um caminho.

**Assinatura.** `output_dir` é o diretório final do artifact, obrigatório e *keyword-only*. Ficam na assinatura o que é identidade ou entrada da capability (`sequence_name`, `run_id`, `run_index`, `debug_level` e os parâmetros próprios, como a linhagem da fusão ou o `artifact_id` da ingestion). Saem `workspace_root` e todo rótulo que só existia para montar o nome do diretório (`selection_label`, `backend_label`, `profile_label`, `channel_label`, `policy_label`):

```python
class StateEstimationRunWriter:
    def __init__(self, *, output_dir: Path, sequence_name: str, run_id: StateEstimationRunId,
                 run_index: int, debug_level: StateEstimationDebugLevel = ...) -> None: ...
```

**Finalização.** O writer cria `output_dir` com `AtomicRunDirectory` (diretório temporário irmão, publicado por rename depois da checagem do inventário), recusa um `output_dir` que já exista e não deixa nada em caso de falha. Ele não cria registro, não escreve `runs.json` e não toca em nenhum outro diretório.

**Identidade.** O writer **nunca aloca** identidade: `run_id` e `run_index` são entregues pelo chamador e gravados como recebidos. `run_index` é um ordinal do chamador (a runtime usa o número de `run-NNNN`) que serve para ordenar candidatos, por exemplo na seleção `latest`; não substitui identidade nem hash. Ids derivados do `run_id`, como `map_id = <sequência>--<run_id>`, continuam iguais, então manifests, leitores e linhagem não mudam. Um executor da runtime deriva `run_id` da identidade do estágio (id do estágio, `config_digest` e hashes de conteúdo das entradas), de modo que execuções idênticas produzem o mesmo id e o mesmo conteúdo, o que mantém o reuso válido entre runs; nos testes, o id é um texto fixo e legível.

**O que se apaga** quando o writer migra: `_sequence_dir`, `allocate_*_run_index`, `rebuild_*_registry`, os helpers que só serviam a eles (`_valid_run_index`, `_registry_record`), a chamada do registro dentro de `finalize()` e os imports de `next_run_index` e `write_run_registry`. Os símbolos saem do `__init__` e do `__all__` da capability. `contextmap.shared.run_directory` **não** muda no PR de uma capability: os helpers de registro só deixam de ter uso quando todas migrarem, e uma limpeza final os remove.

**Leitura.** Os leitores abrem o diretório do artifact diretamente e não mudam: manifests, `manifest.json` e o schema continuam iguais.

**Padrão de teste.** O teste escolhe `output_dir` sob `tmp_path` (por exemplo `tmp_path / "run-0001"`, ou `tmp_path / "state_estimation"`) e abre o leitor **nesse mesmo caminho**, sem procurar `run-0001__...` por glob. Um helper `_run_dir(workspace, index)` centraliza o caminho quando o teste grava vários runs. Testes de alocação de índice e de registro são substituídos por testes de que (1) o artifact aparece exatamente em `output_dir` e nada mais é criado ao redor, (2) `run_id` e `run_index` são gravados como recebidos, (3) um segundo run no mesmo `output_dir` é recusado sem alterar o primeiro.

## Artefatos principais

```mermaid
flowchart TD
    S[SequenceArtifact]
    P[PerceptionRunArtifact]
    T[StateEstimationRunArtifact]
    G[GeometricMapArtifact]
    A[SensorAssociationRunArtifact]
    R[PointRepresentationRunArtifact]
    F[SemanticFusionRunArtifact]
    E[Semantic Entity Artifact]
    ER[EntityResolutionRunArtifact]
    SR[SpatialRelationsRunArtifact]
    C[ContextMapArtifact]

    S --> P
    S --> T
    S --> G
    T --> G
    P --> A
    T --> A
    G --> A
    G --> R
    A --> F
    R -. optional .-> F
    F --> E --> ER --> SR
    G --> C
    ER --> C
    SR --> C
```

## Artefatos materializados hoje

> **Transição de layout.** O layout `<run>/<estágio>/` e o contrato dos writers acima valem para todas as capabilities; cada bloco "atual" abaixo migra junto com o writer da sua capability. Um bloco que ainda mostra `workspace/runs/<capability>/<sequência>/run-000N__...` descreve um writer que **ainda não migrou** para `output_dir`, e um bloco que mostra `<run>/<estágio>/` descreve um que já migrou (`StateEstimationRunArtifact`).

Na `dev`, oito formatos já existem e são integrados:

```mermaid
flowchart LR
    RAW["Fonte registrada"] --> SW["SequenceArtifactWriter"]
    SW --> SEQ["SequenceArtifact"]
    SEQ --> SEL["Selection / replay"]
    SEL --> VP["Visual Perception"]
    VP --> PW["PerceptionRunWriter"]
    PW --> PRA["PerceptionRunArtifact"]
    PRA --> PRR["PerceptionRunReader"]
    PRR --> ES["PerceptionEvidenceSet"]
    SEL --> ST["State Estimation"]
    ST --> SW2["StateEstimationRunWriter"]
    SW2 --> TRA["StateEstimationRunArtifact"]
    TRA --> TRR["StateEstimationRunReader"]
    SEL --> GM["Geometric Mapping"]
    TRA --> GM
    GM --> MW["GeometricMapArtifactWriter"]
    MW --> MAP["GeometricMapArtifact"]
    MAP --> MR["GeometricMapArtifactReader"]
    PRA --> SA["Sensor Association"]
    MAP --> SA
    TRA --> SA
    SA --> AW["SensorAssociationRunWriter"]
    AW --> ASA["SensorAssociationRunArtifact"]
    ASA --> AR["SensorAssociationRunReader"]
    MAP --> PTE["Point Representation"]
    PTE --> PTW["PointRepresentationRunWriter"]
    PTW --> PTA["PointRepresentationRunArtifact"]
    PTA --> PTR["PointRepresentationRunReader"]
    ASA --> SFU["Semantic Fusion"]
    PTA -.-> SFU
    SFU --> SFW["SemanticFusionRunWriter"]
    SFW --> SFA["SemanticFusionRunArtifact"]
    SFA --> SFR["SemanticFusionRunReader"]
    SFA --> SMP["Semantic Mapping"]
    SMP --> SMW["SemanticMappingRunWriter"]
    SMW --> SMA["SemanticMappingRunArtifact"]
    SMA --> SMRD["SemanticMappingRunReader"]
```

`SequenceArtifact` é a sequência canônica concreta produzida por Ingestion. `PerceptionRunArtifact` é o artifact imutável de uma execução de Visual Perception. `StateEstimationRunArtifact` é o artifact imutável de uma execução de State Estimation. `GeometricMapArtifact` é o artifact imutável do mapa que um run construiu. `SensorAssociationRunArtifact` é o artifact imutável das observações espaciais de um run de associação. `PointRepresentationRunArtifact` é o artifact imutável das representações 3D que um encoder produziu sobre um mapa. `SemanticFusionRunArtifact` é o artifact imutável dos suportes de fusão e da evidência fundida. `SemanticMappingRunArtifact` é o artifact imutável das entidades semânticas. Os artifacts downstream do diagrama anterior permanecem planejados.

O mecanismo comum de run (escrita atômica em diretório temporário, inventário com tamanho e SHA-256, índice de run monotônico calculado a partir dos runs válidos e registry reconstruível) é implementado uma vez em `contextmap.shared.run_directory` e usado por `StateEstimationRunArtifact`, `GeometricMapArtifact`, `SensorAssociationRunArtifact`, `PointRepresentationRunArtifact`, `SemanticFusionRunArtifact`, `SemanticMappingRunArtifact` e pelos artifacts das próximas capabilities. Um payload grande é gravado em fluxo (`open_binary`) e hasheado durante a escrita, então um artifact maior que a memória pode ser produzido. Os writers de Ingestion e Visual Perception mantêm suas implementações próprias.

### `SequenceArtifact` atual

```text
workspace/sequences/<sequence-name>/<artifact-id>/
├── manifest.json
├── index.jsonl
├── rgb/
├── pointcloud/
├── calibration/       # opcional
├── provenance/        # opcional
└── diagnostics/       # opcional
```

O índice contém uma observação canônica por linha; payloads grandes de imagem/LiDAR ficam fora do JSONL e são referenciados por path. O manifest inventaria arquivos com tamanho e SHA-256.

### `PerceptionRunArtifact` atual

```text
workspace/runs/visual-perception/<sequence-name>/
├── runs.json
└── run-000N__<selection>__<profile>/
    ├── README.md
    ├── manifest.json
    ├── outputs/
    │   ├── results.jsonl
    │   ├── semantic-interpretations.jsonl
    │   ├── semantic-views/            # bytes exatos fornecidos ao VLM
    │   └── features/                  # opcional em geral; obrigatório se consumido semanticamente
    │       ├── feature-index.jsonl
    │       └── <observation-scope>/*.npy
    ├── metrics/
    │   ├── stage-timings.jsonl
    │   └── feature-extraction.jsonl   # quando há diagnostics de feature
    └── debug/
        ├── 30-feature-extraction/     # somente standard/full
        └── 40-semantic-interpretation/<request-id>/raw-response.txt
```

No schema atual, `manifest.json` também persiste `pipeline_preset` e `configuration_digest`. `runs.json` é somente um registry reconstruível; `PerceptionRunReader` abre um run usando apenas seu próprio diretório.

### `StateEstimationRunArtifact` atual

```text
<run>/state_estimation/            # o output_dir entregue ao writer
├── README.md
├── manifest.json
├── outputs/
│   ├── trajectory.json        # metadados da trajetória
│   ├── poses.jsonl            # uma pose por linha
│   ├── pose-index.jsonl       # leitura de uma pose por identidade ou tempo
│   ├── frame-summary.json
│   └── quality.json
├── metrics/
│   ├── preflight.json
│   ├── motion.json
│   ├── runtime.json           # somente quando medido
│   └── diagnostics.jsonl      # somente quando há eventos
└── debug/                     # somente standard/full; nunca inventariado
```

`manifest.json` traz a linhagem (sequência, seleção, backend e fingerprint de configuração, identidade da calibração, versão do código, frames, clock, contagens) e o inventário dos arquivos contratuais. `debug/` fica fora do inventário, então removê-lo não invalida o run. Um run com preflight de geometria `BLOCKED` nunca é persistido. Detalhes: [State Estimation artifact](../src/contextmap/state_estimation/docs/artifact.md).

### `GeometricMapArtifact` atual

```text
workspace/runs/geometric-mapping/<sequence-name>/
├── runs.json
└── run-000N__<selection>__<profile>/
    ├── README.md
    ├── manifest.json
    ├── lineage.json
    ├── config.json
    ├── environment.json
    ├── outputs/
    │   ├── geometry.bin           # payload empacotado, lido por mapeamento em memória
    │   ├── source-index.jsonl     # um registro por scan: origem → geometria, cadeia e limites
    │   └── map-metadata.json      # GeometricMap: identidade, frame, limites, tempo, proveniência
    ├── metrics/
    │   ├── input-plan.json        # scans selecionados, aceitos e recusados, com o motivo
    │   ├── mapping.json
    │   └── runtime.json           # somente quando medido
    └── debug/                     # somente standard/full; nunca inventariado
```

A identidade do mapa é `<sequência>--<run_id>` e toda `GeometryReference` a carrega; as referências são locais ao artifact. O `manifest.json` inventaria os arquivos contratuais com tamanho e SHA-256, e `debug/` fica fora do inventário. O leitor abre sem ROS, sem FAST-LIO, sem biblioteca de modelo e sem NumPy, e `geometry()` devolve um `GeometrySource` sobre o payload mapeado, sem lê-lo. A configuração é JSON (`config.json`), não YAML. Detalhes: [Geometric Mapping artifact](../src/contextmap/geometric_mapping/docs/artifact.md).

### `SensorAssociationRunArtifact` atual

```text
workspace/runs/sensor-association/<sequence-name>/
├── runs.json
└── run-000N__<selection>__<canal>/
    ├── README.md
    ├── manifest.json
    ├── outputs/
    │   ├── spatial-observations.jsonl     # um SpatialObservation por linha
    │   ├── observation-index.jsonl        # id, frame, região e offsets
    │   ├── geometry-support.u32           # região → geometria, tabela colunar de uint32
    │   ├── observation-quality.jsonl      # ObservationQuality por observação
    │   ├── projection-records.jsonl       # por frame: câmera, pose, extrínseco, cadeia de imagem
    │   ├── visibility-records.jsonl       # por frame: política, contagens, pertencimento
    │   ├── dense-feature-associations.jsonl
    │   └── dense-feature-cells.bin        # índices e pesos, sem vetores de feature
    ├── metrics/
    │   ├── frame-diagnostics.jsonl
    │   ├── summary.json
    │   └── runtime.json                   # somente quando medido
    └── debug/                             # somente standard/full; nunca inventariado
```

O run não repete XYZ nem vetores de embedding: a geometria é referenciada por posição (a identidade do mapa é posicional) e a amostragem densa guarda índices e pesos. `manifest.json` carrega a linhagem exata (sequência, seleção, mapa geométrico, trajetória, calibração, runs de percepção, políticas, fingerprint, código, canais de features com as fontes exatas) e o inventário; `debug/` fica fora dele. Uma execução nativa e uma melhorada de features compartilham os mesmos artifacts a montante e continuam identificáveis de forma independente. O leitor abre sem ROS, sem modelos e sem NumPy. Detalhes: [Sensor Association artifact](../src/contextmap/sensor_association/docs/artifact.md).

### `PointRepresentationRunArtifact` atual

```text
workspace/runs/point-representation/<sequence-name>/
├── runs.json
└── run-000N__<selection>__<backend>/
    ├── README.md
    ├── manifest.json                       # identidade, linhagem e inventário
    ├── outputs/
    │   ├── representations.jsonl           # PointRepresentation canônica, uma por linha
    │   ├── representation-index.jsonl      # representation_id → deslocamento e tamanho
    │   ├── representation-spaces.json      # RepresentationSpace do run + fingerprint
    │   ├── geometry-representation-index.jsonl
    │   ├── failed-supports.jsonl           # suportes que não produziram representação
    │   └── payloads/vectors.f32|f64        # vetores, uma linha de tamanho fixo cada
    ├── metrics/
    │   └── counts.json  support-size.json  norms.json  runtime.json
    └── debug/                              # somente standard/full; nunca inventariado
```

O `manifest.json` traz a linhagem (mapa geométrico consumido, seleção dos centros independente da ordem, política de suporte, espaço, encoder e hash do checkpoint, código, contexto de associação **opcional e explícito**) e o inventário. Um suporte que falhou vai para `failed-supports.jsonl` com o motivo; um run só de falhas é um run válido e explícito, e um vetor nunca é inventado. O run abre sem NumPy, sem biblioteca de modelo e sem o mapa geométrico, e os vetores são lidos sob demanda. O escritor ainda acumula em memória; a escrita em fluxo de `shared.run_directory` ainda não foi adotada por ele. Detalhes: [Point Representation artifact](../src/contextmap/point_representation/docs/artifact.md).

### `SemanticFusionRunArtifact` atual

```text
workspace/runs/semantic-fusion/<sequence-name>/
├── runs.json
└── run-000N__<selection>__<policy>/
    ├── README.md
    ├── manifest.json                          # identidade, linhagem explícita, políticas e inventário
    ├── outputs/
    │   ├── fusion-supports.jsonl              # um FusionSupport por linha
    │   ├── fused-evidence.jsonl               # um FusedEvidence por linha (autocontido)
    │   ├── support-observation-index.jsonl    # suporte → observações, offsets nos dois arquivos
    │   ├── hypothesis-evidence-index.jsonl    # hipótese → evidência exata
    │   ├── physical-observation-groups.jsonl  # grupos por frame físico de cada suporte
    │   ├── contribution-index.jsonl           # contribuição → suporte, observação, resultado, run
    │   └── excluded-observations.jsonl        # evidência pulada, com o motivo
    ├── metrics/
    │   ├── counts.json  distributions.json  payload.json
    │   └── runtime.json                       # somente quando medido
    └── debug/                                 # somente standard/full; nunca inventariado
```

O artifact guarda **todas** as hipóteses, com alternativas, conflitos, abstenções e evidência não pontuada (`None`, nunca zero), e mantém frames físicos e resultados de inferência distintos. Nada a montante é duplicado: claims, scores, features, qualidade e estrutura 3D são referenciados, e a geometria é guardada como deltas posicionais. `manifest.json` traz a linhagem **explícita** (sequência, mapa, runs de associação, percepção e Point Representation), as políticas com fingerprint e as identidades que alimentaram cada canal. Um run é escrito em fluxo e publicado de forma atômica, e o leitor abre sem NumPy, sem runtime de percepção e sem biblioteca de modelo, lendo um suporte sem carregar os outros. Semantic Mapping não pode depender de `debug/`. Detalhes: [Semantic Fusion artifact](../src/contextmap/semantic_fusion/docs/artifact.md).

### `SemanticMappingRunArtifact` atual

```text
workspace/runs/semantic-mapping/<sequence-name>/
├── runs.json
└── run-000N__<selection>__<policy>/
    ├── README.md
    ├── manifest.json                          # identidade, linhagem do run de fusão, política e inventário
    ├── outputs/
    │   ├── entities.jsonl                     # uma Entity canônica por linha (autocontida, autoritativa)
    │   ├── entity-index.jsonl                 # entidade → deslocamento e tamanho
    │   ├── entity-geometry-index.jsonl        # entidade → mapa, frame, pontos, limites, centroide, diagnósticos
    │   ├── entity-evidence-index.jsonl        # entidade → evidência fundida e contagens
    │   ├── entity-observation-index.jsonl     # uma linha por (entidade, frame físico)
    │   ├── entity-semantic-state.jsonl        # entidade → ambiguidade, primária, labels, atributos, incerteza
    │   ├── entity-temporal-state.jsonl        # entidade → first/last seen, contagens, ciclo de vida
    │   └── rejected-candidates.jsonl          # candidatos que não viraram entidade, com o motivo
    ├── metrics/
    │   ├── counts.json  distributions.json  payload.json
    │   └── runtime.json                       # somente quando medido
    └── debug/                                 # somente standard/full; nunca inventariado
```

O artifact guarda **todas** as hipóteses, conflitos, abstenções e sinais sem score de cada entidade, mantém frames físicos e resultados de inferência distintos e **não** contém estado de merge, split ou resolução. Nada a montante é duplicado: a evidência é referenciada com a identidade, a versão do schema e o digest do inventário do artifact de fusão, e a geometria é guardada como deltas posicionais. `manifest.json` traz a linhagem **explícita** (o run de fusão selecionado, o mapa geométrico e, pela linhagem da fusão, as runs de associação, percepção e Point Representation), a política de materialização e o fingerprint da configuração; um run guarda uma política. O leitor abre sem NumPy, sem runtime de percepção, de fusão ou de modelo, resolve uma `EntityReference` sem carregar as demais e recusa uma referência de outro semantic map. Entity Resolution e Spatial Relations não podem depender de `debug/`. Detalhes: [Semantic Mapping artifact](../src/contextmap/semantic_mapping/docs/artifact.md).

### Evidência auditável de Region Discovery

Region Discovery possui um writer de evidência de estágio próprio para experimentação, inspeção e avaliação. Ele não cria uma nova identidade de percepção paralela ao `PerceptionRunArtifact`; registra os intermediários e métricas necessários para explicar como `Region2D[]` foi produzido.

```mermaid
flowchart LR
    PI["PreparedImage"] --> RD["Region Discovery"]
    RD --> REG["Region2D[]"]
    REG --> PRA["PerceptionResult / PerceptionRunArtifact"]
    RD --> W["RegionDiscoveryEvidenceWriter"]
    W --> O["outputs/<br/>regions.jsonl + metrics.json"]
    W --> M["manifest.json + hashes"]
    W -. standard/full .-> D["debug/<br/>candidates, passes, overlays, masks"]
```

O diretório de estágio é finalizado atomicamente. `outputs/` e `manifest.json` são contratuais para esse evidence artifact; `debug/` continua não contratual e pode ser descartado sem alterar a semântica de `Region2D`. O layout e os níveis `none|standard|full` estão documentados em [Region Discovery](../src/contextmap/visual_perception/docs/region-discovery.md).

### Feature Extraction dentro do `PerceptionRunArtifact`

Feature Extraction não cria um segundo run artifact. Metadata de `VisualFeature` permanece em `outputs/results.jsonl`; payloads numéricos opcionais ficam em `outputs/features/`, indexados por `feature-index.jsonl` e carregados sob demanda.

`PerceptionRunWriter.finalize()` cruza cada payload com a feature da mesma observação, valida scope, embedding space, shape, dtype, normalização e referência, inclui todos os arquivos no `file_inventory` e publica o run somente após a checagem de integridade.

Diagnostics mínimos ficam em `metrics/feature-extraction.jsonl`. Previews e metadata auxiliares de inspeção ficam em `debug/30-feature-extraction/` somente quando o nível selecionado é `standard` ou `full`. Remover debug não pode afetar a leitura dos outputs contratuais.

Detalhes: [Feature Extraction](../src/contextmap/visual_perception/docs/feature-extraction.md), [feature store](../src/contextmap/visual_perception/docs/feature_store.md) e [diagnostics](../src/contextmap/visual_perception/docs/feature_diagnostics.md).

### Semantic Interpretation dentro do `PerceptionRunArtifact`

Semantic Interpretation também não cria um artifact paralelo. O
`PerceptionResult` continua sendo a evidência canônica consumida downstream,
enquanto `outputs/semantic-interpretations.jsonl` preserva o limite exato de
cada chamada: request, prompt renderizado, resposta bruta distinguível do
parsing, diagnostics e configuração efetiva.

Cada view selecionada é materializada abaixo de `outputs/semantic-views/` com
SHA-256 obrigatório e entra no `file_inventory`. Se o request consumir uma
`VisualFeature`, o payload numérico correspondente precisa estar no feature
store. `region_id`, `scene_context_reference` e outputs parseados precisam
resolver para evidência do run antes de `finalize()`.

A resposta bruta também é materializada no path de debug declarado pela
provenance. Ela é importante para auditoria, mas o parsing canônico não depende
do arquivo de debug para existir. O schema atual do run artifact é `0.4.0`;
artifacts `0.3.0` são rejeitados na abertura.

Detalhes: [Semantic Interpretation](../src/contextmap/visual_perception/docs/semantic-interpretation.md) e [run artifact](../src/contextmap/visual_perception/docs/run_artifact.md).

Detalhes específicos permanecem nos owners:

- [Ingestion artifact](../src/contextmap/ingestion/docs/artifact.md);
- [Visual Perception run artifact](../src/contextmap/visual_perception/docs/run_artifact.md).

### Artefatos de avaliação

A capability `evaluation` persiste documentos JSON imutáveis, escritos por publicação atômica (recusam sobrescrever) e com digest `sha256` verificado na leitura. Não são artifacts de run do pipeline e não alteram nenhum artifact avaliado.

```text
<reference-set>/<versão>/
├── manifest.json            # ReferenceSetManifest (contextmap.reference-set/v1), com digest
└── annotations/<família>.json  # contextmap.reference.<família>/v1, hash declarado no manifesto

<experimento>/<run>/         # diretório novo por execução
├── experiment.json          # ExperimentManifest (contextmap.experiment/v1)
├── arms/<arm>/run.json      # topologia resolvida, artifacts por estágio, resultado (…-arm-run/v1)
├── arms/<arm>/report.json   # EvaluationReport (contextmap.evaluation-report/v1), só arms concluídos
└── comparison.json          # ComparisonManifest (…-comparison/v1): métricas lado a lado, artifacts compartilhados

evidence.json                # TechniqueEvidence por métrica e estrato (contextmap.technique-evidence/v1)
decision.json                # TechniqueDecision, presa ao digest da evidência (…-decision/v1)
```

- `ReferenceSetManifest` amarra fontes, amostras (por `SourceObservationId`, nunca por saída de percepção), calibrações, anotações com `trust` e proveniência declarados, estratos e splits; a versão muda quando o conteúdo muda.
- Um arm indisponível ou com resultado inconsistente é registrado como tal e não tem `report.json`; a comparação fica `complete: false`.
- Reexecutar um experimento cria outro diretório de run; nunca sobrescreve.
- Há um subconjunto sintético de CI versionado em `tests/fixtures/ci_subset/<versão>/` (manifesto, anotações e catálogo). Não existe reference set real versionado nem execução real de experimento registrada.

Detalhes: [reference set](../src/contextmap/evaluation/docs/reference-set.md), [experimentos](../src/contextmap/evaluation/docs/experiments.md) e [técnicas opcionais](../src/contextmap/evaluation/docs/optional-techniques.md).

### Registros de execução do runtime

Uma execução do runtime deixa um registro em `<workspace>/<dataset>/run-NNNN/`, o diretório do run, e cada estágio que ela executa grava o seu artifact numa pasta ao lado do diário (`<run>/<estágio>/`). Ele não é um artifact de capability (tem formato próprio e não passa por `contextmap.shared.run_directory`), mas registra a linhagem exata da execução. O número é alocado de forma atômica, então execuções concorrentes nunca compartilham um diretório.

```text
<dataset>/run-0001/
├── effective_config.json   # configuração efetiva, digest e camadas; sem segredos
├── plan.json               # topologia resolvida, com digest (só sem problema estrutural)
├── status.json             # estado atual, reescrito atomicamente
├── events.jsonl            # eventos append-only, numerados sem lacuna
├── execution.json          # entradas e saídas exatas por estágio (só run concluído)
├── run.lock                # pid do processo dono, só enquanto o run está vivo
└── <estágio>/              # o artifact de cada estágio executado neste run (não copiado quando reutilizado)
```

`status.json` e `events.jsonl` mudam enquanto o run executa; depois de um estado terminal nada é reescrito, e retomar um run cria um run **novo**. Um processo que morre deixa um registro consistente e inspecionável (`interrupted`). O registro é a autoridade sobre a linhagem de uma execução: os estágios são referenciados por `ArtifactRef` (id e hash) e a inspeção nunca infere o que o registro não contém.

Além do diretório do run, o runtime usa dois insumos explícitos: o **índice de reuso** (um JSON imutável por identidade de estágio, escrito só depois que o estágio conclui) e o **catálogo de runs** para seleção (`{"schema_version": "0.1.0", "entries": [...]}`). Nenhum dos dois é descoberto por varredura de diretório. Detalhes em [lifecycle.md](../src/contextmap/runtime/docs/lifecycle.md), [reuse.md](../src/contextmap/runtime/docs/reuse.md) e [selection.md](../src/contextmap/runtime/docs/selection.md).

## Immutability

Depois da finalização bem-sucedida de um artifact:

- arquivos contratuais não são alterados;
- IDs não são reciclados;
- evidence não é apagada porque uma interpretação posterior mudou;
- reexecução cria outro artifact/run;
- cache/reuse referencia um artifact anterior, nunca o modifica.

Exemplo:

```text
run-0003  # immutable
run-0004  # nova execução com configuração diferente
```

## Identidade de run

A identidade de um artifact de estágio é entregue pelo chamador (`run_id`), nunca alocada pelo writer ([Contrato dos writers de estágio](#contrato-dos-writers-de-estágio)). Cada execução da runtime é um `run-NNNN` próprio dentro do dataset, e cada estágio executado vive em `<run>/<estágio>/`:

```text
workspace/corridor-02/
├── run-0001/state_estimation/    # ExternalPose, versão A
├── run-0002/state_estimation/    # ExternalPose, versão B da configuração
└── run-0003/                     # só o diário: tudo reutilizado por ArtifactRef
```

O `run_index` gravado no manifest é um ordinal legível fornecido pelo chamador, mas não substitui artifact identity, hashes e manifest.

Timestamp de criação pertence ao manifest. Ele não precisa ser o identificador principal do diretório.

## Estrutura comum de run

Cada capability pode ter payloads diferentes, mas a organização conceitual deve permanecer previsível:

```text
run-000N__<selection>__<profile>/
├── README.md
├── manifest.json
├── config.yaml
├── lineage.json
├── environment.json
├── events.jsonl
├── outputs/
├── metrics/
└── debug/
```

Nem todo artifact precisa de todos os arquivos físicos acima, mas os conceitos devem estar representados quando relevantes.

Essa árvore é uma **estrutura conceitual**, não uma obrigação física. Os artifacts implementados hoje são deliberadamente menores: Ingestion concentra metadata em `manifest.json` e arquivos opcionais próprios; Visual Perception v0 não escreve `config.yaml`, `lineage.json`, `environment.json` ou `events.jsonl` separados porque ainda não existem produtores reais para esses arquivos. Criá-los vazios violaria YAGNI.

## `manifest.json`

É o ponto autoritativo de metadata do artifact.

Deve permitir responder:

```text
quem sou eu?
qual schema/version uso?
quais inputs consumi?
qual selection processei?
qual código/configuração/backend produziu o resultado?
quais outputs existem?
quais hashes garantem integridade?
quais warnings/errors ocorreram?
```

Metadata mínima conceitual:

```text
artifact_id
artifact_type
schema_version
run_index, when applicable
created_at
repository / commit SHA
effective configuration reference/hash
selected upstream artifacts
source sequence/selection
backend/model/policy identities
output inventory
content hashes
warnings/errors summary
```

## `config.yaml`

Registra a configuração efetivamente resolvida para o estágio.

Não deve conter secrets.

Exemplos que precisam ser persistidos como valores efetivos quando relevantes:

```text
backend selection
model/checkpoint
thresholds
prompt/template version
sampling policy
fusion policy
relation thresholds
debug level
```

API keys, bearer tokens e credentials nunca devem ser persistidos.

## `lineage.json`

Explicita de quais artifacts e observações o artifact atual depende.

Lineage não é uma descrição textual vaga. Ele deve apontar para identities/hashes suficientes para validar as dependências.

Exemplo conceitual:

```text
SensorAssociationRunArtifact
├── SequenceArtifact
├── PerceptionRunArtifact
├── StateEstimationRunArtifact
├── GeometricMapArtifact
└── Calibration identity
```

## `environment.json`

Quando necessário para reprodução, pode registrar metadata como:

```text
Python/runtime version
platform
relevant package versions
GPU/device identity
precision mode
backend runtime version
```

Evitar despejar informações sem utilidade para a reprodução.

## `events.jsonl`

Registro estruturado de eventos de execução quando necessário:

```text
stage started
stage completed
warning
recoverable failure
retry
artifact finalized
```

Logs de console não substituem outputs contratuais.

## `outputs/`

Contém o resultado contratual consumido downstream.

Exemplos:

```text
PerceptionRunArtifact (schema 0.4.0)
outputs/
├── results.jsonl
├── semantic-interpretations.jsonl   # quando houve execução semântica
├── semantic-views/                  # views content-addressed consumidas
└── features/                        # opcional em geral
    ├── feature-index.jsonl
    └── <observation-scope>/*.npy
```

Cada linha contém um `PerceptionResult` completo com `regions`, `features`, `claims` e `scene_context`. Índices separados podem ser adicionados apenas quando houver um caso de uso medido que justifique a duplicação.

```text
SensorAssociationRunArtifact
outputs/
├── spatial-observations.*
├── projection-records.*
├── visibility-records.*
└── geometry-region-index.*
```

```text
SemanticFusionRunArtifact
outputs/
├── fusion-supports.*
├── fused-evidence.*
├── physical-observation-groups.*
└── contribution-index.*
```

Um módulo downstream pode depender de `outputs/`, nunca de `debug/`.

## `metrics/`

Métricas de execução e qualidade ficam separadas dos outputs de domínio.

Exemplos:

```text
latency
memory
counts
reprojection error
ambiguity rate
artifact size
```

Quality e performance devem permanecer semanticamente separadas. Um backend mais rápido não se torna automaticamente melhor semanticamente.

## `debug/`

Contém evidência auxiliar para investigação humana.

Exemplos:

- RGB overlays;
- masks;
- crops;
- projection visualizations;
- trajectory plots;
- selected transform traces;
- prompt snapshots;
- raw model responses;
- contribution traces;
- relation measurement summaries.

### Regra

Excluir `debug/` não pode tornar `outputs/` ilegíveis ou inutilizáveis.

## Debug levels

A política global usa níveis conceituais:

```text
none
    outputs contratuais + lineage + metadata + métricas requeridas

standard
    principais visualizações e diagnósticos

full
    evidência intermediária detalhada
```

O conteúdo exato de cada nível pertence à capability.

## Artifact vs debug

A distinção precisa ser explícita:

```mermaid
flowchart TD
    R[Run]
    O[Contractual Outputs]
    D[Debug Evidence]
    C[Downstream Consumer]
    H[Human Inspection]

    R --> O --> C
    R --> D --> H
    O --> H
    D -. forbidden dependency .-> C
```

## Payloads grandes

Features densas, point representations e outras matrizes grandes não precisam ficar inline em JSON/Parquet principal.

Preferir metadata leve + `payload_reference`.

Exemplo:

```text
VisualFeature metadata
├── feature_id
├── shape
├── dtype
├── embedding_space_id
├── normalization
├── payload path/reference
└── content hash no índice do payload
```

O payload pode ser lazy-loaded sem carregar todo o artifact.

## Integridade

Um artifact válido precisa detectar, conforme seu schema:

- arquivos obrigatórios ausentes;
- payload referenciado ausente;
- content hash incorreto;
- schema incompatível/desconhecido;
- cross-reference inválida;
- identity de upstream incompatível;
- artifact parcialmente criado.

Uma escrita interrompida não pode parecer um artifact finalizado válido.

## Finalização atômica

A implementação deve preferir um processo conceitual como:

```mermaid
flowchart LR
    TMP[Temporary/incomplete run]
    WRITE[Write outputs + indexes]
    VERIFY[Integrity verification]
    FINAL[Finalize immutable artifact]

    TMP --> WRITE --> VERIFY --> FINAL
```

Somente depois da verificação o run entra no conjunto de artifacts válidos.

## Reprodutibilidade

Um artifact deve registrar metadata suficiente para reproduzir suas decisões principais.

A reprodução exata pode depender de fatores não determinísticos de modelos, mas o sistema precisa preservar ao menos:

```text
source content identity
selection
backend/model/checkpoint identity
policy versions
prompt/template version
configuration digest
code commit
schema versions
upstream artifact identities
```

## Seleção explícita de runs

A existência de vários runs não significa que todos devem ser combinados.

Exemplo:

```text
run-0001  perception baseline
run-0002  different VLM
run-0003  known bad experiment
```

Um Semantic Fusion run deve receber uma lista/seleção explícita de artifacts.

Não existe comportamento implícito:

```text
use every run found in the directory
```

A menos que uma policy futura defina isso explicitamente, versione a decisão e registre a resolução.

## Overlapping selections

Runs podem processar selections diferentes e parcialmente sobrepostas.

```text
run-0001 -> frames 0000..1000
run-0005 -> frames 0430..0480
run-0007 -> frames 0430..0480
```

A leitura multi-run deve preservar que os results em `frame-0450` pertencem à mesma observação física, mesmo vindo de três inferências.

Nenhum payload precisa ser copiado para construir essa view.

## Registry local

Não há registro por capability: o discovery é a listagem dos diretórios `<dataset>/run-NNNN` (`Runtime.list_runs()`), e o índice de reuso e o catálogo de seleção da runtime são insumos explícitos e reconstruíveis.

O artifact individual é self-describing através do próprio manifest.

Consequências:

- um run pode ser aberto sem registry global;
- registry ausente não invalida artifacts corretos;
- registry stale pode ser reconstruído a partir dos manifests;
- diretórios incompletos não entram como runs válidos.

## Reuso e cache identity

Reuso de um artifact exige compatibilidade real, não apenas filename semelhante.

A identity de cache/reuse pode considerar, conforme a capability:

```text
upstream artifact IDs/hashes
selection
schema versions
backend/model/checkpoint
policy versions
relevant effective configuration
code/policy identity
```

### Exemplo

Se somente o prompt da percepção muda:

```text
SequenceArtifact      reusable
StateEstimationRunArtifact    reusable
GeometricMapArtifact          reusable
PerceptionRunArtifact         recompute
SensorAssociationRunArtifact  recompute
SemanticFusion+               recompute
```

Se a trajetória muda:

```text
PerceptionRunArtifact         potentially reusable
StateEstimationRunArtifact    recompute
GeometricMapArtifact          recompute
SensorAssociationRunArtifact  recompute
all geometry-dependent stages recompute
```

O DAG determina downstream invalidation.

## Provenance closure

O mapa final deve indicar quais dependencies são necessárias para entender sua estrutura e quais são opcionais para inspeção profunda.

Distinguir:

```text
required structural dependency
    necessária para resolver o mapa/entidade/relação

optional evidence dependency
    necessária apenas para inspeção detalhada da evidência

debug dependency
    nunca contratual
```

## Lineage de entidade

Esta seção e as seções de lineage de relação e `ContextMapArtifact` abaixo
descrevem artifacts **planejados**, exceto pelo trecho de entidade
(`Source Entity → FusedEvidence → SpatialObservation → PerceptionResult →
SourceObservation`), que o `SemanticMappingRunArtifact` já persiste e que
`trace_entity_evidence` percorre por identidade. As capabilities
`entity_resolution`, `spatial_relations` e `artifact` ainda não existem.

```mermaid
flowchart RL
    RE[ResolvedEntity]
    D[ResolutionDecision]
    E[Source Entity]
    F[FusedEvidence]
    S[SpatialObservation]
    P[PerceptionResult]
    O[SourceObservation]

    RE --> D
    RE --> E --> F --> S --> P --> O
```

## Lineage geométrico

```mermaid
flowchart RL
    E[EntityGeometry]
    R[GeometryReference]
    G[GeometricMapArtifact]
    P[PoseEstimate]
    C[Calibration]
    O[Source LiDAR Observation]

    E --> R --> G
    G --> P
    G --> C
    G --> O
```

## Lineage de relação

```mermaid
flowchart RL
    R[Relation]
    EV[RelationEvidence]
    S[Resolved Subject]
    O[Resolved Object]
    G[Entity Geometry]

    R --> EV
    R --> S --> G
    R --> O --> G
```

## ContextMapArtifact

O artifact final deve compor ou referenciar:

```text
ContextMap
├── metadata
├── GeometricMap identity/reference
├── resolved entities
├── spatial relations
├── indexes
└── lineage/provenance closure
```

Ele deve permanecer legível sem model runtimes.

Um consumidor que só precisa de entidades/relações não deve precisar baixar raw bags, checkpoints ou debug artifacts.

## Política de não duplicação

Evitar copiar grandes payloads entre artifacts apenas por conveniência.

Preferir references estáveis:

```text
Entity
→ GeometryReference
```

em vez de:

```text
Entity
→ duplicated XYZ array
```

Preferir:

```text
SpatialObservation
→ SemanticClaim reference
```

em vez de copiar a claim para cada geometry point.

## Warnings e failures

Failures não devem desaparecer do lineage.

Artifacts devem preservar quando aplicável:

```text
warnings
recoverable failures
skipped observations
unsupported evidence
retries
timeouts
invalid inputs
partial stage state
```

O consumidor precisa distinguir output completo, parcial e bloqueado.

## Regras de segurança

Artifacts nunca devem persistir:

- API keys;
- bearer tokens;
- credentials;
- secrets de providers.

Provider/model identity, request metadata não sensível, latency, retry count e usage podem ser preservados quando necessários para auditoria.

## Invariantes

1. Artifacts finalizados são imutáveis.
2. Reexecução gera nova identity.
3. `outputs/` é contratual, `debug/` não é.
4. Um run pode ser aberto sem registry global.
5. Runs upstream são selecionados explicitamente.
6. Identity depende de conteúdo/configuração relevante, não apenas path.
7. Reuso nunca altera o artifact reutilizado.
8. Lineage deve chegar até observações físicas e artifacts upstream.
9. Payloads grandes podem ser lazy, mas precisam de metadata/hash.
10. Secrets nunca fazem parte de artifact provenance.
