# Documentação do ContextMap2

Este diretório é o ponto de entrada da documentação global do ContextMap2.

O objetivo da documentação principal é permitir que uma pessoa entenda **o que o projeto produz, como o pipeline funciona, quais contratos conectam os módulos e como os resultados permanecem reproduzíveis** sem precisar reconstruir essas decisões a partir das issues.

> A documentação descreve a arquitetura alvo do **canonical pipeline**. Uma capability documentada pode ainda estar planejada ou em implementação. O fato de uma etapa aparecer no pipeline não significa, por si só, que ela já esteja concluída no código.

## O que é o ContextMap2

O ContextMap2 transforma observações robóticas registradas, como RGB, LiDAR, IMU, pose e calibração, em um mapa contextual 3D portátil e versionado.

O produto final do repositório é um `ContextMapArtifact` que conecta:

- geometria 3D persistente;
- entidades semânticas resolvidas;
- relações espaciais entre entidades;
- evidências e provenance suficientes para explicar de onde cada resultado veio.

```mermaid
flowchart LR
    A[Dados registrados] --> B[Sequência canônica]
    B --> C[Percepção + estado + geometria]
    C --> D[Associação 2D ↔ 3D]
    D --> E[Fusão semântica]
    E --> F[Entidades persistentes]
    F --> G[Relações espaciais]
    G --> H[ContextMapArtifact]
```

Visualização, busca em linguagem natural, navegação, planejamento, agentes e dashboards são **consumidores externos**. Eles não pertencem ao núcleo deste repositório.

## Estado materializado na `dev`

A documentação global descreve o canonical pipeline completo, mas o código atualmente materializado deve ser lido de forma separada do alvo futuro. Hoje, os oito primeiros boundaries de domínio estão implementados e integrados por contratos públicos; `evaluation` também existe como capability transversal:

```mermaid
flowchart LR
    RAW["ROS 1 / ROS 2 / fonte registrada"] --> ING["Ingestion<br/>implementado"]
    ING --> SEQ["SequenceArtifact<br/>sequência canônica imutável"]
    SEQ --> VP["Visual Perception Core<br/>implementado"]
    VP --> PR["PerceptionRunArtifact"]
    SEQ --> ST["State Estimation<br/>implementado"]
    ST --> TR["StateEstimationRunArtifact"]
    SEQ --> GM["Geometric Mapping<br/>implementado"]
    TR --> GM
    GM --> MAP["GeometricMapArtifact"]
    PR --> SA["Sensor Association<br/>implementado"]
    MAP --> SA
    TR --> SA
    SA --> AR["SensorAssociationRunArtifact"]
    MAP --> PTR["Point Representation<br/>implementado, opcional"]
    PTR --> PTA["PointRepresentationRunArtifact"]
    AR --> FUS["Semantic Fusion<br/>implementado"]
    PTA -.-> FUS
    FUS --> FUA["SemanticFusionRunArtifact"]
    FUA --> SM["Semantic Mapping<br/>implementado"]
    SM --> SMA["SemanticMappingRunArtifact"]
    SMA --> ER["Entity Resolution<br/>implementado"]
    ER --> ERA["EntityResolutionRunArtifact"]
    ERA -. próximo boundary .-> NEXT["Spatial Relations<br/>e downstream planejados"]
```

Ingestion possui adapters ROS 1/ROS 2, observações canônicas, calibração, sincronização, seleção/replay, provenance, validação e `SequenceArtifact`. Visual Perception possui o core de execução, Region Discovery concreto, Feature Extraction com adapters DINOv2, DINOv3, CLIP e AlphaCLIP, além de compatibilidade de embeddings, payload store, sampling denso, pooling por região, diagnostics e avaliação, e o boundary canônico de Semantic Interpretation. Esse boundary inclui `SemanticClaim`/`SceneContext`, requests auditáveis, prompts/parsing versionados, `SemanticInterpretationExecution`, persistência das views exatas e adapters Qwen/Gemini/Florence-2 isolados por seams testáveis. `SemanticScorer` também possui adapters CLIP/AlphaCLIP separados das claims. `PerceptionRunArtifact` e leitura multi-run continuam preservando evidência sem fusão implícita. Em Feature Extraction, diagnostics `SUCCEEDED`/`WARNING` são vinculados no `finalize()` à `VisualFeature` persistida, inclusive a geometria contratual dos mapas densos; um artifact não pode ser publicado com shape/grid/provenance/payload incompatíveis. A CI usa runtimes/clients injetados; SAM2/SAM3, DINOv2/DINOv3/CLIP, Qwen e Florence-2 semântico possuem execuções reais registradas com escopos distintos (Qwen e Florence-2 semântico com runtimes transformers reais sobre os 20 frames de corridor-02, sem anotações humanas). AlphaCLIP e Gemini continuam sem execução real registrada: o Gemini possui cliente `google-genai` validado só com transporte simulado, sem credencial nem consentimento para enviar frames. Não há avaliação científica comparativa dos adapters semânticos, porque a amostra não tem anotações e a correção é N/A.

State Estimation possui os contratos `PoseEstimate`/`Trajectory`, lookup temporal com interpolação auditável, frame graph estático e preflight de geometria, o port `StateEstimator` com os backends `ExternalPose` e FAST-LIO, o `StateEstimationRunArtifact` e o harness de avaliação em `evaluation`, com o perfil de referência do `corridor-02`. O FAST-LIO roda dentro de um container (wrapper de implantação e receita da imagem no repositório) e foi executado de verdade sobre a janela de 90 s de referência: 888 poses, repetíveis byte a byte, ATE de 0,092 m contra o arquivo de referência com alinhamento SE(3) explícito. O arquivo de referência não tem acurácia caracterizada e provavelmente é derivado de LiDAR, então o resultado é concordância com ele, não erro absoluto. Uma sequência e uma janela; a covariância do FAST-LIO não é exportada.

Geometric Mapping possui os contratos `GeometryPoint`/`GeometryReference`/`GeometricMap`, o estado explícito de correção de movimento, a montagem dos inputs, a transformação fonte→mapa com traces auditáveis, a acumulação com referências estáveis, o acesso espacial por `GeometrySource` com índice derivado e verificável, o `GeometricMapArtifact` e o harness de validação em `evaluation`. A validação real usou a trajetória do dataset como entrada, não uma execução FAST-LIO, e não há geometria de referência confiável para o `corridor-02`.

Sensor Association possui os contratos `SpatialObservation` e `ObservationQuality`, os modelos de câmera calibrados (pinhole, fisheye e MEI, com o MEI adicionado à calibração canônica de Ingestion), a cadeia mapa → câmera → imagem preparada, a resolução de visibilidade e oclusão, o pertencimento à máscara de `Region2D`, a amostragem de features densas (nativas ou melhoradas, como canais distintos), os diagnósticos de calibração, reprojeção e alinhamento temporal, o `SensorAssociationRunArtifact` e o harness de avaliação estratificada em `evaluation`. A verificação usa fixtures sintéticos determinísticos e uma primeira execução real sobre 20 frames do `corridor-02`: a trajetória de referência do dataset como entrada `ExternalPose` (não FAST-LIO), o mapa LiDAR real construído sobre ela e uma calibração MEI derivada do YAML de intrínsecos do dataset (o `SequenceArtifact` persistido não guarda o modelo MEI). A cadeia canônica sobre um mapa construído com a trajetória real do FAST-LIO (que já existe) continua pendente e não há correspondências de referência reais, então a correção da associação não foi medida.

Point Representation é uma capability **opcional**: possui os contratos `PointRepresentation`/`RepresentationSpace`, a extração de suporte local sobre `GeometrySource`, o port `PointEncoder` e o serviço de execução independente de backend, o descritor geométrico determinístico (baseline), a fronteira do backend PTv3 com seu runtime real sobre o Pointcept, o `PointRepresentationRunArtifact` e o harness de ablação em `evaluation`. Ela deve justificar seu custo por avaliação controlada, e essa justificativa **não existe ainda**: o PTv3 e o descritor foram avaliados sobre geometria real do corredor-02 (sem vantagem demonstrada do PTv3 e a um custo cerca de 20 vezes maior), mas a ablação downstream sobre Semantic Fusion e Entity Resolution ainda não foi realizada, pois falta Entity Resolution integrada e anotações de identidade.

Semantic Fusion acumula a evidência multi-vista **sem criar identidade de objeto**: possui `FusionSupport`, `EvidenceContribution`, o agrupamento por observação física, a política baseline de acumulação (hipóteses por chave de label, stances e sinais tipados, abstenção configurável e registros de incerteza), os canais de evidência tipados, a política opcional ciente de qualidade, o `SemanticFusionRunArtifact` e o harness de validação em `evaluation`. A verificação usa fixtures sintéticos e uma primeira execução real sobre 20 frames do `corridor-02` (cadeia real com trajetória `ExternalPose`, mapa LiDAR, associação MEI e percepção existente). Ela **não tem claims nem anotações de referência**, então não há hipótese fundida, a correção é N/A e **nenhuma decisão foi tomada** sobre manter a política ciente de qualidade opcional ou adotá-la.

Semantic Mapping materializa a evidência fundida como entidades persistentes **sem decidir se dois suportes são o mesmo objeto**: possui `Entity` (suporte 3D exato por `GeometryReference` com resumos derivados reproduzíveis, estado semântico que mantém toda hipótese, alternativa, conflito, abstenção e sinal sem score, vínculos de evidência com identidade, versão, digest e sequência canônica do artifact de fusão, e estado temporal com frames físicos e resultados de inferência distintos), `EntityReference` (`(semantic_map_id, entity_id)`, porque um id só vale dentro do mapa que o possui), a política de materialização `one-support-one-entity-v1` com identidade determinística e rejeição explícita de candidatos, o `SemanticMappingRunArtifact` e o harness de validação em `evaluation`. O artifact versiona separadamente o schema do run e o schema canônico da entidade e registra `code_version`/`code_digest` junto da configuração e lineage. Quem decide identidade entre suportes é o estágio seguinte, Entity Resolution: Semantic Mapping em si mantém duas entidades para suportes diferentes mesmo com o mesmo label e geometria adjacente; toda a verificação usa fixtures sintéticos, sem run de mapeamento sobre dados reais.

Entity Resolution decide, **sobre entidades de origem que nunca são mutadas**, quais delas são o mesmo objeto físico: possui `EntityCandidateSet` (recuperação permissiva de candidatos, sem all-pairs), `EntityMatchEvidence` (a evidência de uma comparação, com um canal tipado por vez: geometria, semântica, aparência, temporal e, opcional, representação 3D, cada um medido ou indisponível e sem score que os resuma), `ResolutionDecision` (`MATCH`, `DISTINCT` ou `UNRESOLVED`, este último resultado válido, com as regras que dispararam), a política baseline `conservative-staged-resolution-v1` (sobre o estado dos canais, sem soma ponderada, que prefere `UNRESOLVED` a um `MATCH` sem explicação), `ResolvedEntity`/`ResolvedEntityReference` (componentes conexos de `MATCH`, com agregação exata e linhagem de fusão; um componente com uma contradição de transitividade não é fundido), a detecção opcional de candidatos a divisão (só diagnóstico), o `EntityResolutionRunArtifact` (`output_dir` explícito, escrita atômica, reabre sem NumPy nem runtime) e o harness de avaliação em `evaluation` (fusão falsa, duplicata e falha de recuperação em separado, ablação de canais). Toda a verificação usa fixtures sintéticos: **não há run de resolução sobre dados reais** e, portanto, nenhum limiar foi calibrado com evidência real. Cada componente documenta o detalhe em [`entity_resolution`](../src/contextmap/entity_resolution/docs/README.md).

Evaluation, além dos harnesses por capability, possui a infraestrutura de avaliação controlada: o **reference set versionado** (manifesto com amostras ligadas a `SourceObservation`, calibrações, anotações com trust e proveniência declarados, estratos e splits, com digest), as sete famílias de **anotação** (regiões, semântica open-vocabulary, correspondências 3D↔pixel, identidade, relações, visibilidade e contexto de cena, onde ausência nunca é verdade negativa), a **validação de integridade** e de vazamento entre splits, o **QA do conteúdo das anotações** com divergência visível entre anotadores, o **registro versionado de métricas** por estágio e o relatório comum (qualidade e performance separadas, sem score geral), os **manifestos de experimento** e a execução controlada por arm (só as variáveis declaradas variam; arms indisponíveis ficam explícitos), a checagem de reprodutibilidade dos evaluators, um subconjunto determinístico de fixtures para CI (sem fusão multi-vista nem round-trip do artifact final) e os protocolos e a evidência por estrato das duas técnicas opcionais (resolução de features e fusão ciente de qualidade). Nada disso foi executado sobre dados reais: **não existe reference set real versionado** (só o subconjunto sintético de CI), Entity Resolution tem harness de avaliação, mas só sobre dados sintéticos, Spatial Relations tem o avaliador por predicado de relações persistidas, também só sobre dados sintéticos, e **nenhuma decisão foi tomada** sobre as técnicas opcionais.

Runtime & Configuration compõe essas capabilities: possui a configuração efetiva versionada (perfil, arquivos e overrides, digest determinístico, segredos só do ambiente), o catálogo de stages, pontos de variação e backends, a composition root, o DAG de estágios com preflight, o reuso por identidade de conteúdo, a seleção explícita de runs com linhagem, o ciclo de vida do run (journal, eventos, cancelamento, retomada), o serviço público de ingestion, a CLI `contextmap` e a API pública de aplicação `Runtime` para qualquer frontend. `state_estimation`, `geometric_mapping`, `sensor_association` e `semantic_fusion` têm executor real, composto automaticamente da configuração (`compose_executors`); Visual Perception e Point Representation ainda não têm (dependem de backend com modelo/GPU), e a Ingestion tem um executor real que continua exigindo injeção explícita, porque precisa de um `IngestionRequest` da execução, não da configuração. A orquestração end-to-end com dados reais dessas capabilities restantes ainda não foi validada. O preset `canonical/1` termina em `semantic_fusion`; Semantic Mapping, Entity Resolution, Spatial Relations e o `ContextMapArtifact` ainda não são estágios do preset.

Os detalhes implementados pertencem aos documentos dos módulos. Os documentos globais integram esses boundaries e descrevem como eles se conectam ao restante do canonical pipeline, sem duplicar a especificação interna.

## Ordem recomendada de leitura

1. [PIPELINE.md](PIPELINE.md), fluxo completo da entrada ao `ContextMapArtifact`.
2. [architecture.md](architecture.md), boundaries, ownership, dependências e regras arquiteturais.
3. [module-api.md](module-api.md), superfície pública, encapsulamento e imports permitidos entre capabilities.
4. [shared-primitives.md](shared-primitives.md), critérios para tipos transversais e limites do namespace `shared`.
5. [runtime-composition.md](runtime-composition.md), composition root, configuração, optional stages e limite da orquestração.
6. [CONTRACTS.md](CONTRACTS.md), contratos públicos, identidades e semântica dos dados que cruzam módulos.
7. [ARTIFACTS.md](ARTIFACTS.md), persistência, immutability, lineage, debug e reprodutibilidade.
8. [development.md](development.md), fluxo de desenvolvimento, branches, commits, Python e qualidade.
9. [repository-settings.md](repository-settings.md), políticas esperadas do GitHub e checks.
10. [versioning.md](versioning.md), Semantic Versioning e releases.

## Mapa da documentação

```mermaid
flowchart TD
    R[docs/README.md]

    R --> P[PIPELINE.md]
    R --> A[architecture.md]
    R --> API[module-api.md]
    R --> SH[shared-primitives.md]
    R --> RC[runtime-composition.md]
    R --> C[CONTRACTS.md]
    R --> AR[ARTIFACTS.md]
    R --> D[development.md]
    R --> RS[repository-settings.md]
    R --> V[versioning.md]

    A --> M[docs específicos dos módulos]
    API --> M
    SH --> M
    RC --> M
    P --> M
    C --> M
    AR --> M

    M --> ING[src/contextmap/ingestion/docs/README.md]
    M --> VP[src/contextmap/visual_perception/docs/README.md]
    M --> ST[src/contextmap/state_estimation/docs/README.md]
    M --> GM[src/contextmap/geometric_mapping/docs/README.md]
    M --> SA[src/contextmap/sensor_association/docs/README.md]
    M --> PR[src/contextmap/point_representation/docs/README.md]
    M --> SF[src/contextmap/semantic_fusion/docs/README.md]
    M --> SM[src/contextmap/semantic_mapping/docs/README.md]
    M --> EV[src/contextmap/evaluation/docs/README.md]
    M --> ART[src/contextmap/artifact/docs/README.md]
    M --> RTM[src/contextmap/runtime/docs/README.md]
    RTM --> RTD[configuration / composition / pipeline / reuse / selection / lifecycle / cli / api]
    ING --> ID[contracts / artifact / synchronization / calibration / adapters]
    VP --> VD[contracts / ports / pipeline / service / identity / run_artifact / evidence_set]
    VP --> RD[Region Discovery]
    VP --> FE[Feature Extraction]
    VP --> SI[Semantic Interpretation]
```

## Responsabilidade de cada documento

| Documento | Pergunta principal |
| --- | --- |
| `PIPELINE.md` | Como os dados percorrem o sistema do sensor ao mapa contextual? |
| `architecture.md` | Quem é responsável por cada capability e quem pode depender de quem? |
| `module-api.md` | O que uma capability expõe publicamente e como outras capabilities podem importá-la? |
| `shared-primitives.md` | Quais conceitos podem ser compartilhados sem perder ownership de domínio? |
| `runtime-composition.md` | Onde implementations são construídas e como runtime orquestra sem absorver lógica científica? |
| `CONTRACTS.md` | Qual é o significado dos objetos que atravessam as fronteiras entre módulos? |
| `ARTIFACTS.md` | Como runs e resultados são persistidos, auditados e reutilizados? |
| `development.md` | Como uma mudança deve ser implementada e integrada? |
| `repository-settings.md` | Como o GitHub deve reforçar o fluxo de desenvolvimento? |
| `versioning.md` | Como versões e releases são identificadas? |

## Pipeline em uma linha

```text
raw source
→ ingestion
→ visual perception + state estimation
→ geometric mapping
→ sensor association
→ optional point representation
→ semantic fusion
→ semantic mapping
→ entity resolution
→ spatial relations
→ ContextMapArtifact
```

O detalhamento completo, incluindo branches paralelos e contratos intermediários, está em [PIPELINE.md](PIPELINE.md).

## Documentação por módulo

Decisões globais pertencem a `docs/`. Detalhes internos de uma capability pertencem ao próprio módulo:

```text
src/contextmap/<module>/docs/
├── README.md
├── pipeline.md       # quando o fluxo interno exigir detalhamento
├── contracts.md      # quando houver contratos públicos ou invariantes próprios
├── architecture.md   # quando houver decisões estruturais próprias
├── evaluation.md     # quando houver métricas/protocolo de avaliação
└── backends.md       # quando houver múltiplos backends reais
```

Arquivos complementares só devem existir quando houver conteúdo real. O objetivo é fragmentar por responsabilidade, não multiplicar arquivos vazios.

Módulos com documentação própria:

- [`ingestion`](../src/contextmap/ingestion/docs/README.md) — observações de sensor canônicas, sequência, sincronização, calibração, seleção/replay, provenance e adapters de fonte.
- [`visual_perception`](../src/contextmap/visual_perception/docs/README.md) — evidência visual por run, ports, preset canônico, execução do DAG, artifacts e leitura multi-run; [Region Discovery](../src/contextmap/visual_perception/docs/region-discovery.md) documenta SAM2/SAM3/Florence-2, passes, normalização e avaliação geométrica; [Feature Extraction](../src/contextmap/visual_perception/docs/feature-extraction.md) documenta embeddings, payloads, sampling, pooling, diagnostics, enhancement opcional e os adapters concretos; [Semantic Interpretation](../src/contextmap/visual_perception/docs/semantic-interpretation.md) documenta claims/contexto, seleção auditável de evidência, prompts/parsing, Qwen/Gemini/Florence-2 e persistência da execução.
- [`state_estimation`](../src/contextmap/state_estimation/docs/README.md) — pose dinâmica do rig: `PoseEstimate`, `Trajectory`, convenção de transform e provenance.
- [`geometric_mapping`](../src/contextmap/geometric_mapping/docs/README.md) — geometria 3D persistente no frame global do mapa: `GeometryPoint`, `GeometryReference`, `GeometricMap`, `Bounds3D` e a fronteira de leitura `GeometrySource`.
- [`sensor_association`](../src/contextmap/sensor_association/docs/README.md) — evidência visual 2D ancorada em geometria 3D persistente: `SpatialObservation`, modelos de câmera, visibilidade e oclusão, pertencimento à máscara, features densas e `ObservationQuality`.
- [`semantic_fusion`](../src/contextmap/semantic_fusion/docs/README.md) — acumulação de evidência multi-vista sobre suporte espacial, sem identidade de objeto: `FusionSupport`, `FusedEvidence`, agrupamento por observação física, política baseline e ciente de qualidade, canais tipados e o artifact de run.
- [`spatial_relations`](../src/contextmap/spatial_relations/docs/README.md) — relações espaciais entre entidades resolvidas: `Relation`, `RelationEvidence`, taxonomia versionada de predicados, convenções de frame, candidatos, avaliadores geométricos e de contato, evidência de observação, política de decisão e o artifact de run.
- [`semantic_mapping`](../src/contextmap/semantic_mapping/docs/README.md) — entidades semânticas persistentes sem resolução entre suportes: `Entity`, `EntityReference`, resumos geométricos, estado semântico sem colapso, vínculos de evidência, estado temporal, materialização determinística e o artifact de run.
- [`entity_resolution`](../src/contextmap/entity_resolution/docs/README.md) — identidade entre entidades de origem sem mutá-las: candidatos, evidência por canal sem score único, decisões `MATCH`/`DISTINCT`/`UNRESOLVED`, entidades resolvidas com linhagem, contradições de transitividade, detecção de divisão opcional e o `EntityResolutionRunArtifact`.
- [`point_representation`](../src/contextmap/point_representation/docs/README.md) — representação opcional da estrutura 3D local: `PointRepresentation`, `RepresentationSpace`, o port `PointEncoder`, o descritor determinístico, a fronteira PTv3 e o runtime real `PointceptPTv3Runtime`, já medido em geometria real.
- [`evaluation`](../src/contextmap/evaluation/docs/README.md) — relatórios de qualidade, regressão e custo sem alterar outputs do pipeline; [reference set](../src/contextmap/evaluation/docs/reference-set.md), [anotações](../src/contextmap/evaluation/docs/annotations.md), [integridade](../src/contextmap/evaluation/docs/reference-integrity.md), [QA](../src/contextmap/evaluation/docs/annotation-qa.md), [métricas](../src/contextmap/evaluation/docs/metrics.md), [experimentos](../src/contextmap/evaluation/docs/experiments.md), [fixtures de CI](../src/contextmap/evaluation/docs/ci-fixtures.md) e [técnicas opcionais](../src/contextmap/evaluation/docs/optional-techniques.md).
- [`artifact`](../src/contextmap/artifact/docs/README.md) — o schema do produto final: `ContextMap` (metadados de frame, unidades, âncora e capacidades; geometria referenciada; entidades e relações por referência; linhagem e proveniência; versionamento). O schema é independente de layout, serializador, ROS e modelos; a persistência (layout em disco, escrita atômica, leitura preguiçosa, validação de integridade e bundle portátil) fica em módulos separados do pacote, documentados em [Layout e formatos](../src/contextmap/artifact/docs/storage-layout.md).
- [`runtime`](../src/contextmap/runtime/docs/README.md) — configuração efetiva, composition root, DAG com preflight, reuso, seleção de runs, lifecycle, CLI, serviço de ingestion e a API pública `Runtime` para frontends; compõe e executa, sem decidir ciência.

## Integração da documentação

A documentação é fragmentada fisicamente, mas forma um único grafo de conhecimento.

- `docs/` contém decisões transversais ao sistema;
- cada módulo documenta apenas seu domínio e suas fronteiras;
- conteúdo global é referenciado por link, não copiado para cada módulo;
- documentação de upstream/downstream deve apontar para o contrato público relevante;
- quando um módulo já estiver implementado, seu `src/contextmap/<module>/docs/` é a fonte de verdade para detalhes de comportamento; os documentos globais resumem e conectam esse comportamento ao sistema;
- uma alteração arquitetural ou de contrato deve atualizar código e documentação no mesmo PR;
- `docs/README.md` é o índice canônico dos pontos de entrada globais.

## Estado da arquitetura

O canonical pipeline é **pre-alpha** e evolui por milestones. Os documentos principais descrevem o desenho canônico que as milestones devem materializar.

Ao ler uma etapa do pipeline, diferencie:

- **contrato**, semântica que deve permanecer estável na fronteira do módulo;
- **backend**, implementação substituível de uma capability;
- **canonical pipeline**, composição integrada de referência que as milestones materializam e validam;
- **experimento**, alternativa que não substitui silenciosamente o baseline;
- **artefato**, resultado persistido e imutável de uma execução.

## Regras de escrita

- Markdown em PT-BR;
- nomes de APIs, classes e contratos permanecem em inglês;
- Mermaid é preferido para fluxos, dependências, estados e lineage quando reduzir ambiguidade;
- exemplos devem ser pequenos e ligados a uma decisão real;
- documentos globais não devem repetir detalhes que pertencem ao `docs/` de um módulo;
- arquitetura planejada nunca deve ser apresentada como funcionalidade já implementada.
