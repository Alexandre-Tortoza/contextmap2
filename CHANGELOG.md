# Changelog

As mudanças relevantes do ContextMap2 são registradas aqui, no formato do [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/). As versões seguem [docs/versioning.md](docs/versioning.md); durante a fase de validação a série é `0.x.y`.

Uma entrada só recebe data quando a release é criada. O workflow `Release` recusa uma tag cuja versão não tenha, neste arquivo, a entrada `## [X.Y.Z] - AAAA-MM-DD` nem um `docs/releases/vX.Y.Z.md` que já não seja rascunho.

## [Não lançado]

### Alterado

- Schema do pedido de ingestão: `ingestion_request 0.1.0 → 0.2.0`. A correção de timestamp `constant_offset` passa a ser gravada como `offset_nanoseconds` inteiro, em vez de `offset_seconds` em ponto flutuante, para que o documento reconstrua exatamente a mesma correção (#618).
- **Spatial relations:** `SpatialRelationsRunArtifact 0.1.0 → 0.2.0`. As exclusões de candidatos passaram a ter teto por grupo `(predicado, razão)`: um grupo com mais exclusões que o teto lista só as mais próximas (menor distância entre os limites) e guarda num registro `exclusion_summary` a contagem, as distâncias mínima e máxima e um digest de todas elas, e a memória da geração fica limitada. `metrics/counts.json` conta todas as exclusões e ganha `unlisted_exclusions`. O leitor continua abrindo runs `0.1.0`, que listam todas as exclusões. O avaliador de relações vai à versão 3 e atribui a `excluded_unlisted:<predicado>` a falha de recuperação que pode estar entre as exclusões não listadas (#601).
- Schema do `PerceptionRunArtifact` 0.5.0 → 0.6.0: todo run grava `outputs/region-discovery-audit.jsonl`, um registro por frame com os passes de Region Discovery e os diagnostics do backend em cada um, cada candidato rejeitado com o motivo, cada decisão de merge e o digest da configuração de normalização (#611). As rejeições e os merges deixam de se perder ao fim do run. O `PerceptionRunReader` continua abrindo runs 0.5.0 e informa que a auditoria deles não foi registrada, em vez de devolvê-la vazia.
- Os backends de Region Discovery (SAM2, SAM3 e Florence-2) implementam o novo port `AuditedRegionDiscovery`, cujo `discover_audited()` devolve as regiões canônicas junto com a `RegionDiscoveryAudit` que as explica; o port `RegionDiscovery` não muda. O executor de `visual_perception` do runtime exige esse port, grava a auditoria de cada frame no run e recusa na construção um backend que só devolva regiões (#611).
- Region Discovery: `model_version` passa a ser obrigatório em `Sam2Config`, `Sam3Config` e `Florence2Config`; o default `"unknown"` ia para a provenance e o digest no lugar da versão real, e uma configuração sem o parâmetro agora é recusada na composição. Antes da primeira inferência, os runtimes oficiais conferem o device do modelo carregado contra a configuração, e o `from_model` do SAM2 e o runtime do Florence-2 conferem também o dtype contra `precision`; o Florence-2 passa a converter os inputs de ponto flutuante para esse dtype (#617).

### Removido

- `RegionDiscoveryEvidenceWriter`, `DiscoveryAuditRecord`, `DebugLevel` e `WrittenDiscoveryEvidence`: o writer de evidência de estágio de Region Discovery não tinha chamador de produção, e a evidência contratual que ele gravava agora faz parte do `PerceptionRunArtifact` (#611).

## [0.1.0] - 2026-09-25

Primeira release validada da Solution 1. Escopo congelado em [docs/release-v0.1.0.md](docs/release-v0.1.0.md); notas completas, com capacidades validadas e limitações, em [docs/releases/v0.1.0.md](docs/releases/v0.1.0.md).

### Adicionado

- **Ingestion:** adapters de bag ROS 1 e ROS 2 (`rosbags`), observações canônicas, calibração (pinhole, fisheye e MEI), sincronização, seleção e replay, provenance, validação e `SequenceArtifact`.
- **Visual perception:** core de execução, preset canônico versionado, Region Discovery (SAM2, SAM3, Florence-2), Feature Extraction (DINOv2, DINOv3, CLIP, AlphaCLIP), Semantic Interpretation (Qwen3-VL, Gemini, Florence-2 atrás de seams injetáveis), scoring semântico e `PerceptionRunArtifact`.
- **State estimation:** `PoseEstimate` e `Trajectory`, lookup temporal auditável, backends `ExternalPose` e FAST-LIO e `StateEstimationRunArtifact`.
- **Geometric mapping:** `GeometricMap`, correção de movimento explícita, acumulação com referências estáveis, acesso espacial em blocos e `GeometricMapArtifact`.
- **Sensor association:** `SpatialObservation`, modelos de câmera calibrados, cadeia de projeção, visibilidade e oclusão, pertencimento por máscara, amostragem densa de features, diagnósticos de calibração e reprojeção e `SensorAssociationRunArtifact`.
- **Point representation** (opcional): descritor geométrico determinístico, fronteira e runtime do PTv3 e `PointRepresentationRunArtifact`.
- **Semantic fusion:** `FusionSupport`, acumulação baseline e ciente de qualidade, preservação de ambiguidade e conflito e `SemanticFusionRunArtifact`.
- **Semantic mapping:** modelo de entidades, estado semântico com hipóteses e `SemanticEntityArtifact`.
- **Entity resolution:** recuperação de candidatos, resolução conservadora em estágios, decisões auditáveis e `EntityResolutionRunArtifact`.
- **Spatial relations:** taxonomia, convenções de frame, candidatos por vizinhança, predicados geométricos e de contato, evidência por relação e `SpatialRelationsRunArtifact`.
- **`ContextMapArtifact`:** schema, montagem, escrita atômica, leitor e validador em dois níveis (`STRUCTURAL` e `FULL`), tudo legível na instalação base.
- **Runtime:** configuração e perfil `canonical/1`, composition root, DAG dos 12 estágios, reuso e seleção de runs por referência, ciclo de vida com retomada e o CLI `contextmap` (também `python -m contextmap`), com os executores reais das capabilities ligados.
- **Evaluation:** protocolos determinísticos por capability, registro de métricas versionado, cenário canônico congelado e relatório de aceitação.
- **Empacotamento:** extras `ros1`, `ros2`, `vision` e `gemini`; licença como expressão SPDX (`AGPL-3.0-only`); smoke de instalação em ambiente novo; jobs de CI para compatibilidade de Python, pacote e instalação leve.
- **Release:** workflow com gate de ancestralidade em `main`, reuso da CI, conferência da versão da wheel, checksums e notas validadas.
- **Exemplos:** `examples/v0.1.0/` com artifact de demonstração sintético, configuração canônica, configuração efetiva e plano resolvido, verificados na instalação base.
- **Documentação:** arquitetura, pipeline, contratos, artifacts, configuração de runtime, instalação, licenças de terceiros, estado verificado das configurações do repositório, escopo congelado e notas de release.

### Alterado

- O NumPy passou a ser dependência base (`numpy<2.4`); antes só existia nos extras.
- O workflow de release deixou de usar uma action de terceiros e passou a publicar somente o que a CI construiu e testou.
- `sensor_association` passou a selecionar candidatos por frame antes de projetar e a gravar cada frame em fluxo, em vez de retê-los todos: sobre a janela medida, 10,0x em tempo e 8,1x em pico de memória, com equivalência por `GeometryReference` demonstrada (#562, #563, #564).
- O cenário de aceitação `solution-1-canonical` foi de 1.0.4 para **1.0.5**, o contrato de release: `reproducibility.rerun_equivalence` foi estreitado para runs repetidos a partir do mesmo `PerceptionRunArtifact`, o gate de relatório `reproducibility.semantic_rerun_agreement` foi acrescentado e o backend Qwen3-VL foi declarado experimental. A 1.0.4 permanece imutável e reprovada, como registro da campanha que rodou contra ela.

### Limitações desta release

O backend semântico Qwen3-VL é experimental: execuções independentes da configuração canônica produziram 32,2% (29/90) de concordância exata de claims. Não há conjunto de referência anotado para o `corridor-02`, então os cinco gates de qualidade ficam bloqueados e nenhuma qualidade semântica, de entidades ou de relações é afirmada. Lista completa em [docs/releases/v0.1.0.md](docs/releases/v0.1.0.md).
