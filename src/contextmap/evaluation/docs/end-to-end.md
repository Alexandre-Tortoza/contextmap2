# Cenário end-to-end e matriz de aceitação

Cada capability pode passar nos próprios testes e o pipeline completo ainda falhar por premissas incompatíveis, calibração ruim, linhagem errada ou propagação semântica fraca. Este documento define, de forma congelada e versionada, **o que significa "Solution 1 validada"**: uma sequência registrada, um perfil de backends e uma matriz de gates por estágio.

O código está em `contextmap.evaluation.end_to_end` (tipos, relatório) e `contextmap.evaluation.canonical_scenario` (o cenário congelado). O produto de uma validação é um `AcceptanceReport`, nunca um score.

## Princípios

- **Sem score global.** Não existe "nota do mapa", "passou/falhou geral" nem campo agregado no relatório. Cada gate tem resultado próprio e nomeia a capability que responde pela falha.
- **Bloqueio é explícito.** Um gate que não pode ser decidido (capability ausente, anotação ausente) fica `blocked` e cita o que o bloqueia; nunca vira zero, `passed` ou omissão.
- **Real versus contrato.** Um gate só conta como cumprido com `passed` e evidência `real`. O ensaio sobre fixtures sintéticas (`fake_contract`) protege regressões e nunca valida a Solution 1.
- **Anotação é só avaliação.** Nenhuma anotação de referência entra na inferência. A trajetória do dataset entra como **entrada de pose declarada e hasheada**, não como referência.
- **Nada é reparado no avaliador.** Se um artifact intermediário é inválido, o gate falha e nomeia a capability; o avaliador não conserta o artifact.

## O cenário congelado

`solution-1-canonical` versão `1.0.0` tem dois **sujeitos** que compartilham perfil e matriz (mesmo `matrix_digest`):

| Sujeito | Evidência | O que é |
|---|---|---|
| `corridor-02-sample` | `real` | Janela de 90 s da sequência `corridor-02` (artifact `e145f73f…`), 20 imagens, 892 scans LiDAR, 17 983 amostras IMU, relógio único `corridor-02-header`. |
| `ci-synthetic-subset` | `fake_contract` | O subconjunto sintético `contextmap-ci-subset` 1.0.1 (3 frames, sem GPU, rede ou modelo). |

### Sujeito real

A janela vem de uma regra de seleção, registrada antes de qualquer resultado final: janelas de 90 s (passo de 1 s) mantidas se a maior lacuna de pose ≤ 605 ms, a maior lacuna de LiDAR ≤ 250 ms, o percurso ≥ 40 m e a guinada acumulada ≥ 60° (trecho reto mais curvas, para que erros de montagem e de tempo sejam observáveis); as imagens são as mais próximas de alvos espaçados de 4,5 s, mantidas só com um scan a menos de 60 ms e lacuna de pose ≤ 450 ms. O cenário grava a janela, as 20 imagens com o scan mais próximo, a `selection_identity` (`sha256:dc641b34…`) e o digest do manifesto do artifact de sequência.

A sequência **não tem observações de pose**: a trajetória vem de `corridor-02-gt.txt` (ordem TUM, relógio do header), declarada como entrada `pose_source` com `sha256`. Por isso o gate de acurácia da trajetória é **não aplicável** para o `ExternalPose`: comparar a trajetória com o arquivo de que ela veio daria erro zero por construção.

**Ainda não existe reference set anotado para o `corridor-02`.** O sujeito declara `reference_set = null`: os gates que precisam de anotação ficam `blocked`. Quando um reference set existir, ele entra em uma **nova versão** do cenário (a versão é o mecanismo de imutabilidade), nunca por edição.

### Sujeito de CI

O reference set do subconjunto sintético é fixado por identidade (`contextmap-ci-subset`, `1.0.1`, digest do manifesto commitado). Um teste compara a identidade do cenário com o `manifest.json` commitado: se o subconjunto mudar de versão, o cenário precisa de uma nova versão.

## Perfil canônico

Escolher os backends canônicos é uma decisão científica que pertence a esta validação, não à runtime. O critério foi: backend **local, de pesos abertos, que já executou de verdade na amostra e cabe em uma GPU de 8 GB**. O `model` é a família; a revisão exata ou o hash do checkpoint é gravado pelo próprio run e verificado pelo gate `runtime.provenance_identity`.

| Estágio | Ponto de variação | Backend canônico | Modelo |
|---|---|---|---|
| `ingestion` | `ingestion.source_adapter` | `ros1_bag` | — |
| `visual_perception` | `region_discovery` | `sam2` | `facebook/sam2.1-hiera-tiny` |
| | `dense_features` | `dinov2` | `facebook/dinov2-base` |
| | `region_features` | `clip` | `openai/clip-vit-large-patch14` |
| | `semantic_interpretation` | `qwen` | `Qwen/Qwen3-VL-4B-Instruct` |
| `state_estimation` | `estimator` | `external_pose` | — |
| `geometric_mapping` | — | sem escolha de backend | — |
| `sensor_association` | — | políticas de visibilidade e de pertencimento versionadas pela capability | — |
| `semantic_fusion` | `support` | `geometry-jaccard-support-v1` | — |
| | `accumulation` | `baseline-evidence-accumulation-v1` | — |
| `semantic_mapping`, `entity_resolution`, `spatial_relations`, `context_map` | — | políticas versionadas pelas respectivas capabilities, gravadas no run | — |

Nenhum componente canônico chama serviço externo. O `Qwen/Qwen3-VL-4B-Instruct` é o candidato local que coube em uma GPU de 8 GB na auditoria anterior (com quantização); a taxa de respostas interpretáveis foi parcial e a validação real do runtime pertence à milestone de Semantic Interpretation. O run canônico só é aceito se a revisão e a quantização efetivas ficarem gravadas na proveniência, e uma troca de modelo é uma nova versão do cenário.

### O perfil como configuração do runtime

O cenário é a **única fonte** do perfil canônico. `scenario_runtime_document()` devolve o documento de configuração do runtime que o expressa: o preset `canonical/1`, os estágios opcionais desligados (`point_representation: false`) e **só a seleção de backend** de cada ponto de variação. Checkpoint, revisão e limiares pertencem a quem os possui e ao run que os grava; o documento não os fixa.

Gravado como `.json` e passado ao runtime, o documento resolve a configuração efetiva sem problema de seleção (`check_selection()` vazio). Os testes de `tests/end_to_end/test_canonical_runtime_profile.py` verificam que todo estágio exigido pertence à topologia do runtime, que todo backend citado existe no catálogo e que o digest da configuração efetiva não depende do caminho do arquivo. Os estágios `semantic_mapping`, `entity_resolution`, `spatial_relations` e `context_map` estão habilitados no documento e continuam **indisponíveis** no runtime até as capabilities existirem: a execução os reporta, não os ignora.

### Opcionais: só por ablação

Capabilities implementadas que **não** fazem parte do run canônico. Uma opção só sai desta lista por uma ablação end-to-end controlada que mostre benefício relativo ao custo, uma nova versão do cenário e um run canônico aprovado; nunca pelo resultado de um paper ou por uma métrica de estágio isolada.

| Componente | Opção comparada | Por que não é canônica |
|---|---|---|
| `point_representation` | `off` × descritor determinístico × PTv3 | custo não justificado por avaliação controlada; desligada |
| `semantic_fusion.accumulation` | `quality-aware-evidence-accumulation-v1` | sem decisão de adoção |
| `visual_perception.semantic_interpretation` | `gemini` | envia frames a serviço externo pago; exige identidade exata, disponibilidade e custo |
| | `florence2` | fala por task tokens; a tradução para claims é experimento próprio |
| `visual_perception.region_discovery` | `sam3` | segmentação conceitual por texto, semântica diferente do SAM2 |
| `visual_perception.dense_features` | `dinov3` | sem execução real registrada |
| | `feature-resolution-enhancement` | estágio opcional do DAG; comparado sobre o mesmo artifact DINO |
| `visual_perception.region_features` | `alphaclip` | sem checkpoint de origem verificável |
| `state_estimation.estimator` | `fast_lio` | comparado com a pose de entrada como estimador |
| `entity_resolution` | evidência semântica e de aparência além da geometria | a linha de base resolve por geometria |

## Matriz de aceitação

`INVARIANT` é uma propriedade binária: violá-la reprova o gate. `REPORT` exige que as métricas sejam reportadas com denominadores explícitos; **não há limiar de aprovação na versão 1.0.0**. Um limiar é uma decisão congelada que exige nova versão do cenário, calibrada em um split que não seja o held-out, para nunca ser ajustada depois de ver o resultado final.

| Gate | Capability | Tipo | Métricas do registro | Anotações |
|---|---|---|---|---|
| `ingestion.sequence_integrity` | `ingestion` | invariante | `ingestion.integrity.violations`, `ingestion.modality.coverage` | — |
| `state_estimation.trajectory_coverage` | `state_estimation` | invariante | `state.gap.ratio` | — |
| `state_estimation.accuracy` | `state_estimation` | relatório | `state.ate.rmse`, `state.rpe.translation.rmse` | — (referência independente) |
| `geometric_mapping.map_frame_consistency` | `geometric_mapping` | invariante | — | — |
| `geometric_mapping.quality_report` | `geometric_mapping` | relatório | `geometry.scan_overlap.plane_distance.median` | — |
| `visual_perception.evidence_completeness` | `visual_perception` | invariante | — | — |
| `visual_perception.region_quality` | `visual_perception` | relatório | `region.iou.mean`, `region.recall.mean`, `region.duplicate_rate.mean` | `regions` |
| `visual_perception.semantic_quality` | `visual_perception` | relatório | `semantic.acceptable_claim_rate`, `semantic.unsupported_claim_rate`, `semantic.ambiguity_preservation_rate` | `semantics` |
| `sensor_association.projection_validity` | `sensor_association` | invariante | — | — |
| `sensor_association.projection_quality` | `sensor_association` | relatório | `association.visible_support.ratio`, `association.reprojection_error.median`, `association.feature_anchoring.rate` | — |
| `semantic_fusion.evidence_preservation` | `semantic_fusion` | invariante | — | — |
| `semantic_fusion.reference_recovery` | `semantic_fusion` | relatório | `fusion.reference_recovery.rate`, `fusion.ambiguity_retention.rate`, `fusion.view_consistency.rate` | `semantics` |
| `entity_resolution.identity_lineage` | `entity_resolution` | invariante | — | — |
| `entity_resolution.identity_quality` | `entity_resolution` | relatório | `entity.false_merge.rate`, `entity.duplicate.rate`, `entity.semantic_accuracy.rate` | `identity` |
| `spatial_relations.reference_integrity` | `spatial_relations` | invariante | — | — |
| `spatial_relations.relation_quality` | `spatial_relations` | relatório | `relations.f1`, `relations.negative_violation.rate` | `relations` |
| `artifact.integrity` | `artifact` | invariante | `artifact.integrity.violations`, `artifact.round_trip.mismatches` | — |
| `cross_stage.lineage_closure` | `cross_stage` | invariante | — | — |
| `cross_stage.coordinate_consistency` | `cross_stage` | invariante | — | — |
| `cross_stage.evidence_traceability` | `cross_stage` | invariante | — | — |
| `cross_stage.physical_observation_identity` | `cross_stage` | invariante | — | — |
| `runtime.provenance_identity` | `runtime` | invariante | — | — |
| `runtime.resource_reporting` | `runtime` | relatório | `runtime.wall_time`, `runtime.throughput`, `runtime.peak_memory`, `runtime.storage_size`, `runtime.failure_rate` | — |
| `reproducibility.rerun_equivalence` | `runtime` | invariante | — | — |
| `reproducibility.interruption_recovery` | `runtime` | invariante | — | — |

Os enunciados completos de cada gate (requisito e evidência) estão no snapshot JSON do cenário. Os gates `cross_stage.*` falam de uma **fronteira**: o resultado nomeia a capability que quebrou o contrato (por exemplo, `geometric_mapping` quando o frame do mapa diverge do da trajetória).

## Como um gate é atribuído

- `GateResult.passed`/`failed` exigem classe de evidência (`real` ou `fake_contract`) e ao menos uma referência exata (identidade de artifact, digest de relatório ou caminho). Um resultado sem evidência é recusado.
- `GateResult.failed` e `GateResult.blocked` exigem a capability responsável (`failing_capabilities`/`blocked_by`).
- `GateResult.not_evaluated` diz explicitamente que ninguém tentou decidir o gate neste passo.
- `assemble_acceptance_report()` exige **exatamente um resultado por gate**: uma omissão é erro, não um `passed` implícito.
- `unmet_required_gates()` devolve todo gate que não foi cumprido com evidência real, com a capability e o motivo; um relatório sem pendências só existe se todo gate passou com evidência real.

## Invariantes entre estágios

`contextmap.evaluation.cross_stage` valida as **fronteiras** entre os artifacts persistidos, lendo só manifests e objetos públicos das capabilities. Nunca repara um artifact inválido: uma fronteira quebrada vira um `CrossStageFinding` que nomeia a capability responsável, e `cross_stage_gate_results()` o converte nos quatro gates `cross_stage.*`.

| Gate | O que verifica |
|---|---|
| `cross_stage.lineage_closure` | todo artifact declara a mesma sequência; mapa, associações e fusão apontam para a trajetória, o run de estado, o mapa e os runs de percepção e de associação exatos que foram consumidos; a calibração do mapa e das associações é a mesma e é a do artifact de sequência (quando informada) |
| `cross_stage.coordinate_consistency` | o frame do mapa é o de referência da trajetória e ambos usam o mesmo relógio; toda referência de geometria de uma observação ou suporte aponta para o mapa do run **e resolve nele** |
| `cross_stage.evidence_traceability` | toda observação espacial aponta para um resultado, região e claims de percepção que existem; todo suporte de fusão cita observações que uma associação produziu; nenhuma claim mantida pela associação some da contribuição da fusão |
| `cross_stage.physical_observation_identity` | a fusão tem um grupo por frame físico das observações do suporte (inferência repetida não vira observação nova) e cada observação está no grupo do seu frame |

Regras de decisão:

- Um gate com achado **falha** e nomeia as capabilities; o que mais estiver sem verificar não o salva.
- Um gate sem achado, mas com fronteiras ainda não verificáveis (Entity Resolution, Spatial Relations, artifact final), fica **`blocked`** por essas capabilities: passá-lo afirmaria uma fronteira que ninguém checou. O detalhe registra quantas checagens rodaram (um gate com zero checagens não provou nada).
- Uma identidade que um artifact não registra vira **limitação declarada**, não passe silencioso. Exemplo real: o `ExternalPose` não consome calibração e a trajetória grava `calibration_identity = null`; a linhagem de calibração fica verificada só entre o mapa e as associações.

## Harness sintético de CI

`tests/end_to_end/chain.py` executa o código real de cada capability implementada sobre a sequência sintética do subconjunto de CI e persiste o artifact real de cada estágio (sequência, `ExternalPose`, mapa, uma associação por run de percepção, fusão), relido pelo leitor público. Só as saídas de modelo são falsas (máscaras e claims enlatadas); uma segunda run repete o `frame-0000` e discorda, para cobrir inferência repetida sobre uma observação física, alternativa e contradição. Duas execuções produzem os mesmos hashes contratuais em todos os estágios.

`tests/end_to_end/acceptance.py` monta com essa cadeia um `AcceptanceReport` sobre o cenário de CI. Cinco gates passam com evidência `fake_contract` (integridade da sequência, cobertura da trajetória, validade da projeção, preservação de evidência na fusão, equivalência entre reexecuções); o gate de frame do mapa fica `not_evaluated` porque os traços de transformação e o round trip do artifact não rodam na cadeia; os de Entity Resolution, Spatial Relations e artifact final e os `cross_stage.*` ficam `blocked` nomeando a capability; os demais ficam `not_evaluated`. Por ser contrato, `unmet_required_gates()` continua listando os 25 gates: **o ensaio de CI nunca valida a Solution 1**.

## Reprodutibilidade e versionamento

O cenário é reproduzível a partir de entradas documentadas: a sequência (identidade e digest do manifesto), a seleção (identidade e janela), a entrada de pose (digest do arquivo), o perfil (backends) e a matriz. `digest` cobre o cenário inteiro e `matrix_digest` só a matriz.

Os snapshots JSON em [`scenarios/`](scenarios/) são a forma revisável do cenário. O teste `test_the_committed_scenario_snapshots_are_exactly_what_the_code_produces` regenera cada snapshot e exige igualdade byte a byte: qualquer decisão congelada alterada quebra a CI. Para mudar uma decisão:

1. aumente `SCENARIO_VERSION`;
2. gere e commite o novo par de snapshots (a escrita não sobrescreve versões anteriores);
3. registre o motivo no histórico abaixo e explique por que a mudança não foi ajustada depois de ver o resultado final.

### Histórico de versões

- **1.0.0** — versão inicial: sujeito real `corridor-02-sample` sem reference set anotado, sujeito de CI sintético e matriz de 25 gates (16 invariantes e 9 de relatório) sem limiares de aprovação.

## Validação real registrada

- [`validation/e2e-real-sample-20260921.md`](validation/e2e-real-sample-20260921.md) — primeira execução **real e parcial** sobre o `corridor-02` (geometria, associação, fusão e as checagens entre estágios, em CPU): 3 gates cumpridos com evidência real, 1 reprovado (`cross_stage.lineage_closure`, porque o `SequenceArtifact` pinado não carrega modelo de câmera), 12 bloqueados e 9 não avaliados, cada um com o motivo. O relatório de aceitação está em [`validation/e2e-real-sample-20260921.acceptance-report.json`](validation/e2e-real-sample-20260921.acceptance-report.json) e um teste o mantém coerente com o cenário congelado.

## Evidência de contrato pela runtime

`tests/end_to_end/test_runtime_chain.py` roda a cadeia sintética por `run_plan` e `RunJournal`, com os executores reais de `contextmap.runtime.executors` (a sequência e a percepção são dublês). É evidência **`fake_contract`**, e cobre, por gate:

| Gate | O que o teste mostra |
|---|---|
| `reproducibility.rerun_equivalence` | dois runs da mesma execução produzem as mesmas identidades de artifact e os mesmos hashes de conteúdo em todos os estágios |
| `reproducibility.interruption_recovery` | um run interrompido em `semantic_fusion` deixa os estágios concluídos, nenhuma pasta do estágio que falhou e nenhum `.tmp-*`; `resume_plan` o retoma como run novo reutilizando os concluídos **por referência** e refaz só o restante |
| reuso | um terceiro run reutiliza todos os estágios por `ArtifactRef` (`location` aponta para o run anterior), sem pasta nem cópia no run novo |
| layout | todo estágio grava só em `<workspace>/<dataset>/<run>/<estágio>/`; não existe `sequences/`, `runs/` nem `runs.json` |

Nenhum desses gates fica cumprido por isso: só evidência `real` cumpre um gate.

## O que ainda não é avaliável

O estado real de cada gate é decidido pelo relatório de cada run. Nesta versão:

- **Executores.** Existem para ingestion, State Estimation, Geometric Mapping, Sensor Association, Semantic Fusion, Semantic Mapping, Entity Resolution e Spatial Relations ([`runtime/docs/executors.md`](../../runtime/docs/executors.md)); as políticas dos dois últimos vêm de quem constrói o executor, e nenhuma configuração do runtime as carrega. Faltam: percepção (modelos e GPU: entra por referência) e `context_map`.
- **Não existe o passo de montagem do `ContextMapArtifact`.** O schema, o writer e o leitor existem, mas nada transforma os runs de resolução e de relações em um `ContextMap`; só o builder de teste fixa à mão o estado semântico. Os gates `artifact.integrity` e `cross_stage.*` que atravessam esse estágio continuam `blocked`.
- **Não há run real pela runtime.** A sequência pinada não tem observações de pose nem modelo de câmera, e a **ingestion não tem janela de seleção**: o adaptador ROS 1 lê o bag inteiro (o hash da fonte custa O(24 GB) e só se desliga com `hash_source = false`). Uma ingestão só da janela de 90 s também mudaria os ids de observação (o contador é por tópico desde o início do bag), então os runs de percepção existentes deixariam de valer. Os runs de Semantic Interpretation da validação (`q3vl4b-nf4`, `f2-*`) são registros de execução (`run.json`, `executions.jsonl`), **não** `PerceptionRunArtifact`, e estão ligados à sequência pinada.
- **Reingestão da calibração do `corridor-02`.** O `SequenceArtifact` pinado não carrega modelo de câmera; um run canônico real exige um novo artifact de sequência (uma ingestão com o modelo MEI, sem hash da fonte inteira) e, por isso, uma nova versão do cenário.
- Os gates que precisam de anotações ficam `blocked` até existir um reference set anotado (#179) e os que dependem de modelos externos e chaves (#180) ficam abertos por decisão.
- `runtime.resource_reporting` só tem medições parciais de CPU dos estágios da validação real parcial; não há perfil de GPU nem de armazenamento de um run canônico completo.

## Fora do escopo

Visualização, busca em linguagem natural, navegação, planejamento e agentes consomem o `ContextMapArtifact` e não fazem parte do cenário. Este módulo mede; não escolhe backend em tempo de execução, não altera artifacts e não substitui a avaliação de cada capability.
