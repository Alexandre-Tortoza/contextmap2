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

## Reprodutibilidade e versionamento

O cenário é reproduzível a partir de entradas documentadas: a sequência (identidade e digest do manifesto), a seleção (identidade e janela), a entrada de pose (digest do arquivo), o perfil (backends) e a matriz. `digest` cobre o cenário inteiro e `matrix_digest` só a matriz.

Os snapshots JSON em [`scenarios/`](scenarios/) são a forma revisável do cenário. O teste `test_the_committed_scenario_snapshots_are_exactly_what_the_code_produces` regenera cada snapshot e exige igualdade byte a byte: qualquer decisão congelada alterada quebra a CI. Para mudar uma decisão:

1. aumente `SCENARIO_VERSION`;
2. gere e commite o novo par de snapshots (a escrita não sobrescreve versões anteriores);
3. registre o motivo no histórico abaixo e explique por que a mudança não foi ajustada depois de ver o resultado final.

### Histórico de versões

- **1.0.0** — versão inicial: sujeito real `corridor-02-sample` sem reference set anotado, sujeito de CI sintético e matriz de 25 gates (16 invariantes e 9 de relatório) sem limiares de aprovação.

## O que ainda não é avaliável

O estado real de cada gate é decidido pelo relatório de cada run. Nesta versão do repositório:

- Os gates de `entity_resolution`, `spatial_relations`, `artifact` e os `cross_stage.*` que atravessam esses estágios dependem de capabilities que ainda não estão na `dev`; ficam `blocked`, nomeando a capability.
- Os gates que precisam de anotações ficam `blocked` até existir um reference set anotado para o `corridor-02`.
- `reproducibility.*` e `runtime.resource_reporting` dependem de um run canônico completo e de execuções repetidas.

## Fora do escopo

Visualização, busca em linguagem natural, navegação, planejamento e agentes consomem o `ContextMapArtifact` e não fazem parte do cenário. Este módulo mede; não escolhe backend em tempo de execução, não altera artifacts e não substitui a avaliação de cada capability.
