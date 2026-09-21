# Registro de métricas e relatório comum

Cada capability tem semântica de qualidade própria: juntar IoU de região, erro de pose, erro de reprojeção, qualidade semântica, identidade de entidade, relações e runtime em um único score esconderia a causa de uma falha. Este módulo define (1) um **registro versionado de métricas por estágio** (`contextmap.evaluation.metrics`) e (2) um **envelope de relatório comum** (`contextmap.evaluation.report_schema`) que dá a todo relatório os mesmos metadados de reprodutibilidade, mantendo as métricas específicas de cada estágio.

Não existe score geral. O envelope não tem esse campo.

## Definição de métrica

Cada `MetricDefinition` é legível por máquina (`to_record()`/`from_record()`) e identifica:

| Campo | Significado |
|---|---|
| `name` / `version` | identidade (`name/version`); mudar o significado é uma nova versão |
| `stage` | estágio avaliado; `None` só para métricas de performance transversais |
| `kind` | `quality` (quão correto) ou `performance` (quanto custou) |
| `population` | sobre o que se calcula, incluindo a população anotada e a janela de tempo |
| `unit`, `minimum`, `maximum` | unidade e faixa de valores |
| `direction` | `higher_is_better`, `lower_is_better` ou `not_ordered` |
| `aggregation` | como valores por amostra viram o reportado |
| `required_annotations` | identificadores exatos de schema de anotação (ver [`annotations.md`](annotations.md)) |
| `missing_data` | `exclude_unannotated` (a população fica só com amostras anotadas; não aplicável se não houver nenhuma) ou `not_applicable` (qualquer lacuna a torna não aplicável) |
| `evaluator_id` / `evaluator_version` | quem calcula |

Regras aplicadas na construção:

- a unidade é uma de `ALLOWED_UNITS`; **`probability` não existe**: um score de suporte arbitrário nunca é chamado de probabilidade calibrada (use `score`);
- métrica de qualidade nomeia o estágio; métrica de performance não exige anotações;
- os schemas de anotação exigidos precisam existir.

## Registro

`MetricRegistry` é imutável, tem `registry_id`/`registry_version` e um `digest()` sobre todas as definições; `identity()` é o que os relatórios citam. `get(name, version)` recusa métrica ou versão desconhecida. O `default_metric_registry()` (**v2**) cobre todos os estágios. A v2 acrescenta as métricas que a avaliação de técnicas opcionais ([`optional-techniques.md`](optional-techniques.md)) precisa: `association.feature_anchoring.rate`, `fusion.view_consistency.rate`, `entity.semantic_accuracy.rate` e `runtime.failure_rate`.

| Estágio | Métricas de qualidade (v2) |
|---|---|
| `ingestion_integrity` | `ingestion.integrity.violations`, `ingestion.modality.coverage` |
| `state_estimation` | `state.ate.rmse`, `state.rpe.translation.rmse`, `state.gap.ratio` |
| `geometric_mapping` | `geometry.scan_overlap.plane_distance.median`, `geometry.expected_point.error.max` |
| `region_discovery` | `region.iou.mean`, `region.recall.mean`, `region.duplicate_rate.mean` |
| `feature_extraction` | `feature.finite_ratio`, `feature.repeatability.max_abs_diff` |
| `semantic_interpretation` | `semantic.acceptable_claim_rate`, `semantic.unsupported_claim_rate`, `semantic.ambiguity_preservation_rate` |
| `sensor_association` | `association.reprojection_error.median`, `association.visible_support.ratio`, `association.feature_anchoring.rate` |
| `point_representation` | `pointrep.repeatability.cosine` |
| `semantic_fusion` | `fusion.reference_recovery.rate`, `fusion.ambiguity_retention.rate`, `fusion.view_consistency.rate` |
| `entity_resolution` | `entity.false_merge.rate`, `entity.duplicate.rate`, `entity.semantic_accuracy.rate` |
| `spatial_relations` | `relations.f1`, `relations.negative_violation.rate` |
| `artifact_integrity` | `artifact.integrity.violations`, `artifact.round_trip.mismatches` |
| `runtime` | só performance (abaixo) |

Performance (transversal a qualquer estágio): `runtime.wall_time`, `runtime.peak_memory` (por estrato `device=cpu|gpu`), `runtime.storage_size` (por `artifact_role=intermediate|final`), `runtime.throughput` e `runtime.failure_rate` (falhas, inclusive OOM).

As definições de Entity Resolution e Spatial Relations existem para que os relatórios desses estágios tenham onde se apoiar; os avaliadores correspondentes dependem dos contratos das milestones 12–14 e ainda não existem.

## Relatório comum

`EvaluationReport` traz o estágio, os `ReproducibilityMetadata`, as métricas de **qualidade** e as de **performance** em campos separados, e o relatório do próprio estágio (`stage_report`) intocado.

`ReproducibilityMetadata` reúne: evaluator e versão, identidade do reference set (`id`, `version`, `digest`), schemas de anotação disponíveis, artifacts de entrada, digest da configuração, versão do código e identidade do registro de métricas.

Cada `MetricResult` tem `status`:

| Status | Uso |
|---|---|
| `value` | há valor, dentro da faixa da definição |
| `not_applicable` | anotação ou população ausente; **nunca** vira zero |
| `unsupported` | o evaluator ou o backend não produz a métrica |

Um resultado que não é `value` não carrega valor. `sample_count` é o tamanho da população; `strata` marca um resultado por estrato (uma métrica pode aparecer uma vez por combinação de estratos).

## Validação

`assemble_evaluation_report()` (e `decode_evaluation_report()`, que revalida) confere contra o registro:

- a métrica e a versão existem, e o relatório cita o mesmo registro (digest);
- qualidade e performance não se misturam: uma métrica só entra no contêiner do seu `kind`;
- uma métrica de estágio só entra no relatório desse estágio; um relatório `runtime` carrega só performance;
- o valor respeita `minimum`/`maximum`; nenhum resultado se repete para os mesmos estratos;
- um `value` de métrica que exige anotações exige um reference set citado e o schema de anotação **exato** entre os disponíveis, senão `MetricCompatibilityError`. Um evaluator pode chamar `require_annotation_compatibility()` para recusar uma incompatibilidade de versão antes de calcular.

## Adaptadores dos harnesses existentes

Os avaliadores por capability continuam calculando suas métricas. Os adaptadores só as expressam no envelope, sem recalcular:

- `region_discovery_evaluation_report()` — `region.iou.mean`, `region.recall.mean`, `region.duplicate_rate.mean` sobre os frames anotados (frames sem anotação saem da população, não valem zero) e `runtime.wall_time`/`runtime.peak_memory`;
- `semantic_interpretation_evaluation_report()` — taxas sobre claims, `semantic.ambiguity_preservation_rate` não aplicável quando nenhuma requisição tinha anotação ambígua, e o custo;
- `wrap_stage_report()` — qualquer relatório de estágio (State Estimation, Sensor Association, Semantic Fusion, Geometric Mapping, Feature Extraction, Point Representation) segue dentro do envelope, com os metadados comuns, sem métricas tipadas. Adaptadores específicos para esses estágios podem substituir a chamada sem alterar o relatório do estágio.

## O que este módulo não faz

- não calcula métricas: cada evaluator segue dono do seu cálculo;
- não decide qual configuração vence: não há agregação entre métricas nem score geral;
- não define a execução de experimentos ou ablações (manifestos de experimento).
