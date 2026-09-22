# Evaluation

## Responsabilidade

Medir qualidade, regressões e custo das capabilities do ContextMap2 sem alterar os resultados do pipeline. As implementações atuais cobrem Feature Extraction, Region Discovery e Semantic Interpretation por meio de relatórios determinísticos sobre contratos públicos de `visual_perception`, State Estimation (relatórios sobre `Trajectory` e seus artifacts), Sensor Association (relatórios estratificados sobre um `SensorAssociationRunArtifact`), Geometric Mapping (validação de um `GeometricMapArtifact` persistido), Point Representation (harness de ablação entre `off`, descritor determinístico e encoders aprendidos) Semantic Fusion (consistência multi-vista, preservação de incerteza e ablações de política e de canais) e Semantic Mapping (validação de um `SemanticMappingRunArtifact` em seis camadas: contrato, preservação semântica, linhagem de evidência, tempo, fronteira de materialização e round-trip) e Entity Resolution (validação de um `EntityResolutionRunArtifact` e avaliação de identidade contra uma referência explícita, com fusão falsa, duplicata e recuperação em separado). Além dos harnesses por capability, o módulo possui o **reference set** versionado (manifesto de amostras, anotações, proveniência e splits) contra o qual as avaliações oficiais rodam.

## O que este módulo explicitamente não possui

- extração de features, pooling ou transformação espacial;
- seleção/composição de backends;
- mutação de `PerceptionRunArtifact` ou de qualquer artifact avaliado;
- um score global que misture qualidade, runtime, memória e tamanho de payload;
- regras de compatibilidade de embedding — consome a regra pública de `visual_perception`.

## Contratos públicos

### Feature Extraction

- `FeatureEvaluationContext` — identidades mantidas fixas em uma comparação controlada: observação, conteúdo da imagem preparada, artifact/modelo de origem, conjunto de regiões e configuração downstream.
- `FeatureEvaluationReport` — relatório com blocos independentes `FeatureNumericalReport`, `FeatureSpatialReport` e `FeatureCostReport`.
- `FeatureResolutionComparisonReport` — agrupamento auditável de mapas densos native e enhanced compatíveis, sem colapsar métricas em um score único.
- `evaluate_feature_payload()` — valida shape, dtype, finitude, dimensão, normalização declarada e geometria densa antes de construir o relatório.
- `assert_repeatable_feature_outputs()` — verifica metadata e valores repetíveis sob o mesmo contexto, ignorando custo e identidades locais de runs imutáveis.
- `compare_feature_map_resolutions()` — exige contexto controlado, mesma imagem e mesmo `EmbeddingSpace` para comparar resoluções.
- `encode_feature_evaluation_report()`/`encode_feature_resolution_comparison_report()` — representação JSON-compatible para artifacts de avaliação futuros.

### Region Discovery

- `RegionDiscoveryReferenceSet`/`ReferenceFrame`/`GroundTruthRegion` — referência versionada e inspecionável para avaliação geométrica.
- `RegionDiscoveryEvaluator`/`RegionDiscoveryEvaluationReport` — métricas por frame e agregadas, diagnósticos e custo sem alterar evidência de descoberta.
- `compare_region_discovery_reports()` — comparação controlada entre relatórios que rejeita drift de variáveis opacas.
- `write_region_discovery_reference_set()`/`write_region_discovery_report()` — persistência imutável dos inputs e resultados de avaliação.

### Semantic Interpretation

- `SemanticEvaluationContext` — reference-set, seleção, run, artifact,
  pipeline digest e versão do evaluator.
- `SemanticAnnotation`, `SemanticEvaluationInput` e
  `SemanticEvaluationFailure` — referência open-vocabulary, variante de
  evidência e falha `parser`/`backend` explicitamente classificadas.
- `SemanticEvaluationReport`, `SemanticSampleReport`,
  `SemanticQualityReport` e `SemanticCostReport` — amostras, qualidade, custo e
  falhas em blocos separados.
- `evaluate_semantic_interpretation()` — avaliação por request/evidence variant
  com matching policy versionada.
- `MATCHING_POLICY`, `SemanticBackendComparison` e
  `compare_semantic_backends()` — exigem exatamente os mesmos requests sobre as
  mesmas observações, regiões, modos e variants (falhas incluídas, com identidade
  física completa) e o mesmo reference-set, seleção, run de percepção, versão do
  evaluator e matching policy, para Qwen, Gemini e Florence-2.

### State Estimation

- `evaluate_state_estimation()`/`StateEstimationEvaluationReport` — relatório comum a qualquer backend, com seções independentes: `StructuralReport`, `MotionReport`, `TransformTraceReport`, `AccuracyReport` e `CostReport`.
- `MotionThresholds` — limiares de movimento fornecidos pelo perfil de referência; não há valores padrão.
- `ReferenceTrajectory`/`ReferenceRole`/`ReferenceComparisonConfig`/`AlignmentMethod` — referência com papel declarado e protocolo de comparação explícito (associação, alinhamento, ATE, RPE).
- `StateEstimationReferenceProfile`/`decode_reference_profile()`/`encode_reference_profile()` — perfil de referência de um dataset: papel declarado da referência, amarrado ao hash do arquivo (e `declare_reference()` só aceita uma trajetória cujas poses são amostras desse arquivo), limiares de movimento e protocolo de comparação. O perfil do `corridor-02` está em `profiles/`.
- `trace_transform_chain()` — cadeia `T_reference_sensor(t)` reconstruível com erro numérico de composição e de round trip.
- `compare_state_estimation_reports()`/`StateEstimationComparison` — comparação controlada entre backends que rejeita drift de sequência, seleção, referência ou protocolo, ou duas calibrações diferentes (um backend que não consome calibração não conflita com um que consome).
- `encode_state_estimation_report()` — representação JSON do relatório com todas as identidades.

### Semantic Fusion

- `evaluate_semantic_fusion()`/`SemanticFusionEvaluationReport` — relatório de um run de fusão persistido, com seções separadas: correlação, incerteza, canais, ponderação, recuperação da referência, estratos e custo, e a linhagem completa (`FusionLineage`).
- `FusionStratificationProfile` — bordas das estratificações; não há valores padrão.
- `ReferenceAnnotation` — rótulo de referência por observação espacial; ausência é "não aplicável".
- `FusionArmRole`, `compare_semantic_fusion_reports()`/`SemanticFusionComparison` — comparação controlada de braços (baseline, ciente de qualidade e ablações de canais) sobre a mesma base de evidência, sem vencedor nem escore.
- `encode_semantic_fusion_report()`/`encode_semantic_fusion_comparison()` — representação JSON com todas as identidades.

### Semantic Mapping

- `evaluate_semantic_mapping()`/`SemanticMappingEvaluationReport` — validação de um run de mapeamento persistido contra o run de fusão e o mapa geométrico de origem, em seis camadas (`SemanticMappingValidationLayer`), cada `SemanticMappingValidationCheck` com o que examinou e as falhas exatas; sem score composto.
- `SemanticMappingEvaluationLineage` — identidades do run de fusão, do mapa geométrico, da política e versão de materialização, do schema das entidades, do código e do avaliador.
- `SemanticMappingEvaluationError` — recusa a avaliação quando o upstream oferecido não é o da linhagem.
- `encode_semantic_mapping_report()` — representação JSON com todas as identidades.

Detalhes: [validação de Semantic Mapping](semantic_mapping.md).

### Entity Resolution

- `evaluate_entity_resolution()`/`EntityResolutionEvaluationReport` — validação de um run de resolução persistido em três camadas (`ResolutionValidationLayer`) e avaliação de identidade contra uma `IdentityAnnotationSet` explícita: `entity.false_merge.rate` e `entity.duplicate.rate` do registro, mais cada classe de falha em separado (fusão falsa, fusão perdida por causa, `DISTINCT` errado, abstenção); sem score composto.
- `OccurrenceLink` — vínculo explícito entre uma ocorrência anotada (amostra, observação, região) e a entidade de origem; nunca adivinhado.
- `IdentityEvaluation.identity_of_resolved_entity` — identidade anotada de cada entidade resolvida, para quem avalia saídas construídas sobre elas (Spatial Relations).
- `EvaluationReproducibility` — versão de schema e digest do run avaliado (`resolution_artifact_digest`), referência, políticas e código.
- `evaluate_splits()`/`SplitReference` — diagnóstico de divisão à parte da qualidade de fusão.
- `evaluate_channel_ablation()`/`ResolutionArm`/`ArmEvaluation` — braços por conjunto de canais sobre as mesmas entidades, candidatos e referência.
- `encode_entity_resolution_report()` — representação JSON com todas as camadas separadas.

Detalhes: [avaliação de Entity Resolution](entity_resolution.md).

### Sensor Association

- `evaluate_sensor_association()`/`SensorAssociationEvaluationReport` — relatório estratificado de um run persistido, lido pelo leitor público, com a linhagem completa (`SensorAssociationLineage`).
- `StratificationProfile` — bordas das faixas de alcance, visibilidade, densidade de suporte, distância à borda da imagem e ângulo de visada; não há valores padrão.
- `StratificationReport`/`StratumReport` — observações particionadas por um fator mensurável, com o estrato `unavailable` explícito e contagem de **frames físicos** distintos.
- `FeaturePathReport` — como cada canal de features densas (nativo ou melhorado) foi amostrado, com as fontes exatas do manifest.
- `TimingReport`/`ReprojectionReport` — alinhamento temporal das poses e resíduos de reprojeção, só onde há referência confiável.
- `compare_sensor_association_reports()`/`SensorAssociationComparison` — comparação controlada que rejeita tudo que não seja o caminho de features.
- `encode_sensor_association_report()`/`encode_sensor_association_comparison()` — representação JSON com todas as identidades.

### Geometric Mapping

- `evaluate_geometric_mapping()`/`GeometricMappingEvaluationReport` — relatório de um mapa persistido, com seções independentes: `GeometricMappingStructureReport`, `GeometricMappingTraceReport`, `ExpectedPointCheck`, `MapDensityReport`, `MapRangeReport`, `MapBoundsReport`, `ScanOverlapReport`, `ReferenceGeometryReport` e `GeometricMappingCostReport`.
- `GeometricMappingProtocol` — todas as decisões de amostragem e todos os limiares, fornecidos pelo perfil; sem valores padrão.
- `ExpectedPoint` — ponto de origem com posição global conhecida independentemente do mapeamento.
- `ReferenceGeometry`/`GeometryReferenceRole` — nuvem com papel declarado; um `.pcd` nunca é referência pelo nome; sem alinhamento (`alignment: "none"`).
- `evaluate_round_trip()`/`RoundTripReport` — o mapa construído contra o reaberto do artefato.
- `compare_contractual_inventories()`/`ReproducibilityReport` — reprodutibilidade dos arquivos contratuais de dois runs.
- `compare_geometric_mapping_reports()`/`GeometricMappingComparison` — comparação controlada que só deixa o mapeamento variar.
- `encode_geometric_mapping_report()` — representação JSON com todas as identidades.

### Point Representation

- `RepresentationArm`/`RepresentationArmRole` — um braço da ablação (`off`, descritor determinístico, 3D aprendido, pré-treinado/destilado) com o encoder já construído.
- `evaluate_representation_arm()`/`RepresentationArmReport` — cobertura, repetibilidade, distribuição de normas, sensibilidade a variações controladas, custo e downstream em seções separadas.
- `GeometryVariation` e `translation_variation()`/`rotation_about_z_variation()`/`noise_variation()`/`subsample_variation()` — variações reproduzíveis da geometria, sem modificar a original.
- `compare_representation_arms()`/`RepresentationAblationReport` — comparação lado a lado que rejeita drift de mapa, centros e configuração downstream; sem score nem vencedor.
- `encode_representation_arm_report()`/`encode_representation_ablation_report()` — representação JSON com todas as identidades.

### Reference set

- `ReferenceSetManifest`/`ReferenceSetIdentity` — manifesto versionado e hasheado do reference set: fontes, calibrações, amostras ligadas a `SourceObservationId`, estratos, anotações, proveniência e splits.
- `ReferenceTrust` — trust declarado por arquivo de anotação (`trusted_ground_truth`, `approximate_annotation`, `derived_measurement`, `diagnostic_only`); nunca inferido do nome do arquivo. Anotações de origem `model_inference` só podem ser `diagnostic_only`.
- `encode_reference_set()`/`decode_reference_set()`/`write_reference_set()`/`read_reference_set()` — persistência imutável com digest verificado na leitura.
- `verify_annotation_files()` e `require_version_bump_on_change()` — hashes dos arquivos de anotação e regra de que a versão muda quando o conteúdo muda.

### Anotações de referência

- `AnnotationFamily` — as sete famílias versionadas (`regions`, `semantics`, `geometry`, `identity`, `relations`, `visibility`, `scene_context`) e seus schemas `contextmap.reference.<família>/v1`.
- `RegionAnnotationSet`, `SemanticAnnotationSet`, `GeometryAnnotationSet`, `IdentityAnnotationSet`, `RelationAnnotationSet`, `VisibilityAnnotationSet` e `SceneContextAnnotationSet` — o conteúdo de cada família; anotações parciais, ambiguidade e desconhecido são valores explícitos, e ausência nunca é verdade negativa.
- `LabelNormalization` — política de normalização explícita e versionada (`casefold-exact/1`, `casefold-alias/1`) sob a qual rótulos literais são comparados.
- `write_annotation_set()`/`read_annotation_set()`/`encode_annotation_set()`/`decode_annotation_set()` — persistência imutável e leitura despachada pelo schema.
- `ground_truth_regions()` e `SemanticAnnotationSet.to_semantic_annotation()` — entrega das anotações aos avaliadores de Region Discovery e Semantic Interpretation.

### Integridade do reference set

- `validate_reference_set()`/`ReferenceSetIntegrityReport` — relatório de blockers e warnings com identidade (`id`, `version`, `digest`), checagem opcional dos arquivos de anotação e auditoria de proveniência independente de saídas de modelo.
- `require_valid_reference_set()`/`open_validated_reference_set()`/`ValidatedReferenceSet` — entradas das ferramentas de avaliação: recusam por padrão um reference set com blockers e não aceitam um relatório inválido, de outro reference set ou sem a checagem dos arquivos.
- Política de split explícita por tarefa: unidade, justificativa, chaves de grupo e janela de adjacência; sobreposição, vazamento por unidade, observações compartilhadas e vizinhança temporal entre splits são blockers.

### Subconjunto de fixtures para CI

- `generate_ci_fixture_subset()`/`build_synthetic_sequence()` — gera, só com fórmulas, uma sequência sintética canônica (RGB, LiDAR, pose, calibração), o reference set do subconjunto e o catálogo; commitado em `tests/fixtures/ci_subset/<versão>/`.
- `FixtureCatalogue`/`FixtureCase`/`CoverageEntry` — casos com id estável, casos-limite, saídas esperadas, tolerâncias, proveniência, licença, redistribuição e hash; a matriz de cobertura registra explicitamente o que o subconjunto **não** cobre (fusão multi-vista e round-trip do `ContextMapArtifact`).
- O subconjunto protege contra regressões e não substitui a avaliação com dados reais. A regressão entre módulos que ele sustenta é parcial: a cadeia "da ingestão até o artifact final" (#172) depende de fusão multi-vista, do `ContextMapArtifact` e de Entity Resolution/Spatial Relations.

### Registro de métricas e relatório comum

- `MetricRegistry`/`MetricDefinition`/`default_metric_registry()` — definições versionadas e legíveis por máquina de todas as métricas por estágio: nome/versão, população, unidade/faixa, anotações exigidas, agregação, comportamento com dados ausentes, evaluator e direção. Não existe unidade `probability`: score de suporte arbitrário não é probabilidade calibrada.
- `EvaluationReport`/`ReproducibilityMetadata`/`MetricResult` — envelope comum com metadados de reprodutibilidade compartilhados, métricas de **qualidade** e de **performance** em campos separados e o relatório do estágio intocado; `not_applicable`/`unsupported` nunca viram zero e não há score geral.
- `assemble_evaluation_report()`/`decode_evaluation_report()` — validam o relatório contra o registro; `require_annotation_compatibility()` recusa versões de anotação incompatíveis.
- `region_discovery_evaluation_report()`/`semantic_interpretation_evaluation_report()`/`wrap_stage_report()` — levam os relatórios existentes para o envelope sem recalcular métricas.

### Experimentos e ablações

- `ExperimentManifest`/`ExperimentVariable`/`ExperimentArm`/`ResolvedTopology` — manifesto versionado e hasheado de uma comparação controlada: seleção exata de amostras do reference set, topologia resolvida por arm, artifacts upstream pinados, variáveis sob teste, controles fixos, métricas e política de captura de recursos. Só se constrói se **apenas as variáveis declaradas variam** e se o trecho variado consome artifacts imutáveis pinados.
- `ablation_cells()`/`AblationMode` — a matriz de ablação (`one_at_a_time` ou `full_factorial`), determinística.
- `validate_experiment_manifest()` — confere a seleção contra o reference set, recusa tuning no split held-out e valida as métricas contra o registro e as anotações disponíveis.
- `run_experiment()`/`ExperimentRun`/`ArmExecutor` — executa cada arm por um executor injetado e emite um run manifest e um relatório por arm mais um `ComparisonManifest`; arms indisponíveis ou com resultado inconsistente ficam explícitos e tornam a comparação incompleta (`require_complete_comparison()`), sem fallback e sem score geral.

### QA das anotações e reprodutibilidade

- `check_annotation_quality()`/`AnnotationQaReport`/`AnnotationQaPolicy` — verifica o conteúdo dos arquivos de anotação (tamanho de máscara/caixa e imagem, geometria 3D, consistência de identidade, existência de sujeito/objeto e simetria/inverso de relações, semântica de ambiguidade, duplicatas e conflitos) e embute o relatório de integridade. **Blockers**, warnings, observações permissíveis (ambiguidade, desconhecido, cobertura parcial) e divergências entre anotadores ficam separados.
- `DisagreementSummary` — divergência entre anotadores, visível e sem escolha silenciosa: mostra o que cada arquivo diz; não há campo de resolução.
- `certify_reference_set()`/`CertifiedReferenceSet` — exige integridade e QA sem blockers; a integridade é pré-requisito das execuções oficiais.
- `check_evaluator_reproducibility()`/`compare_evaluation_reports()`/`NondeterministicField` — rodam um evaluator repetidamente sobre as mesmas entradas e recusam qualquer diferença não declarada; valores de recursos são excluídos explicitamente e o não determinismo inevitável exige motivo.

### Avaliação de técnicas opcionais

Protocolo e harness apenas: não há execução real nem evidência, e nenhuma decisão foi tomada (#197 segue aberta para a execução e a decisão).

- `build_feature_resolution_protocol()`/`build_quality_aware_fusion_protocol()`/`TechniqueProtocol` — os dois experimentos controlados (features nativas × melhoradas; fusão uniforme × ciente de qualidade) como um manifesto por estágio avaliado, com o mesmo artifact upstream pinado nos dois arms, os estratos do protocolo e as métricas de qualidade e de custo separadas.
- `build_technique_evidence()`/`TechniqueEvidence`/`EffectPolicy` — efeito por métrica **e por estrato** (inclusive regressões escondidas por um ganho global), custos à parte, estágios não avaliados e arms indisponíveis explícitos, artifacts compartilhados e disponibilidade dos estratos; sem score geral.
- `record_technique_decision()`/`TechniqueDecision` — decisão humana (manter, adiar, propor mudar o default) presa ao digest da evidência; nunca altera configuração e exige re-validação E2E para mudar o default.

## Módulos consumidos

`contextmap.ingestion` para a identidade da observação física, `contextmap.visual_perception`, `contextmap.state_estimation`, `contextmap.geometric_mapping`, `contextmap.sensor_association`, `contextmap.point_representation` e `contextmap.semantic_fusion`, exclusivamente por suas APIs públicas.

## Módulos que consomem este

Experimentos, benchmarks e gates de regressão. A composition root planejada em
`contextmap.runtime` não precisará importar `evaluation` para executar o
pipeline principal.

## Onde estão os documentos detalhados

- [`feature_extraction.md`](feature_extraction.md) — protocolo, invariantes, fixtures determinísticas e trade-offs da avaliação de features.
- [`../../visual_perception/docs/feature-extraction.md`](../../visual_perception/docs/feature-extraction.md) — visão do core produtor que esta avaliação mede.
- [`region-discovery.md`](region-discovery.md) — referência, métricas geométricas, diagnósticos, custo e comparação controlada de Region Discovery.
- [`semantic-interpretation.md`](semantic-interpretation.md) — convenção
  open-vocabulary, ablações, qualidade, custo e falhas sem fusion.
- [`semantic_fusion.md`](semantic_fusion.md) — seções do relatório, anotações, estratificação, comparação controlada e ablações da avaliação de Semantic Fusion.
- [`semantic_mapping.md`](semantic_mapping.md) — as seis camadas de validação de Semantic Mapping, a linhagem do relatório e a robustez a upstream corrompido.
- [`entity_resolution.md`](entity_resolution.md) — camadas de checagem, métricas de identidade do registro, classes de falha por causa, vínculo explícito de ocorrências, mapa entidade resolvida -> identidade, ablação de canais e limites.
- [`sensor_association.md`](sensor_association.md) — estratificação, denominadores explícitos, caminhos de features, linhagem e comparação controlada da avaliação de Sensor Association.
- [`point_representation.md`](point_representation.md) — braços, seções do relatório, variações controladas, comparação sem score e medição de amostra.
- [`optional-techniques.md`](optional-techniques.md) — protocolos das técnicas opcionais, estratos, evidência por estrato, custos separados e decisão manter/adiar/mudar o default.
- [`annotation-qa.md`](annotation-qa.md) — verificações do conteúdo das anotações por família, ambiguidade permissível, divergência entre anotadores, certificação e reprodutibilidade dos evaluators.
- [`experiments.md`](experiments.md) — manifesto de experimento, regras de comparação controlada, ablações de backend/política/canais/DAG, execução por arm e manifesto de comparação.
- [`metrics.md`](metrics.md) — registro de métricas por estágio, envelope de relatório comum, validação contra o registro e adaptadores dos harnesses existentes.
- [`ci-fixtures.md`](ci-fixtures.md) — subconjunto determinístico de fixtures para CI: conteúdo, casos, matriz de cobertura (com lacunas explícitas), regressão entre módulos e regras de versionamento.
- [`reference-integrity.md`](reference-integrity.md) — catálogo de checagens (blockers e warnings), política de split, auditoria de proveniência e entradas que recusam reference sets inválidos.
- [`annotations.md`](annotations.md) — famílias de anotação, parcialidade e verdade negativa explícita, normalização open-vocabulary, identidade/relações e ligação com observações físicas.
- [`reference-set.md`](reference-set.md) — manifesto do reference set, regras de identidade, trust e proveniência, digest/versão e persistência.
- [`state_estimation.md`](state_estimation.md) — camadas do relatório, referência confiável, protocolo de comparação (associação, alinhamento, ATE, RPE), perfil de referência, limiares por perfil, o baseline `ExternalPose` e o relatório do FAST-LIO.
- [`geometric_mapping.md`](geometric_mapping.md) — camadas do relatório, concordância ponto-plano entre scans, referência sem alinhamento, reprodutibilidade, fixtures sintéticas e a execução real de referência.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e direção de dependências.
- [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) — imutabilidade e separação entre outputs, métricas e debug.
