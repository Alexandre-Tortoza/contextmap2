# Evaluation

## Responsabilidade

Medir qualidade, regressões e custo das capabilities do ContextMap2 sem alterar os resultados do pipeline. As implementações atuais cobrem Feature Extraction e Region Discovery (relatórios determinísticos sobre contratos públicos de `visual_perception`) State Estimation (relatórios sobre `Trajectory` e seus artifacts) e Geometric Mapping (validação de um `GeometricMapArtifact` persistido).

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

### State Estimation

- `evaluate_state_estimation()`/`StateEstimationEvaluationReport` — relatório comum a qualquer backend, com seções independentes: `StructuralReport`, `MotionReport`, `TransformTraceReport`, `AccuracyReport` e `CostReport`.
- `MotionThresholds` — limiares de movimento fornecidos pelo perfil de referência; não há valores padrão.
- `ReferenceTrajectory`/`ReferenceRole`/`ReferenceComparisonConfig`/`AlignmentMethod` — referência com papel declarado e protocolo de comparação explícito (associação, alinhamento, ATE, RPE).
- `trace_transform_chain()` — cadeia `T_reference_sensor(t)` reconstruível com erro numérico de composição e de round trip.
- `compare_state_estimation_reports()`/`StateEstimationComparison` — comparação controlada entre backends que rejeita drift de sequência, seleção, calibração, referência ou protocolo.
- `encode_state_estimation_report()` — representação JSON do relatório com todas as identidades.

### Geometric Mapping

- `evaluate_geometric_mapping()`/`GeometricMappingEvaluationReport` — relatório de um mapa persistido, com seções independentes: `GeometricMappingStructureReport`, `GeometricMappingTraceReport`, `ExpectedPointCheck`, `MapDensityReport`, `MapRangeReport`, `MapBoundsReport`, `ScanOverlapReport`, `ReferenceGeometryReport` e `GeometricMappingCostReport`.
- `GeometricMappingProtocol` — todas as decisões de amostragem e todos os limiares, fornecidos pelo perfil; sem valores padrão.
- `ExpectedPoint` — ponto de origem com posição global conhecida independentemente do mapeamento.
- `ReferenceGeometry`/`GeometryReferenceRole` — nuvem com papel declarado; um `.pcd` nunca é referência pelo nome; sem alinhamento (`alignment: "none"`).
- `evaluate_round_trip()`/`RoundTripReport` — o mapa construído contra o reaberto do artefato.
- `compare_contractual_inventories()`/`ReproducibilityReport` — reprodutibilidade dos arquivos contratuais de dois runs.
- `compare_geometric_mapping_reports()`/`GeometricMappingComparison` — comparação controlada que só deixa o mapeamento variar.
- `encode_geometric_mapping_report()` — representação JSON com todas as identidades.

## Módulos consumidos

`contextmap.ingestion` para a identidade da observação física, `contextmap.visual_perception`, `contextmap.state_estimation` e `contextmap.geometric_mapping`, exclusivamente por suas APIs públicas.

## Módulos que consomem este

Experimentos, benchmarks e gates de regressão. `runtime` não precisa importar `evaluation` para executar o pipeline principal.

## Onde estão os documentos detalhados

- [`feature_extraction.md`](feature_extraction.md) — protocolo, invariantes, fixtures determinísticas e trade-offs da avaliação de features.
- [`../../visual_perception/docs/feature-extraction.md`](../../visual_perception/docs/feature-extraction.md) — visão do core produtor que esta avaliação mede.
- [`region-discovery.md`](region-discovery.md) — referência, métricas geométricas, diagnósticos, custo e comparação controlada de Region Discovery.
- [`state_estimation.md`](state_estimation.md) — camadas do relatório, referência confiável, protocolo de comparação (associação, alinhamento, ATE, RPE), limiares por perfil e o baseline `ExternalPose`.
- [`geometric_mapping.md`](geometric_mapping.md) — camadas do relatório, concordância ponto-plano entre scans, referência sem alinhamento, reprodutibilidade, fixtures sintéticas e a execução real de referência.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e direção de dependências.
- [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) — imutabilidade e separação entre outputs, métricas e debug.
