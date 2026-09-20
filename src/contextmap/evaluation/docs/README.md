# Evaluation

## Responsabilidade

Medir qualidade, regressões e custo das capabilities do ContextMap2 sem alterar
os resultados do pipeline. As implementações atuais cobrem Feature Extraction,
Region Discovery e Semantic Interpretation por meio de relatórios
determinísticos sobre contratos públicos de `visual_perception`.

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
- `SemanticEvaluationReport` — qualidade, custo e falhas em blocos separados.
- `evaluate_semantic_interpretation()` — avaliação por request/evidence variant
  com matching policy versionada.
- `compare_semantic_backends()` — exige exatamente os mesmos requests e
  variants para Qwen, Gemini e Florence-2.

## Módulos consumidos

`contextmap.ingestion` para a identidade da observação física e `contextmap.visual_perception` exclusivamente por sua API pública.

## Módulos que consomem este

Experimentos, benchmarks e gates de regressão. `runtime` não precisa importar `evaluation` para executar o pipeline principal.

## Onde estão os documentos detalhados

- [`feature_extraction.md`](feature_extraction.md) — protocolo, invariantes, fixtures determinísticas e trade-offs da avaliação de features.
- [`../../visual_perception/docs/feature-extraction.md`](../../visual_perception/docs/feature-extraction.md) — visão do core produtor que esta avaliação mede.
- [`region-discovery.md`](region-discovery.md) — referência, métricas geométricas, diagnósticos, custo e comparação controlada de Region Discovery.
- [`semantic-interpretation.md`](semantic-interpretation.md) — convenção
  open-vocabulary, ablações, qualidade, custo e falhas sem fusion.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e direção de dependências.
- [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) — imutabilidade e separação entre outputs, métricas e debug.
