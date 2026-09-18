# Evaluation

## Responsabilidade

Medir qualidade, regressões e custo das capabilities do ContextMap2 sem alterar os resultados do pipeline. A primeira implementação concreta cobre Feature Extraction por meio de relatórios determinísticos sobre contratos públicos de `visual_perception`.

## O que este módulo explicitamente não possui

- extração de features, pooling ou transformação espacial;
- seleção/composição de backends;
- mutação de `PerceptionRunArtifact` ou de qualquer artifact avaliado;
- um score global que misture qualidade, runtime, memória e tamanho de payload;
- regras de compatibilidade de embedding — consome a regra pública de `visual_perception`.

## Contratos públicos

- `FeatureEvaluationContext` — identidades mantidas fixas em uma comparação controlada: observação, conteúdo da imagem preparada, artifact/modelo de origem, conjunto de regiões e configuração downstream.
- `FeatureEvaluationReport` — relatório com blocos independentes `FeatureNumericalReport`, `FeatureSpatialReport` e `FeatureCostReport`.
- `FeatureResolutionComparisonReport` — agrupamento auditável de mapas densos native e enhanced compatíveis, sem colapsar métricas em um score único.
- `evaluate_feature_payload()` — valida shape, dtype, finitude, dimensão, normalização declarada e geometria densa antes de construir o relatório.
- `assert_repeatable_feature_outputs()` — verifica metadata e valores repetíveis sob o mesmo contexto, ignorando custo e identidades locais de runs imutáveis.
- `compare_feature_map_resolutions()` — exige contexto controlado, mesma imagem e mesmo `EmbeddingSpace` para comparar resoluções.
- `encode_feature_evaluation_report()`/`encode_feature_resolution_comparison_report()` — representação JSON-compatible para artifacts de avaliação futuros.

## Módulos consumidos

`contextmap.ingestion` para a identidade da observação física e `contextmap.visual_perception` exclusivamente por sua API pública.

## Módulos que consomem este

Experimentos, benchmarks e gates de regressão. `runtime` não precisa importar `evaluation` para executar o pipeline principal.

## Onde estão os documentos detalhados

- [`feature_extraction.md`](feature_extraction.md) — protocolo, invariantes, fixtures determinísticas e trade-offs da avaliação de features.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e direção de dependências.
- [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) — imutabilidade e separação entre outputs, métricas e debug.
