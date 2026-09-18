# Visual Perception

## Responsabilidade

Transformar uma `SourceObservation` física (Ingestion) em evidência visual canônica e agnóstica de backend — regiões, features, claims semânticas — para uma execução configurada (`PerceptionRun`), sem decidir identidade persistente de entidade 3D, significado semântico final, ou projeção 2D↔3D.

## O que este módulo explicitamente não possui

- identidade persistente de entidade 3D (pertence a Entity Resolution/Semantic Mapping);
- projeção 2D↔3D (pertence a Sensor Association);
- fusão multi-view ou acumulação de confiança (pertence a Semantic Fusion);
- pose canônica do mapa (pertence a State Estimation).

## Contratos públicos

- `PerceptionRun`/`PerceptionResult` — execução configurada e seu resultado para uma `SourceObservation`.
- `Region2D`, `VisualFeature`, `SemanticClaim`, `SceneContext` — evidência visual canônica.
- `BackendProvenance` — rastreabilidade até o backend que produziu uma evidência.
- `RegionId`/`FeatureId`/`ClaimId` — identidades **locais a um `PerceptionResult`**, nunca identidade persistente de entidade nem comparável entre resultados diferentes sem associação explícita posterior.

- `RegionDiscovery`, `FeatureExtractor`, `SemanticInterpreter`, `SemanticScorer` — ports (`Protocol`) que qualquer backend concreto implementa; `PreparedImage`, `SemanticSupport`.

- `execute_stage_graph()`/`assemble_perception_result()` — executor de grafo de estágios e montagem de `PerceptionResult`; `StageDefinition`, `StageOutcome`, `StageStatus`, `StageGraphError`.

- `PipelinePreset`, `StageSpec`, `CANONICAL_PRESET_V1`, `KNOWN_CAPABILITIES` — presets de pipeline versionados e o grafo de estágios declarativo; `resolve_pipeline()`/`ResolvedPipeline`, `validate_pipeline_preset()`, `encode_pipeline_preset()`/`decode_pipeline_preset()`, `PipelineConfigError`, `StageBackendFactory`.

- `EmbeddingSpace` — o que `VisualFeature.embedding_space_id` identifica; `embedding_space_fingerprint()`, `ensure_compatible_embedding_spaces()`/`ensure_compatible_features()`, `EmbeddingSpaceMismatchError`, `encode_embedding_space()`/`decode_embedding_space()`.
- `DenseFeatureMap`/`DenseFeatureSampling` — geometria explícita entre o payload denso e a imagem preparada; `map_box_to_grid_cells()`/`pool_region_feature()` fazem associação e pooling mask-aware determinísticos com `RegionPoolingDiagnostics` e proveniência auditável.

- `perception_result_id_for()`, `region_id_for()`, `feature_id_for()`, `claim_id_for()` — geradores de identidade determinística.

- `PerceptionRunWriter`/`PerceptionRunReader` — persistência local imutável de um run de percepção; `RunArtifactManifest`, `allocate_run_index()`, `rebuild_run_registry()`.
- `FeatureStoreWriter`/`FeatureStoreReader` — persistência, indexação e carregamento sob demanda do payload numérico de um `VisualFeature` (`PerceptionRunWriter.add_feature_payload()`/`PerceptionRunReader.feature_store()`); `FeaturePayloadEntry`, `FeatureStoreError`, `FeaturePayloadIntegrityError`.
- `FeatureExtractionDiagnostic`/`FeatureDebugLevel` — métricas comuns e debug auditável para features densas, globais e de região; integração por `PerceptionRunWriter.add_feature_diagnostic()`/`add_feature_preview()`.
- `encode_perception_result()`/`decode_perception_result()` (e equivalentes por tipo) — serialização JSON dos contratos públicos.

- `PerceptionEvidenceSet` — view de leitura sobre múltiplos runs selecionados explicitamente; `ObservationEvidence`, `EvidenceSetError`.

Ver [`contracts.md`](contracts.md) para a referência completa de campos e a regra central de ownership (observação física vs. resultado de inferência), [`ports.md`](ports.md) para os pontos de substituição de backend, [`service.md`](service.md) para a execução do grafo de estágios e a política de isolamento de falhas, [`pipeline.md`](pipeline.md) para os presets versionados e o grafo declarativo, [`embedding_space.md`](embedding_space.md) para a identidade de espaço de embedding e a regra de compatibilidade, [`dense_region_association.md`](dense_region_association.md) para a transformação espacial e o pooling mask-aware, [`identity.md`](identity.md) para a cadeia completa de rastreabilidade, [`run_artifact.md`](run_artifact.md) para o formato do artefato de run persistido, [`feature_store.md`](feature_store.md) para a persistência/carregamento sob demanda do payload numérico de uma feature, [`feature_diagnostics.md`](feature_diagnostics.md) para métricas/debug auditáveis, e [`evidence_set.md`](evidence_set.md) para a view de evidência multi-run.

## Testes de contrato ponta a ponta

`tests/visual_perception/fakes.py` reúne implementações fake determinísticas de cada port (`FakeRegionDiscovery`, `FakeDenseFeatureExtractor`, `FakeRegionFeatureExtractor`, `FakeSemanticInterpreter`, `FakeFailingRegionDiscovery`) — sem GPU, download de modelo, ou dependência de rede — usadas por `tests/visual_perception/test_end_to_end.py` para validar o caminho completo: observação canônica → `PreparedImage` → grafo de estágios → `PerceptionResult` → `PerceptionRunArtifact` → `PerceptionRunReader` isolado → `PerceptionEvidenceSet` multi-run. `tests/visual_perception/test_pipeline.py` reutiliza os mesmos fakes para validar `pipeline.py` isoladamente: resolução única de backends, validação antes do carregamento de backend, rejeição de preset inválido/cíclico, inserção de estágio opcional sem tocar código de capability, e o `configuration_digest` determinístico.

## Módulos consumidos

`contextmap.shared` e `contextmap.ingestion` (`SourceObservationId`, e futuramente sequência/seleção para orquestração).

## Módulos que consomem este

`sensor_association`, `state_estimation` (indiretamente via pose), `semantic_fusion` e demais capabilities a jusante, sempre através de `contextmap.visual_perception` (nunca de `contextmap.visual_perception.models` diretamente).

## Onde estão os documentos detalhados

- [`contracts.md`](contracts.md) — campos, ownership, exemplos.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e regras de dependência.
- [`docs/CONTRACTS.md`](../../../../docs/CONTRACTS.md) — `PerceptionResult` no contexto global de contratos.
