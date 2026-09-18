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

Ver [`contracts.md`](contracts.md) para a referência completa de campos e a regra central de ownership (observação física vs. resultado de inferência), e [`ports.md`](ports.md) para os pontos de substituição de backend.

## Módulos consumidos

`contextmap.shared` e `contextmap.ingestion` (`SourceObservationId`, e futuramente sequência/seleção para orquestração).

## Módulos que consomem este

`sensor_association`, `state_estimation` (indiretamente via pose), `semantic_fusion` e demais capabilities a jusante, sempre através de `contextmap.visual_perception` (nunca de `contextmap.visual_perception.models` diretamente).

## Onde estão os documentos detalhados

- [`contracts.md`](contracts.md) — campos, ownership, exemplos.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e regras de dependência.
- [`docs/CONTRACTS.md`](../../../../docs/CONTRACTS.md) — `PerceptionResult` no contexto global de contratos.
