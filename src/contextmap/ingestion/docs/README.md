# Ingestion

## Responsabilidade

Normalizar fontes registradas (ROS 1 bags, ROS 2 bags, datasets gravados) em observações de sensor canônicas, agnósticas de backend, que o restante do ContextMap2 pode consumir sem depender de mensagens ROS, APIs de bag ou estruturas específicas de dataset.

## O que este módulo explicitamente não possui

- pose canônica do mapa (`PoseEstimate`/trajetória pertencem a State Estimation);
- percepção semântica;
- projeção 2D↔3D;
- fusão ou entidades persistentes;
- armazenamento em nuvem/objeto remoto.

## Contratos públicos

- `SourceObservation` — union das observações canônicas: `ImageObservation`, `LidarObservation`, `ImuObservation`, `ExternalPoseMeasurement`.
- `SourceObservationId`, `SensorId`, `FrameId`, `CalibrationReferenceId` — identificadores tipados.
- `SourceProvenance` — rastreabilidade até a fonte bruta.
- `SequenceArtifactWriter`/`SequenceArtifactReader` — persistência local imutável de uma sequência ingerida; `SequenceArtifactManifest`, `SequenceArtifactFileEntry`, `SequenceArtifactId`.
- `SequenceArtifactError`, `IncompleteSequenceArtifactError` — exceções semânticas de leitura/escrita do artefato.
- `synchronize()` — agrupa observações em `ProcessingObservation`s auditáveis; `SynchronizationConfig`, `ModalityAssociation`, `SynchronizationDiagnostics`, `DroppedEvent`.
- `observation_modality()`, `MODALITY_NAMES` — utilitário para consumir `SourceObservation` de forma genérica por modalidade.
- `CalibrationSet`/`CalibrationEntry` — calibração e frames de coordenadas canônicos; `CameraModel` (`PinholeCameraModel`/`FisheyeCameraModel`), `RigidTransform`, `validate_calibration_set()`.
- `resolve_selection()` — lê um subconjunto determinístico de uma sequência; `SequenceSelection` (`FullSequenceSelection`/`FrameRangeSelection`/`TimestampRangeSelection`/`ExplicitIdsSelection`), `selection_identity()`.
- `SourceAdapter` — fronteira (`Protocol`) que qualquer adapter de fonte concreto implementa; `SourceAdapterConfig`, `SourceTopicMapping`, `SourceAdapterCapabilities`, `SourceAdapterWarning`.
- `contextmap.ingestion.adapters.ros1_bag.Ros1BagSourceAdapter` — implementação concreta para ROS 1 (não reexportada por `contextmap.ingestion`; importada pelo path completo, como qualquer backend).

Ver [`contracts.md`](contracts.md) para a referência completa de campos, unidades e exemplos de mapeamento ROS 1/ROS 2, [`artifact.md`](artifact.md) para o formato do artefato persistido e o layout do workspace local, [`synchronization.md`](synchronization.md) para a política de sincronização e suas limitações conhecidas, [`calibration.md`](calibration.md) para o contrato de calibração e convenções de frame, [`selection.md`](selection.md) para o modelo de seleção e replay, [`adapters.md`](adapters.md) para a fronteira de source adapters, e [`backends.md`](backends.md) para decisões específicas de cada adapter concreto.

## Módulos consumidos

Apenas `contextmap.shared` (`SourceTimestamp`).

## Módulos que consomem este

`visual_perception`, `state_estimation`, e demais capabilities a jusante, sempre através de `contextmap.ingestion` (nunca de `contextmap.ingestion.models` diretamente).

## Fluxo interno (alto nível)

Fonte bruta → adapter → eventos normalizados → índice temporal/sincronização → `SourceObservation` canônica → artefato de sequência persistido (`SequenceArtifactWriter`). Sincronização/agrupamento temporal, calibração completa, seleção/replay e adapters de fonte são issues separadas desta milestone — ver [`docs/PIPELINE.md`](../../../../docs/PIPELINE.md#1-ingestion).

## Onde estão os documentos detalhados

- [`contracts.md`](contracts.md) — campos, unidades, ownership, exemplos ROS 1/ROS 2.
- [`artifact.md`](artifact.md) — formato do artefato de sequência persistido e layout do workspace local.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e regras de dependência.
- [`docs/CONTRACTS.md`](../../../../docs/CONTRACTS.md) — `SourceObservation` no contexto global de contratos.
