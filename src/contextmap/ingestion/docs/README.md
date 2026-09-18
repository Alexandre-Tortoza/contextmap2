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

Ver [`contracts.md`](contracts.md) para a referência completa de campos, unidades e exemplos de mapeamento ROS 1/ROS 2.

## Módulos consumidos

Apenas `contextmap.shared` (`SourceTimestamp`).

## Módulos que consomem este

`visual_perception`, `state_estimation`, e demais capabilities a jusante, sempre através de `contextmap.ingestion` (nunca de `contextmap.ingestion.models` diretamente).

## Fluxo interno (alto nível)

Fonte bruta → adapter → eventos normalizados → índice temporal/sincronização → `SourceObservation` canônica → artefato de sequência persistido. Este módulo (issue #38) define apenas o contrato de observação em memória; sincronização, persistência, calibração completa, seleção/replay e adapters são issues separadas desta milestone — ver [`docs/PIPELINE.md`](../../../../docs/PIPELINE.md#1-ingestion).

## Onde estão os documentos detalhados

- [`contracts.md`](contracts.md) — campos, unidades, ownership, exemplos ROS 1/ROS 2.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e regras de dependência.
- [`docs/CONTRACTS.md`](../../../../docs/CONTRACTS.md) — `SourceObservation` no contexto global de contratos.
