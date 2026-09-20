# State Estimation

## Responsabilidade

Produzir a **pose dinâmica** do rig ao longo do tempo: onde o frame do corpo (`body`) está no frame do mapa (`map`) em cada instante. O resultado é publicado como contratos canônicos (`PoseEstimate`, `Trajectory`) que Geometric Mapping e Sensor Association consomem sem conhecer ROS, o estimador que produziu a pose ou o formato de pose de um dataset.

```mermaid
flowchart LR
    EXT["ExternalPoseMeasurement<br/>(Ingestion)"] --> BE["Backend de estimação"]
    LIDAR["LiDAR + IMU<br/>(Ingestion)"] --> BE
    BE --> POSE["PoseEstimate[]"]
    POSE --> TRAJ["Trajectory"]
    TRAJ --> DOWN["Geometric Mapping /<br/>Sensor Association"]
```

Uma medição de pose vinda da fonte é apenas entrada. Ela só se torna `PoseEstimate` depois que um backend a valida e publica com frames, unidades e provenance explícitos.

## O que este módulo explicitamente não possui

- calibração estática (extrínsecos, intrínsecos, identidade de frames): pertence a Ingestion; State Estimation apenas consome o subconjunto de que o frame graph precisa;
- projeção 3D→2D e associação com imagens: Sensor Association;
- geometria persistente e mapa global: Geometric Mapping;
- qualquer conceito semântico.

## Estado implementado

Somente os contratos canônicos existem neste momento. Estão **planejados**, e serão documentados aqui quando forem implementados: port `StateEstimator` e backends (`ExternalPose`, FAST-LIO), lookup temporal/interpolação, preflight do frame graph, `StateEstimationRunArtifact` e o harness de validação.

## Contratos públicos

- `PoseEstimate`, `PoseEstimateId`, `PoseValidity`, `PoseProvenance` — pose `T_parent_child` em um instante, com frames, unidades, validade e provenance explícitos.
- `Trajectory`, `TrajectoryId`, `TrajectoryGap`, `TrajectoryProvenance`, `EstimatorProvenance` — sequência ordenada de poses de um corpo em um frame de referência.
- `TimeBounds`, `TrajectoryQualitySummary` — resumos derivados de uma trajetória.
- `pose_estimate_id_for()` — identidade determinística de uma pose dentro da trajetória.

Ver [`contracts.md`](contracts.md) para a referência de campos, a convenção de transform e as invariantes.

## Módulos consumidos

- `contextmap.ingestion`: `FrameId`, `SourceObservationId`, `SequenceArtifactId`, `Covariance6x6`.
- `contextmap.shared`: `SourceTimestamp`, `Vector3`, `Quaternion`, `is_unit_quaternion`.

## Módulos que consomem este

`geometric_mapping` e `sensor_association`, sempre através de `contextmap.state_estimation`.

## Onde estão os documentos detalhados

- [`contracts.md`](contracts.md) — contratos, convenção de transform, invariantes e serialização.
- [`docs/architecture.md`](../../../../docs/architecture.md) — ownership e direção de dependências.
- [`docs/CONTRACTS.md`](../../../../docs/CONTRACTS.md) — `PoseEstimate` e `Trajectory` no contexto global de contratos.
- [`docs/shared-primitives.md`](../../../../docs/shared-primitives.md) — primitivas geométricas compartilhadas.
