# Contratos de observação canônica

Este documento descreve os tipos definidos em `src/contextmap/ingestion/models.py`: unidades, campos obrigatórios/opcionais, ownership e exemplos de mapeamento a partir de mensagens ROS 1/ROS 2 representativas.

Este é o contrato **em memória**. O formato persistido em disco (artefato de sequência canônica) é definido pela issue de artefato/workspace desta milestone; ver [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md).

## Princípio de modalidade

Cada modalidade física (imagem, LiDAR, IMU, pose externa) é um tipo próprio, não um único registro com um campo por modalidade possível. Uma modalidade ausente é representada pela ausência de uma instância daquele tipo, nunca por um campo zerado. Um subcampo opcional dentro de uma modalidade (ex.: orientação de IMU) é `None` quando a fonte não o fornece — nunca um valor sentinela como quaternion identidade ou zero.

## Campos comuns (`_SourceObservationBase`)

| Campo | Tipo | Unidade/convenção |
| --- | --- | --- |
| `observation_id` | `SourceObservationId` | Identidade estável da observação física; não muda entre reprocessamentos. |
| `sensor_id` | `SensorId` | Identidade do sensor físico. |
| `frame_id` | `FrameId` | Frame de coordenadas da observação. Convenções completas de frame ficam a cargo do contrato de calibração. |
| `timestamp` | `SourceTimestamp` (`contextmap.shared`) | Segundos + nanossegundos inteiros + `clock_id` explícito. Dois timestamps só são comparáveis com o mesmo `clock_id`. |
| `provenance` | `SourceProvenance` | Rastreabilidade até a fonte bruta. |
| `calibration_id` | `CalibrationReferenceId \| None` | Referência à calibração aplicável; `None` quando ainda não associada. |

## `ImageObservation`

| Campo | Tipo | Unidade/convenção |
| --- | --- | --- |
| `width`, `height` | `int` | pixels |
| `encoding` | `ImageEncoding` | `rgb8`, `bgr8`, `mono8`, `mono16` (nomes ROS-style) |
| `data` | `bytes` | Buffer row-major canônico, sem padding por linha, consistente com `width`/`height`/`encoding`; `mono16` usa ordem little-endian. |

## `LidarObservation`

| Campo | Tipo | Unidade/convenção |
| --- | --- | --- |
| `point_count` | `int` | número de registros de ponto |
| `point_step_bytes` | `int` | bytes por registro |
| `fields` | `Sequence[PointFieldDescriptor]` | layout nomeado (nome, offset, tipo, count) — equivalente agnóstico de `PointCloud2.fields` |
| `is_dense` | `bool` | `False` indica possíveis pontos inválidos/NaN |
| `data` | `bytes` | buffer row-major sem padding por linha, com `point_count` registros de `point_step_bytes`; campos numéricos multibyte usam ordem little-endian |

## `ImuObservation`

| Campo | Tipo | Unidade/convenção |
| --- | --- | --- |
| `linear_acceleration` | `tuple[float, float, float] \| None` | m/s², em `frame_id` |
| `angular_velocity` | `tuple[float, float, float] \| None` | rad/s, em `frame_id` |
| `orientation` | `tuple[float, float, float, float] \| None` | quaternion unitário (x, y, z, w) |
| `linear_acceleration_covariance` | `tuple[float, ...] \| None` | matriz 3×3 row-major associada à aceleração |
| `angular_velocity_covariance` | `tuple[float, ...] \| None` | matriz 3×3 row-major associada à velocidade angular |
| `orientation_covariance` | `tuple[float, ...] \| None` | matriz 3×3 row-major associada à orientação |

## `ExternalPoseMeasurement`

**Regra de ownership**: esta é uma medição de entrada, nunca o `PoseEstimate` canônico do mapa — trajetória e pose canônica pertencem a State Estimation.

| Campo | Tipo | Unidade/convenção |
| --- | --- | --- |
| `parent_frame` | `FrameId` | frame de referência, ex.: `"odom"` |
| `frame_id` (herdado) | `FrameId` | frame filho cuja pose é reportada, ex.: `"base_link"` |
| `translation` | `tuple[float, float, float]` | metros, posição de `frame_id` em `parent_frame` |
| `orientation` | `tuple[float, float, float, float]` | quaternion unitário (x, y, z, w) de `frame_id` em `parent_frame` |
| `pose_covariance` | `tuple[float, ...] \| None` | matriz 6×6 row-major da pose |
| `linear_velocity` | `tuple[float, float, float] \| None` | m/s, expressa conforme a semântica da fonte |
| `angular_velocity` | `tuple[float, float, float] \| None` | rad/s, expressa conforme a semântica da fonte |
| `twist_covariance` | `tuple[float, ...] \| None` | matriz 6×6 row-major do twist |

## Exemplos de mapeamento

### ROS 1 → canônico

```text
sensor_msgs/Image (tópico /camera/image_raw)
    header.stamp        → timestamp (SourceTimestamp, clock_id compartilhado pelo adapter)
    timestamp do bag    → provenance.raw_metadata["bag_timestamp_nanoseconds"]
    header.frame_id      → frame_id
    width, height         → width, height
    encoding               → encoding (ImageEncoding("rgb8"))
    step, is_bigendian, data → data canônico (sem padding; mono16 little-endian)
    (bag path + índice)  → provenance (source_type="ros1_bag", source_topic="/camera/image_raw")

sensor_msgs/PointCloud2 (tópico /velodyne_points)
    header.stamp        → timestamp
    header.frame_id      → frame_id
    width * height         → point_count
    point_step               → point_step_bytes
    fields[]                   → fields (PointFieldDescriptor por entrada)
    is_dense                  → is_dense
    row_step, is_bigendian, data → data canônico (sem padding de linha; campos little-endian)

sensor_msgs/Imu (tópico /imu/data)
    header.stamp                        → timestamp
    header.frame_id                      → frame_id
    linear_acceleration (x,y,z)   → linear_acceleration
    angular_velocity (x,y,z)         → angular_velocity
    orientation (x,y,z,w)               → orientation (None se covariância de orientação for -1, indicando "não fornecido")
    *_covariance                            → covariâncias correspondentes (None quando [0] == -1)

nav_msgs/Odometry (tópico /odom)
    header.stamp                                → timestamp
    header.frame_id                              → parent_frame
    child_frame_id                                → frame_id
    pose.pose.position (x,y,z)             → translation
    pose.pose.orientation (x,y,z,w)     → orientation
    pose.covariance                              → pose_covariance
    twist.twist.linear/angular                 → linear_velocity/angular_velocity
    twist.covariance                            → twist_covariance
```

### ROS 2 → canônico

O mapeamento de campos é idêntico ao ROS 1 (os tipos `sensor_msgs`/`nav_msgs` mantêm o mesmo layout lógico); a diferença fica no adapter de origem (`ros2_bag` em vez de `ros1_bag`) e em `clock_id`/`source_type` na provenance:

```text
sensor_msgs/msg/Image (tópico /camera/color/image)
    → mesmo mapeamento do ROS 1, com
      provenance.source_type = "ros2_bag"
      timestamp.clock_id = identidade compartilhada do clock de header do adapter
```

Observações cujo `sensor_id` aparece em um `CalibrationSet` recebem o `calibration_id` correspondente. A associação é explícita; calibração não vira estado global implícito.
