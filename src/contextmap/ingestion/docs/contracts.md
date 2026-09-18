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
| `data` | `bytes` | Buffer row-major, sem padding por linha, consistente com `width`/`height`/`encoding`. |

## `LidarObservation`

| Campo | Tipo | Unidade/convenção |
| --- | --- | --- |
| `point_count` | `int` | número de registros de ponto |
| `point_step_bytes` | `int` | bytes por registro |
| `fields` | `Sequence[PointFieldDescriptor]` | layout nomeado (nome, offset, tipo, count) — equivalente agnóstico de `PointCloud2.fields` |
| `is_dense` | `bool` | `False` indica possíveis pontos inválidos/NaN |
| `data` | `bytes` | buffer row-major de `point_count` registros de `point_step_bytes` bytes |

## `ImuObservation`

| Campo | Tipo | Unidade/convenção |
| --- | --- | --- |
| `linear_acceleration` | `tuple[float, float, float] \| None` | m/s², em `frame_id` |
| `angular_velocity` | `tuple[float, float, float] \| None` | rad/s, em `frame_id` |
| `orientation` | `tuple[float, float, float, float] \| None` | quaternion unitário (x, y, z, w) |

## `ExternalPoseMeasurement`

**Regra de ownership**: esta é uma medição de entrada, nunca o `PoseEstimate` canônico do mapa — trajetória e pose canônica pertencem a State Estimation.

| Campo | Tipo | Unidade/convenção |
| --- | --- | --- |
| `parent_frame` | `FrameId` | frame de referência, ex.: `"odom"` |
| `frame_id` (herdado) | `FrameId` | frame filho cuja pose é reportada, ex.: `"base_link"` |
| `translation` | `tuple[float, float, float]` | metros, posição de `frame_id` em `parent_frame` |
| `orientation` | `tuple[float, float, float, float]` | quaternion unitário (x, y, z, w) de `frame_id` em `parent_frame` |

## Exemplos de mapeamento

### ROS 1 → canônico

```text
sensor_msgs/Image (tópico /camera/image_raw)
    header.stamp        → timestamp (SourceTimestamp, clock_id="ros1_bag:/camera/image_raw")
    header.frame_id      → frame_id
    width, height         → width, height
    encoding               → encoding (ImageEncoding("rgb8"))
    data                     → data (bytes, já row-major em sensor_msgs/Image)
    (bag path + índice)  → provenance (source_type="ros1_bag", source_topic="/camera/image_raw")

sensor_msgs/PointCloud2 (tópico /velodyne_points)
    header.stamp        → timestamp
    header.frame_id      → frame_id
    width * height         → point_count
    point_step               → point_step_bytes
    fields[]                   → fields (PointFieldDescriptor por entrada)
    is_dense                  → is_dense
    data                        → data

sensor_msgs/Imu (tópico /imu/data)
    header.stamp                        → timestamp
    header.frame_id                      → frame_id
    linear_acceleration (x,y,z)   → linear_acceleration
    angular_velocity (x,y,z)         → angular_velocity
    orientation (x,y,z,w)               → orientation (None se covariância de orientação for -1, indicando "não fornecido")

nav_msgs/Odometry (tópico /odom)
    header.stamp                                → timestamp
    header.frame_id                              → parent_frame
    child_frame_id                                → frame_id
    pose.pose.position (x,y,z)             → translation
    pose.pose.orientation (x,y,z,w)     → orientation
```

### ROS 2 → canônico

O mapeamento de campos é idêntico ao ROS 1 (os tipos `sensor_msgs`/`nav_msgs` mantêm o mesmo layout lógico); a diferença fica no adapter de origem (`ros2_bag` em vez de `ros1_bag`) e em `clock_id`/`source_type` na provenance:

```text
sensor_msgs/msg/Image (tópico /camera/color/image)
    → mesmo mapeamento do ROS 1, com
      provenance.source_type = "ros2_bag"
      timestamp.clock_id = "ros2_bag:/camera/color/image"
```

Os adapters concretos que realizam essa conversão (lendo o bag e construindo estes tipos) são escopo de issues específicas desta milestone (source adapter boundary, ROS 1 adapter, ROS 2 adapter) — este documento descreve apenas o mapeamento semântico de campos, não a implementação do decoder.
