# Adapters concretos

Este documento complementa [`adapters.md`](adapters.md) com decisões específicas de cada implementação concreta de `SourceAdapter`.

## `Ros1BagSourceAdapter` (issue #44)

`src/contextmap/ingestion/adapters/ros1_bag.py`.

### Dependência opcional

Usa [`rosbags`](https://pypi.org/project/rosbags/) — biblioteca Python pura que lê/escreve bags ROS 1 e ROS 2 **sem precisar de uma instalação ROS**. Instalável via `pip install contextmap[ros1]`; declarada em `pyproject.toml` como extra, nunca como dependência obrigatória do pacote, para que quem só lê `SequenceArtifact`s já persistidos não precise dela. `tests/architecture/test_boundaries.py` inclui `"rosbags"` em `HEAVY_SDK_ROOTS`, então importá-la fora de `adapters/` é um erro de arquitetura detectado por CI.

### Escopo v0

Decodifica `sensor_msgs/Image` (RGB), `sensor_msgs/PointCloud2` (LiDAR), `sensor_msgs/Imu`, `nav_msgs/Odometry` (pose externa) e `sensor_msgs/CameraInfo` (calibração, via `read_calibration()` — método adicional, fora do `Protocol SourceAdapter`, já que calibração não é uma `SourceObservation`).

**Fora do escopo v0**: TF estático/dinâmico (`tf2_msgs/TFMessage`) não é decodificado — `CalibrationSet.static_transforms` sempre vem vazio deste adapter. Extrínsecos precisando de TF ficam para trabalho futuro; a limitação está documentada aqui, não escondida.

### Decisões de mapeamento

- **Identidade**: `observation_id = "<topico-sanitizado>-<indice:06d>"` (índice por tópico, 0-based); `sensor_id` deriva do próprio nome do tópico sanitizado (`/camera/image_raw` → `camera_image_raw`) — o v0 não tem configuração separada de nome de sensor.
- **Timestamp**: sempre `header.stamp` da própria mensagem (não o timestamp de gravação do bag), com `clock_id = f"ros1_bag:{topic}"` — mesma convenção documentada em [`contracts.md`](contracts.md).
- **IMU**: cada um de `linear_acceleration`, `angular_velocity`, `orientation` é `None` quando o respectivo `*_covariance[0] == -1.0` — convenção padrão do `sensor_msgs/Imu` para "não fornecido", verificada independentemente por campo.
- **Pose externa**: `Odometry.header.frame_id` → `ExternalPoseMeasurement.parent_frame`; `Odometry.child_frame_id` → `ExternalPoseMeasurement.frame_id` (herdado da base) — mesma convenção de `docs/CONTRACTS.md`/issue #38.
- **Calibração**: usa a **última** mensagem do tópico `camera_info` (intrínsecos tratados como estáticos para o bag inteiro no v0); `distortion_model == "equidistant"` vira `FisheyeCameraModel`, qualquer outro valor reconhecido (`plumb_bob`, `rational_polynomial`) vira `PinholeCameraModel` com o `DistortionModel` correspondente; um `distortion_model` desconhecido cai em `DistortionModel.NONE` com os coeficientes brutos preservados — validação (issue #41) detecta a inconsistência depois, em vez do decoder falhar silenciosamente ou travar a leitura.

### Erros vs. warnings

Tópico obrigatório (`required_topics`) ausente do bag → `MissingRequiredTopicError`, antes de qualquer observação ser produzida. Mensagem individual malformada ou com encoding/tipo de campo não suportado (`KeyError`/`ValueError`/`AttributeError` durante a decodificação) → registrada via `SourceAdapterWarning` e pulada; o restante da leitura continua.

### Fixtures de teste

Os testes geram um bag ROS 1 sintético e determinístico em tempo de execução com `rosbags.rosbag1.Writer`, em vez de versionar um arquivo binário `.bag` no repositório — evita problemas de tamanho/licença e mantém a fixture auditável como código Python (`tests/ingestion/adapters/test_ros1_bag.py`).
