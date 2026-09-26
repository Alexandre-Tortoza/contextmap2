# Adapters concretos

Este documento complementa [`adapters.md`](adapters.md) com decisões específicas de cada implementação concreta de `SourceAdapter`.

## Lógica compartilhada entre ROS 1 e ROS 2

`src/contextmap/ingestion/adapters/_ros_common.py` (privado ao pacote `adapters/`) contém a decodificação de mensagem que é **idêntica** entre ROS 1 e ROS 2 — `rosbags` normaliza ambas as distribuições para os mesmos nomes de campo em `sensor_msgs/Image`, `PointCloud2`, `Imu` e `nav_msgs/Odometry`. A única diferença real entre as duas versões nas mensagens usadas aqui é o casing de `sensor_msgs/CameraInfo` (`D`/`K` maiúsculo no ROS 1, `d`/`k` minúsculo no ROS 2), tratada localmente em cada adapter via `_ros_common.build_camera_model()`, que recebe os valores já extraídos como primitivos.

## `Ros1BagSourceAdapter` (issue #44)

`src/contextmap/ingestion/adapters/ros1_bag.py`.

### Dependência opcional

Usa [`rosbags`](https://pypi.org/project/rosbags/) — biblioteca Python pura que lê/escreve bags ROS 1 e ROS 2 **sem precisar de uma instalação ROS**. Instalável via `pip install contextmap[ros1]`; declarada em `pyproject.toml` como extra, nunca como dependência obrigatória do pacote, para que quem só lê `SequenceArtifact`s já persistidos não precise dela. `tests/architecture/test_boundaries.py` inclui `"rosbags"` em `HEAVY_SDK_ROOTS`, então importá-la fora de `adapters/` é um erro de arquitetura detectado por CI.

### Escopo v0

Decodifica `sensor_msgs/Image` (RGB), `sensor_msgs/PointCloud2` (LiDAR), `sensor_msgs/Imu`, `nav_msgs/Odometry` (pose externa) e `sensor_msgs/CameraInfo` (calibração, via `read_calibration()`, parte do `Protocol SourceAdapter`).

**Fora do escopo v0**: TF estático/dinâmico (`tf2_msgs/TFMessage`) não é decodificado. Extrínsecos estáticos entram como `CalibrationSet` canônico em `SourceAdapterConfig.calibration` e são preservados por `read_calibration()`. O adapter não fabrica um modelo pinhole para calibrações externas que `camera_info` não descreve, como MEI omnidirecional: um `MeiCameraModel` entra pela calibração externa (`SourceAdapterConfig.calibration`), nunca derivado de `camera_info`.

### Decisões de mapeamento

- **Identidade**: `observation_id = "<topico-sanitizado>-<indice:06d>"` (índice por tópico, 0-based); `sensor_id` deriva do próprio nome do tópico sanitizado (`/camera/image_raw` → `camera_image_raw`) — o v0 não tem configuração separada de nome de sensor.
- **Timestamp**: `header.stamp` da mensagem usa um `clock_id` compartilhado por todos os tópicos do adapter (`SourceAdapterConfig.timestamp_clock_id`, ou identidade determinística da fonte). O timestamp de gravação do bag permanece separado em `provenance.raw_metadata["bag_timestamp_nanoseconds"]`; ele não substitui nem é confundido com o clock do sensor.
- **Imagem e point cloud**: padding por linha é removido. `mono16` big-endian e campos multibyte de `PointCloud2` big-endian são normalizados para little-endian, e a provenance registra o layout original.
- **IMU**: cada vetor é `None` quando a respectiva `*_covariance[0] == -1.0`; quando disponível, a matriz 3×3 correspondente também é preservada.
- **Pose externa**: `Odometry.header.frame_id` → `parent_frame`; `child_frame_id` → `frame_id`; pose, twist e as duas covariâncias 6×6 são preservados.
- **Calibração**: todas as mensagens de `camera_info` devem ser equivalentes durante a sequência; mudança de intrínsecos falha explicitamente. `distortion_model == "equidistant"` vira `FisheyeCameraModel`; `plumb_bob` e `rational_polynomial` viram `PinholeCameraModel`. A calibração descoberta é mesclada com a fornecida externamente, conflitos por sensor são rejeitados e observações recebem o `calibration_id` aplicável.

### Erros vs. warnings

Tópico obrigatório (`required_topics`) ausente do bag → `MissingRequiredTopicError`, e janela inválida → `InvalidSourceWindowError`, ambos levantados pela própria chamada de `read_observations()`, antes de o iterador devolvido ser avançado (só a lista de tópicos e o índice do bag são lidos para checá-los; a decodificação começa no primeiro `next()`). Mensagem individual com conteúdo inválido — encoding de imagem não suportado, `PointField.datatype` fora de 1–8, layout de linha/stride inconsistente — levanta `ValueError` durante a decodificação e é registrada via `SourceAdapterWarning` e pulada; o restante da leitura continua. A `reason` do warning é a mensagem semântica do decoder (por exemplo, `unsupported PointField datatype 9 for field 'intensity'; supported datatypes are 1-8`), nunca só o valor cru que falhou.

`ValueError` é a única exceção contratual de "mensagem não decodificável" (issue #605). `KeyError`/`AttributeError` durante a decodificação — mensagem estruturalmente incompatível com o kind configurado (por exemplo, um tipo de mensagem ausente do typestore em um tópico configurado) ou erro de programação — **não** viram warning: propagam de `read_observations()` com o traceback original e abortam o run, em vez de pular todas as mensagens do tópico e completar "com sucesso" com zero observações.

### Fixtures de teste

Os testes geram um bag ROS 1 sintético e determinístico em tempo de execução com `rosbags.rosbag1.Writer`, em vez de versionar um arquivo binário `.bag` no repositório — evita problemas de tamanho/licença e mantém a fixture auditável como código Python (`tests/ingestion/adapters/test_ros1_bag.py`).

## `Ros2BagSourceAdapter` (issue #45)

`src/contextmap/ingestion/adapters/ros2_bag.py`. Mesmo escopo v0, mesmas decisões de mapeamento e mesma política de erros/warnings do adapter ROS 1 (acima) — reaproveita `_ros_common.py` para toda a decodificação exceto `CameraInfo`, cujo casing de campo difere.

### Diferenças reais em relação ao ROS 1

- **Path**: um bag ROS 2 é um **diretório** (contém `metadata.yaml` + arquivo(s) de storage, tipicamente SQLite3), não um arquivo único como o `.bag` do ROS 1. `SourceAdapterConfig.path` aponta para esse diretório.
- **Deserialização**: `rosbags` usa CDR (`typestore.deserialize_cdr`) para ROS 2, em vez de `deserialize_ros1`.
- **`std_msgs/Header`**: o ROS 2 removeu o campo `seq` (nunca usado por este adapter — a identidade da observação já vem do índice por tópico, não de `seq`).
- **`sensor_msgs/CameraInfo`**: campos `d`/`k`/`r`/`p` minúsculos, em vez de `D`/`K`/`R`/`P`.

### Prova de contrato único

`tests/ingestion/adapters/test_ros2_bag.py::test_ros1_and_ros2_adapters_produce_the_same_canonical_shape` decodifica a mesma imagem lógica de um bag ROS 1 e de um bag ROS 2 e verifica que o `ImageObservation` resultante é idêntico — demonstrando a exigência central da issue #45: "o mesmo código downstream consome sequências independente de qual ROS as produziu".
