# Fronteira de source adapters

Este documento descreve `src/contextmap/ingestion/source_adapter.py`: o contrato (`Protocol`) que qualquer adapter de fonte (ROS 1, ROS 2, dataset) implementa, e onde implementações concretas devem viver.


## Boundary da fonte

```mermaid
flowchart LR
    R1[ROS 1 bag] --> A1[Ros1BagSourceAdapter]
    R2[ROS 2 bag] --> A2[Ros2BagSourceAdapter]
    DS[Dataset gravado] --> AX[Adapter específico]

    A1 --> P[SourceAdapter Protocol]
    A2 --> P
    AX --> P

    P --> O[SourceObservation]
    P --> C[CalibrationSet]
    P --> W[SourceAdapterWarning]

    O --> I[Pipeline canônica de Ingestion]
    C --> I
    W --> D[Diagnostics]
```

O ponto de integração é o contrato canônico, não a API nativa da fonte. Por isso ROS e datasets permanecem detalhes de borda e não vazam para capabilities downstream.


## Decisão central: adapters produzem `SourceObservation`s, não eventos brutos

`SourceAdapter.read_observations()` produz diretamente `SourceObservation`s canônicas (issue #38) — uma por mensagem física da fonte, **sem agrupamento**. O agrupamento/sincronização (`synchronize()`, issue #40) é um estágio separado, aplicado sobre a saída do adapter. Isso evita duplicar a lógica de sincronização dentro de cada adapter e mantém os adapters simples: decodificar e normalizar, nada além disso.

## Nenhum objeto ROS-nativo cruza a fronteira

Toda assinatura de `SourceAdapter` retorna apenas contratos canônicos (`SourceObservation`, `SourceAdapterCapabilities`, `SourceAdapterWarning`) ou primitivos. Nenhum método aceita ou devolve `rosbag.Bag`, `rclpy` message types, ou qualquer struct nativa de SDK de fonte.

## Onde implementações concretas vivem

Implementações concretas (`Ros1BagSourceAdapter`, issue #44; `Ros2BagSourceAdapter`, issue #45) devem viver em `src/contextmap/ingestion/adapters/<nome>.py`. Isso não é apenas convenção — é **imposto** por `tests/architecture/test_boundaries.py`: `HEAVY_SDK_ROOTS` (`rclpy`, `rosbag`, `rosbag2_py`, `torch`, `transformers`, `segment_anything`) só pode ser importado dentro de `backends/`, `infrastructure/` ou `adapters/` de uma capability. Este módulo (`source_adapter.py`) fica na raiz de `ingestion` porque o `Protocol` em si não depende de nenhuma dessas bibliotecas — só as implementações concretas dependem.

Consequência prática: quem só lê um `SequenceArtifact` já persistido nunca precisa instalar ROS 1, ROS 2 ou qualquer SDK de fonte — essas dependências ficam isoladas em `adapters/`, e `pyproject.toml` as declara como extra opcional.

## Configuração: `SourceAdapterConfig`

Forma comum a todos os adapters, alinhada ao YAML mostrado nas issues #44/#45:

```yaml
source:
  type: ros1_bag       # SourceAdapterConfig.source_type
  path: data/example.bag  # SourceAdapterConfig.path
  timestamp_clock_id: robot-clock  # opcional; compartilhado entre os headers
  topics:                    # SourceAdapterConfig.topics (SourceTopicMapping)
    rgb: /camera/image_raw
    camera_info: /camera/camera_info
    lidar: /velodyne_points
    imu: /imu/data
    pose: /odom
```

`required_topics` (ex.: `{"rgb", "lidar"}`) declara quais campos de `topics` são obrigatórios — se ausentes na fonte real, o adapter levanta `MissingRequiredTopicError` em vez de simplesmente pular a modalidade em silêncio.

`timestamp_clock_id` identifica o domínio de clock de `header.stamp` compartilhado pelos tópicos configurados. Quando omitido, o adapter deriva uma identidade determinística da família e do path da fonte. O timestamp de gravação do bag permanece separado na provenance. `calibration` recebe um `CalibrationSet` canônico opcional, inclusive extrínsecos estáticos que não estejam representados por `CameraInfo`.

Configuração específica de um adapter que não cabe na forma comum (ex.: identidade de storage/serialização do ROS 2) vai em `SourceAdapterConfig.extra`, em vez de a fronteira crescer um campo por família de adapter — mantém o contrato pequeno o suficiente para não precisar de um sistema de plugins genérico.

## Erros vs. warnings

- **Erro** (`MissingRequiredTopicError`, `UnsupportedSourceMessageError`, ambos `SourceAdapterError`): interrompe a leitura — tópico obrigatório ausente, ou mensagem que o adapter não sabe decodificar de forma alguma.
- **Warning** (`SourceAdapterWarning`, acumulado e devolvido por `warnings()`): mensagem malformada ou não suportada que foi pulada, mas não impede o restante da leitura. Nunca silenciosamente descartada sem rastro — sempre aparece em `warnings()` com `topic`/`message_index`/`reason`.

## Capacidades

`capabilities()` reporta o que a fonte configurada realmente oferece (`rgb`, `lidar`, `imu`, `external_pose`, `calibration`), permitindo que um consumidor decida antecipadamente se uma modalidade necessária está disponível, sem precisar iterar toda a fonte primeiro.

## Mesmo boundary para ROS 1 e ROS 2

Qualquer classe que implemente `capabilities()`, `read_observations()`, `read_calibration()` e `warnings()` satisfaz `SourceAdapter` (`Protocol` com `@runtime_checkable`) — código downstream nunca precisa de `if isinstance(adapter, Ros1BagSourceAdapter)` ou qualquer branch por versão de ROS.

## Adapter `pose_file` (issue #376)

`contextmap.ingestion.adapters.pose_file.PoseFileSourceAdapter` lê um arquivo de pose autônomo (não um tópico de bag) e produz `ExternalPoseMeasurement`s. Existe porque nem toda fonte publica pose num tópico ROS — o `corridor-02`, por exemplo, só tem câmera, LiDAR e IMU no bag; sua trajetória (`corridor-02-gt.txt`) é um arquivo texto separado.

Segue a mesma forma de `SourceAdapterConfig` que os adapters de bag, sem usar `topics` (todos os campos ficam `None`/vazios); a configuração específica do formato vai em `extra`:

```yaml
source:
  type: pose_file          # SourceAdapterConfig.source_type
  path: datasets/corridor-02/corridor-02-gt.txt
  timestamp_clock_id: corridor-02-header   # para casar com o clock do bag, quando aplicável
  extra:
    format: tum             # único suportado no v0; "t x y z qx qy qz qw"
    parent_frame: map       # obrigatório — o arquivo não declara frame nenhum
    body_frame: epson       # obrigatório — frame cuja pose o arquivo reporta
    sensor_id: external_pose_file   # opcional, padrão "external_pose_file"
```

`format`, `parent_frame` e `body_frame` são obrigatórios e validados na construção do adapter (`PoseFileConfigError`, antes de qualquer leitura) — nada é inferido do nome do arquivo (um arquivo chamado `*-gt.txt` não vira "ground truth" por convenção de nome; ele continua sendo uma medição de entrada, ver `state_estimation/docs/backends.md`). `format="tum"` já declara a convenção de unidades do v0 (segundos decimais, metros, quaternion unitário `(x, y, z, w)`); um segundo formato com unidades diferentes ganharia seu próprio valor de `format`, não um campo de unidades solto.

Cada observação carrega, em `provenance.raw_metadata`, o `format` declarado e o hash `sha256` do conteúdo do arquivo inteiro (`compute_source_content_hash`, de `sequence_provenance.py`) — a mesma função usada para o hash de fonte de bag, computada uma vez por chamada de `read_observations()`. Uma linha malformada (contagem de campos errada, timestamp ou número não finito) vira `SourceAdapterWarning` e é pulada, como nos adapters de bag — nunca interrompe a leitura das demais linhas. `read_calibration()` só repassa `SourceAdapterConfig.calibration`, quando fornecida; o arquivo de pose não tem calibração própria.

`observation_id` segue `"<sensor_id>-<número da linha, 6 dígitos>"`, único e estável para uma mesma execução sobre o mesmo arquivo (linhas em branco e comentários `#` não contam como linha de dado, mas contam para a numeração de linha usada nos warnings).
