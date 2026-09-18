# Fronteira de source adapters

Este documento descreve `src/contextmap/ingestion/source_adapter.py`: o contrato (`Protocol`) que qualquer adapter de fonte (ROS 1, ROS 2, dataset) implementa, e onde implementações concretas devem viver.

## Decisão central: adapters produzem `SourceObservation`s, não eventos brutos

`SourceAdapter.read_observations()` produz diretamente `SourceObservation`s canônicas (issue #38) — uma por mensagem física da fonte, **sem agrupamento**. O agrupamento/sincronização (`synchronize()`, issue #40) é um estágio separado, aplicado sobre a saída do adapter. Isso evita duplicar a lógica de sincronização dentro de cada adapter e mantém os adapters simples: decodificar e normalizar, nada além disso.

## Nenhum objeto ROS-nativo cruza a fronteira

Toda assinatura de `SourceAdapter` retorna apenas contratos canônicos (`SourceObservation`, `SourceAdapterCapabilities`, `SourceAdapterWarning`) ou primitivos. Nenhum método aceita ou devolve `rosbag.Bag`, `rclpy` message types, ou qualquer struct nativa de SDK de fonte.

## Onde implementações concretas vivem

Implementações concretas (`Ros1BagSourceAdapter`, issue #44; `Ros2BagSourceAdapter`, issue #45) devem viver em `src/contextmap/ingestion/adapters/<nome>.py`. Isso não é apenas convenção — é **imposto** por `tests/architecture/test_boundaries.py`: `HEAVY_SDK_ROOTS` (`rclpy`, `rosbag`, `rosbag2_py`, `torch`, `transformers`, `segment_anything`) só pode ser importado dentro de `backends/`, `infrastructure/` ou `adapters/` de uma capability. Este módulo (`source_adapter.py`) fica na raiz de `ingestion` porque o `Protocol` em si não depende de nenhuma dessas bibliotecas — só as implementações concretas dependem.

Consequência prática: quem só lê um `SequenceArtifact` já persistido (issue #39) nunca precisa instalar ROS 1, ROS 2 ou qualquer SDK de fonte — essas dependências ficam isoladas em `adapters/`, e o próprio `pyproject.toml` pode declará-las como extras opcionais quando os adapters concretos forem implementados.

O diretório `adapters/` não é criado por esta issue — só passa a existir quando a primeira implementação concreta (#44) precisar dele, para não antecipar estrutura vazia (`docs/development.md`, YAGNI).

## Configuração: `SourceAdapterConfig`

Forma comum a todos os adapters, alinhada ao YAML mostrado nas issues #44/#45:

```yaml
source:
  type: ros1_bag       # SourceAdapterConfig.source_type
  path: data/example.bag  # SourceAdapterConfig.path
  topics:                    # SourceAdapterConfig.topics (SourceTopicMapping)
    rgb: /camera/image_raw
    camera_info: /camera/camera_info
    lidar: /velodyne_points
    imu: /imu/data
    pose: /odom
```

`required_topics` (ex.: `{"rgb", "lidar"}`) declara quais campos de `topics` são obrigatórios — se ausentes na fonte real, o adapter levanta `MissingRequiredTopicError` em vez de simplesmente pular a modalidade em silêncio.

Configuração específica de um adapter que não cabe na forma comum (ex.: identidade de storage/serialização do ROS 2) vai em `SourceAdapterConfig.extra`, em vez de a fronteira crescer um campo por família de adapter — mantém o contrato pequeno o suficiente para não precisar de um sistema de plugins genérico.

## Erros vs. warnings

- **Erro** (`MissingRequiredTopicError`, `UnsupportedSourceMessageError`, ambos `SourceAdapterError`): interrompe a leitura — tópico obrigatório ausente, ou mensagem que o adapter não sabe decodificar de forma alguma.
- **Warning** (`SourceAdapterWarning`, acumulado e devolvido por `warnings()`): mensagem malformada ou não suportada que foi pulada, mas não impede o restante da leitura. Nunca silenciosamente descartada sem rastro — sempre aparece em `warnings()` com `topic`/`message_index`/`reason`.

## Capacidades

`capabilities()` reporta o que a fonte configurada realmente oferece (`rgb`, `lidar`, `imu`, `external_pose`, `calibration`), permitindo que um consumidor decida antecipadamente se uma modalidade necessária está disponível, sem precisar iterar toda a fonte primeiro.

## Mesmo boundary para ROS 1 e ROS 2

Qualquer classe que implemente `capabilities()`, `read_observations()` e `warnings()` satisfaz `SourceAdapter` (`Protocol` com `@runtime_checkable`) — código downstream nunca precisa de `if isinstance(adapter, Ros1BagSourceAdapter)` ou qualquer branch por versão de ROS.
