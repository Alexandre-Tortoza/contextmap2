# Adaptação genérica de fontes, datasets e streaming

> Status: **planejamento futuro, não comprometido**
>
> Este documento registra uma direção arquitetural para tornar o ContextMap2 adaptável a datasets heterogêneos e streams reais de robôs sem contaminar o core com exceções específicas de fonte.
>
> Ele **não define milestone, prazo, implementação obrigatória ou mudança imediata no perfil canônico**. A prioridade atual continua sendo validar a Solution 1 existente, congelar cenários reproduzíveis e medir qualidade antes de ampliar a superfície do sistema.

## 1. Motivação

O objetivo de longo prazo do ContextMap2 é receber dados multimodais de diferentes robôs e datasets e produzir a mesma forma de memória espacial contextual, independentemente de a origem ser:

- ROS 1 bag;
- ROS 2 bag;
- MCAP ou outro container registrado;
- dataset acadêmico com arquivos e manifests próprios;
- sequência de imagens, depth e point clouds em diretórios;
- trajetória/pose fornecida externamente;
- stream online de um robô.

A genericidade desejada não significa que o sistema deve "adivinhar" automaticamente o significado de qualquer dataset.

A propriedade arquitetural desejada é:

> uma nova fonte deve exigir configuração e, quando necessário, um adapter de borda; os módulos downstream não devem exigir mudanças por causa da origem dos dados.

A fronteira central continua sendo:

```text
raw source
    -> source adapter
    -> canonical observations
    -> validation / calibration / synchronization
    -> SequenceArtifact
    -> pipeline downstream
```

Depois de publicado um `SequenceArtifact`, visual perception, state estimation, geometric mapping, sensor association, semantic fusion e os estágios superiores não devem precisar saber se a sequência veio de ROS 1, ROS 2, KITTI, TartanGround, um diretório customizado ou um robô online.

## 2. Estado atual relevante

O projeto já possui decisões que devem ser preservadas.

### 2.1 Source adapters já são uma fronteira explícita

`contextmap.ingestion.SourceAdapter` converte fontes específicas em `SourceObservation` canônica. Objetos ROS e tipos de SDK não atravessam essa fronteira.

`SourceAdapterConfig` já possui uma forma comum com:

- `source_type`;
- `path`;
- `topics`;
- `timestamp_clock_id`;
- calibração externa opcional;
- tópicos obrigatórios;
- janela temporal;
- `extra` para detalhes específicos do adapter.

Essa separação deve continuar sendo a base. Não há intenção de criar um "mega-adapter" que contenha conhecimento de todos os datasets.

### 2.2 O clock de gravação já é separado do clock do sensor

`SourceWindow` opera explicitamente no clock de gravação da fonte.

Nos adapters ROS, esse clock é diferente do `header.stamp` da observação e o projeto já rejeita a hipótese de que os dois sejam implicitamente equivalentes.

A issue #554 estende essa direção com normalização temporal configurável e dataset-scoped, preservando o valor bruto.

### 2.3 Sincronização já é separada da decodificação

A política atual `nearest_within_tolerance` é aplicada depois que o adapter produziu observações canônicas.

Isso deve ser preservado. O adapter decodifica; a sincronização associa observações.

### 2.4 Calibração já é um contrato canônico

`CalibrationSet` fornece um caminho para calibração descoberta na fonte e calibração fornecida externamente.

A evolução futura deve tornar a precedência, conflitos e provenance mais explícitos sem transportar formatos ROS ou de dataset para o core.

### 2.5 Runtime já separa topologia, backends e execução

A configuração runtime já distingue:

- `pipeline`;
- `components`;
- `inputs`;
- `resources`;
- `policies`.

A adaptação de fontes deve se integrar a esse modelo sem transformar configuração de dataset em parâmetros científicos de percepção ou mapping.

## 3. Princípio de projeto

A arquitetura futura deve separar cinco categorias de configuração que hoje podem aparecer misturadas em um único invocation/config.

```text
Source Profile
    como ler a fonte

Rig Profile
    quais sensores existem e como estão relacionados

Sequence/Dataset Profile
    particularidades daquela captura/dataset

Pipeline Profile
    quais capacidades e backends executar

Run Overrides
    janela, seleção, device, debug, experimentos
```

Um run concreto deve ser resolvido pela composição explícita dessas camadas.

Exemplo conceitual:

```text
sources/ros1.yaml
+ rigs/robot-x.yaml
+ datasets/corridor-02.yaml
+ pipelines/canonical-mapping.yaml
+ runs/window-debug.yaml
        |
        v
effective configuration
        |
        v
preflight
        |
        v
execution
```

A configuração efetiva continua sendo persistida e versionada para reprodução.

## 4. Objetivo futuro

O sistema deve conseguir expressar, sem alterar módulos downstream:

1. diferentes quantidades e combinações de sensores;
2. diferentes relógios e erros conhecidos de tempo;
3. diferentes árvores de frames;
4. diferentes convenções de coordenadas;
5. diferentes modelos de câmera;
6. diferentes formatos de depth e point cloud;
7. calibração proveniente de fontes distintas;
8. diferentes estratégias de sincronização;
9. pose externa ou state estimation calculada;
10. replay offline ou streaming online;
11. ausência de modalidades opcionais;
12. requisitos diferentes de pipeline.

O sistema não deve prometer sucesso semântico ou geométrico em qualquer dataset. Ele deve conseguir representar a fonte corretamente, validar suas premissas e falhar de forma explicável quando uma configuração não satisfaz os requisitos do pipeline.

## 5. Área A: inventário genérico de sensores

### Problema

`SourceTopicMapping` possui atualmente slots únicos:

```text
rgb
camera_info
lidar
imu
pose
```

Isso é suficiente para a Solution 1 atual, mas não representa rigs arbitrários como:

- seis câmeras;
- câmera RGB + câmera térmica;
- dois LiDARs;
- múltiplas IMUs;
- RGB e depth em sensores distintos;
- sensores com clocks diferentes.

### Direção proposta

Evoluir, quando houver necessidade comprovada, para um inventário de sensores identificados por ID estável.

Exemplo conceitual:

```yaml
sensors:
  camera_front:
    modality: image
    channel: /camera/front/image
    frame: camera_front
    clock: camera-clock

  camera_rear:
    modality: image
    channel: /camera/rear/image
    frame: camera_rear
    clock: camera-clock

  lidar_top:
    modality: point_cloud
    channel: /ouster/points
    frame: lidar_top
    clock: lidar-clock

  imu_main:
    modality: imu
    channel: /imu/data
    frame: imu_link
    clock: imu-clock
```

### Requisitos

- IDs de sensor são estáveis e independentes de nomes ROS;
- modalidade é explícita;
- channel/topic/path pertence ao adapter, não ao downstream;
- frame e clock são referenciados por identidade;
- múltiplos sensores da mesma modalidade são permitidos;
- capabilities da fonte devem refletir o inventário real;
- `SequenceArtifact` preserva qual sensor originou cada observação.

### Migração

Não substituir `SourceTopicMapping` até existir um dataset/robô real que exija múltiplos sensores da mesma modalidade.

A migração deve manter compatibilidade com os perfis canônicos congelados.

## 6. Área B: política de tempo e clock domains

### Problema

Fontes reais podem apresentar:

- epoch incorreta com progressão relativa correta;
- clock da máquina diferente do clock do sensor;
- sensores independentes com clocks diferentes;
- timestamps ausentes;
- resets;
- jumps;
- drift;
- mensagens fora de ordem.

### Direção proposta

Manter explicitamente:

```text
recording_time_raw
source_timestamp_raw
normalized_event_time
clock_id
```

A normalização nunca substitui os valores brutos.

A issue #554 cobre o primeiro caso necessário: `constant_offset` configurado por dataset.

### Evoluções possíveis, somente quando justificadas por dados reais

- `none`;
- `constant_offset`;
- clock mapping conhecido entre sensores;
- correção afim `a*t+b` para skew mensurável;
- clock reset segmentation;
- online clock estimation.

Essas opções não devem ser implementadas antecipadamente.

### Regra

Toda transformação temporal deve:

- ter identidade/versionamento;
- ser explicitamente configurada;
- produzir diagnostics;
- persistir parâmetros no provenance;
- ser reproduzível;
- ser isolada ao source/dataset profile;
- nunca ser ativada por heurística global baseada em ano, topic ou nome do dataset.

## 7. Área C: frames e convenções de coordenadas

### Problema

Datasets variam em:

- nomes de frames;
- direção de transforms;
- eixo vertical;
- handedness;
- unidades;
- frame que representa o corpo do robô;
- presença ou ausência de `map`, `odom`, `base_link`.

### Direção proposta

Introduzir um profile declarativo de frames quando a diversidade real exigir.

Exemplo:

```yaml
frames:
  world: map
  body: base_link

  aliases:
    velodyne: lidar_top
    epson: imu_link

  convention:
    handedness: right
    length_unit: meter
    vertical_axis: z
```

### Requisitos

- não inferir convenções pelo nome do frame;
- aliases são resolvidos antes do downstream;
- transformações realizadas devem ser auditáveis;
- unidades canônicas devem ser explícitas;
- conflitos ou loops na árvore de frames falham no preflight;
- nenhuma conversão de eixo pode ocorrer silenciosamente.

## 8. Área D: calibração e precedência de fontes

### Problema

Uma fonte pode conter parte da calibração e depender de arquivos externos para o restante.

Exemplos:

- intrínsecos em `CameraInfo`;
- extrínsecos em TF;
- extrínsecos em YAML externo;
- calibração de fábrica em outro formato;
- valores conflitantes entre bag e arquivo.

### Direção proposta

Representar a estratégia de calibração como configuração explícita.

Exemplo:

```yaml
calibration:
  sources:
    - source_camera_info
    - source_tf
    - file: calibration/robot-x.yaml

  conflict_policy: fail
```

### Políticas aceitáveis

O default científico deve favorecer erro explícito.

Possíveis políticas futuras:

- `fail`;
- `require_equal_within_tolerance`;
- preferência explícita por uma fonte quando justificada.

Nunca usar "último valor vence" silenciosamente para calibração geométrica.

### Diagnostics

Registrar:

- origem de cada intrínseco/extrínseco;
- diferenças encontradas;
- tolerâncias;
- transforms ausentes;
- versão/hash dos arquivos externos.

## 9. Área E: modelos de câmera e imagem

### Problema

Pinhole não cobre todas as câmeras de robótica.

Datasets podem utilizar:

- pinhole;
- fisheye;
- equidistant;
- imagens retificadas;
- imagens cropped;
- resoluções diferentes entre RGB e depth;
- diferentes encodings.

### Direção proposta

O contrato de calibração deve conseguir expressar os modelos efetivamente suportados sem forçar cada downstream a conhecer formatos de origem.

Para cada modelo suportado, devem existir:

- parâmetros canônicos;
- projection/unprojection testadas;
- domínio válido de pixels;
- distortion/undistortion explícitos;
- testes numéricos;
- fixtures de calibração.

### Regra de implementação

Adicionar um modelo somente quando houver dataset real e teste objetivo.

Não expandir a enumeração de modelos por completude teórica.

## 10. Área F: depth e point cloud interpretation

### Depth

A configuração pode precisar declarar, quando a fonte não é autodescritiva:

- unidade/scale;
- valor inválido;
- depth registered ou não;
- frame;
- modelo da câmera;
- relação com RGB.

Exemplo conceitual:

```yaml
sensors:
  depth_front:
    modality: depth
    scale_to_meters: 0.001
    invalid_value: 0
    registered_to: camera_front
```

### Point clouds

Campos podem variar:

- XYZ;
- intensity;
- ring;
- per-point relative time;
- RGB;
- normals;
- return number.

O contrato deve distinguir:

1. campos geométricos obrigatórios;
2. campos opcionais preservados;
3. campos necessários por um backend específico.

Um point representation backend não deve exigir que Ingestion invente normals ou RGB que a fonte não possui.

## 11. Área G: sincronização extensível

### Estado atual

A política `nearest_within_tolerance` é simples, explícita e adequada à Solution 1.

### Necessidades futuras possíveis

- exact timestamp;
- nearest;
- nearest com tolerância por modalidade;
- interpolação de pose;
- interpolação de IMU;
- associação many-to-one;
- trigger por câmera ou LiDAR;
- sincronização entre múltiplas câmeras.

### Direção proposta

Tratar sincronização como uma policy selecionável.

Exemplo conceitual:

```yaml
synchronization:
  policy: nearest_within_tolerance
  anchor: camera_front

  tolerances:
    lidar_top: 50000000
    imu_main: 5000000
```

### Restrições

- clocks incompatíveis nunca são comparados implicitamente;
- interpolação precisa preservar quais amostras foram usadas;
- offsets e resíduos devem ser diagnosticados;
- eventos descartados continuam explícitos;
- não permitir uma política que esconda perda sistemática de observações.

## 12. Área H: pose e state estimation

### Problema

Datasets podem fornecer:

- ground truth;
- wheel odometry;
- visual odometry;
- GNSS/INS;
- SLAM trajectory;
- somente IMU + LiDAR;
- nenhuma pose.

### Regra de domínio

Uma pose lida da fonte continua sendo uma **medição externa**, não automaticamente a trajetória canônica do mapa.

### Direção futura

A configuração deve deixar claro quem possui a estimativa usada pelo mapping.

Exemplos:

```yaml
state_estimation:
  backend: external_pose
```

```yaml
state_estimation:
  backend: fast_lio
```

```yaml
state_estimation:
  backend: orb_slam3
```

O preflight deve verificar se as modalidades requeridas pelo backend existem antes da execução.

## 13. Área I: adapters específicos de dataset

### Regra

Um dataset com estrutura própria pode ter adapter próprio.

Exemplos futuros:

```text
adapters/
    ros1_bag.py
    ros2_bag.py
    mcap.py
    kitti.py
    tartanground.py
    directory_manifest.py
```

A existência de adapters específicos não é uma falha de genericidade.

A genericidade está no fato de todos produzirem o mesmo contrato canônico.

### Configuração específica

`SourceAdapterConfig.extra` é adequado enquanto poucos adapters precisam de parâmetros pequenos.

Se `extra` crescer ou ficar difícil de validar, cada adapter deve ganhar um schema próprio de parâmetros validado na composition boundary.

Exemplo:

```yaml
source:
  type: directory_manifest
  parameters:
    image_pattern: rgb/*.png
    lidar_pattern: lidar/*.bin
    timestamps: timestamps.csv
```

O downstream não conhece esses parâmetros.

## 14. Área J: streaming online

Streaming não deve ser tratado apenas como "bag sem fim".

Ele introduz problemas próprios:

- mensagens atrasadas;
- mensagens fora de ordem;
- backpressure;
- buffer finito;
- sensores temporariamente indisponíveis;
- reconexão;
- segmentação/persistência incremental;
- shutdown incompleto;
- latência máxima.

### Source contract

Uma implementação futura pode compartilhar os contratos de `SourceObservation`, mas provavelmente precisará de uma interface de lifecycle distinta do replay de arquivo.

Não forçar `SourceAdapter.read_observations()` offline a modelar todos os problemas online se isso degradar sua simplicidade.

### Policy conceitual

```yaml
stream:
  reorder_window_ms: 100
  buffer_duration_s: 5
  late_event_policy: drop
  missing_sensor_policy: warn
  artifact_segment_duration_s: 60
```

### Requisitos

- backpressure explícito;
- bounded memory;
- late events contabilizados;
- clock diagnostics online;
- artifacts finalizados atomicamente por segmento;
- nenhuma perda silenciosa;
- possibilidade de replay posterior do mesmo segmento.

### Questão aberta

Avaliar se streaming deve publicar `SequenceArtifact` em segmentos imutáveis ou introduzir um artifact incremental temporário que somente se torna `SequenceArtifact` ao final de uma janela.

Essa decisão precisa de protótipo e métricas antes de implementação definitiva.

## 15. Área K: requirements/capabilities do pipeline

O pipeline não deve assumir que toda fonte possui todas as modalidades.

Exemplo:

```yaml
requirements:
  required_modalities:
    - image
    - lidar
  optional_modalities:
    - imu
    - external_pose
```

Um backend específico pode adicionar requisitos próprios.

Exemplo:

```text
FAST-LIO
    requires lidar + imu

RGB-D geometry
    requires rgb/depth + pose

LiDAR-only geometry
    requires lidar + pose/state-estimation path
```

O preflight deve comparar:

```text
source capabilities
        +
selected backend requirements
        +
pipeline requirements
        =
executable plan or actionable failure
```

## 16. Área L: profile composition

A configuração deve permitir reutilizar conhecimento sem copiar grandes documentos YAML.

### Source profile

Define apenas como acessar/decodificar a fonte.

```text
source type
storage
channels
decode parameters
```

### Rig profile

Define hardware relativamente estável.

```text
sensor inventory
sensor IDs
frames
nominal calibration
clock domains
```

### Dataset/sequence profile

Define particularidades daquela coleta.

```text
paths
topic/channel remapping
known clock correction
external calibration override
known missing sensors
dataset-specific quirks
```

### Pipeline profile

Define a experiência computacional/científica.

```text
enabled stages
backends
model identities
scientific parameters
```

### Run overrides

São escolhas efêmeras.

```text
window
selection
device
workspace
debug level
experiment-specific overrides
```

### Regra

Uma configuração de dataset não deve escolher modelo semântico só porque aquele dataset foi usado originalmente com esse modelo.

Fonte e algoritmo permanecem concerns distintos.

## 17. Provenance necessária

Toda adaptação que mude a interpretação dos dados precisa ser persistida.

O provenance futuro deve permitir responder:

- qual arquivo/stream originou esta observação?
- qual sensor físico/lógico a produziu?
- qual timestamp bruto foi recebido?
- qual clock foi usado?
- houve normalização temporal?
- qual calibração foi aplicada?
- de onde essa calibração veio?
- quais frame aliases/conversões ocorreram?
- quais unidades foram convertidas?
- qual política de sincronização associou as observações?
- quais eventos foram descartados?
- qual configuração efetiva resolveu essas decisões?

O artifact downstream não precisa duplicar todos esses dados, mas deve manter lineage suficiente até o `SequenceArtifact`.

## 18. Validation e preflight

A adaptabilidade só é útil se configurações erradas falharem antes de gerar mapas plausíveis mas incorretos.

### Preflight de fonte

Verificar:

- fonte acessível;
- channels esperados;
- message types suportados;
- capabilities reais;
- ranges temporais;
- clock domains;
- campos necessários;
- calibração disponível;
- frames necessários.

### Validation temporal

Verificar:

- monotonicidade;
- resets;
- jumps;
- distribuição de deltas;
- cobertura da janela;
- residual de correções configuradas.

### Validation geométrica

Verificar:

- intrínsecos;
- resolução;
- transforms finitos;
- quaternion normalization;
- árvore de frames;
- determinante/ortogonalidade onde aplicável;
- projection sanity.

### Validation multimodal

Verificar:

- taxa de associação;
- distribuição de offsets;
- coverage por sensor;
- observações descartadas;
- ausência prolongada de modalidade requerida.

### Resultado

Validation deve produzir diagnostics mensuráveis e persistidos.

Não corrigir automaticamente um problema somente para permitir que a pipeline continue.

## 19. Estratégia de avaliação

Nenhuma dessas extensões deve ser aceita apenas porque "funciona" em uma visualização.

Cada nova flexibilidade precisa de um caso objetivo.

### Matriz mínima futura

Para uma feature genérica ser promovida, testar:

1. dataset que motivou a feature;
2. dataset existente que não precisa dela, para detectar regressão;
3. fixture sintética controlada;
4. round-trip de provenance;
5. reproducibility da configuração;
6. diagnostics de configuração inválida.

### Exemplos

**Clock correction**

- sequência com epoch errada e offset constante conhecido;
- sequência normal sem correção;
- sequência com reset que deve rejeitar constant offset.

**Multi-camera**

- duas câmeras com frames diferentes;
- uma câmera ausente;
- pipeline que requer somente uma delas.

**Camera fisheye**

- projection/unprojection contra referência numérica;
- dataset pinhole existente sem alteração.

**Streaming**

- eventos em ordem;
- eventos fora de ordem;
- burst de mensagens;
- sensor desconectado;
- shutdown e recuperação dos artifacts.

## 20. Sequência futura de implementação

Esta ordem é somente uma proposta técnica. Não constitui roadmap.

### Fase A: consolidar a Solution 1 atual

Antes de ampliar abstrações:

- terminar a validação E2E;
- congelar os scenarios canônicos;
- estabilizar artifacts e contracts;
- medir qualidade e runtime;
- corrigir bugs observados em dados reais.

### Fase B: extrair variações que já apareceram em dados reais

Primeiros candidatos:

1. timestamp normalization, issue #554;
2. frame/calibration profile explícito, se a validação mostrar necessidade;
3. sensor inventory genérico, quando aparecer um rig multi-sensor que não caiba no modelo atual.

### Fase C: segundo dataset ou segundo robô

Selecionar deliberadamente uma fonte que difira da canonical em vários aspectos.

O objetivo não é obter melhor benchmark, mas verificar portability.

Critério:

> integrar a nova fonte deve exigir adapter/profile e configuração, não mudanças em módulos downstream.

### Fase D: streaming

Somente depois que o modelo offline estiver estável.

Implementar primeiro um stream replayável/determinístico para testar lifecycle, buffering e late events antes de executar em robô real.

## 21. Critérios para transformar uma ideia deste documento em issue/milestone

Uma seção deste documento só deve virar trabalho planejado quando existir pelo menos um dos seguintes gatilhos:

- um dataset real não pode ser representado corretamente pelo contrato atual;
- um robô real exige a capability;
- uma hipótese científica depende dela;
- uma limitação atual bloqueia avaliação relevante;
- uma duplicação concreta de adapters mostra necessidade de abstração;
- uma failure mode observada exige política configurável.

Além disso, antes de implementação devem existir:

1. problema reproduzível;
2. input real ou fixture;
3. comportamento esperado;
4. métrica ou invariant verificável;
5. impacto nos contracts;
6. estratégia de compatibilidade;
7. testes de não regressão.

"Seria útil suportar" não é critério suficiente.

## 22. Anti-patterns a evitar

### Dataset-specific branches no core

Evitar:

```python
if dataset == "corridor-02":
    ...
```

Dataset-specific behavior deve estar em profile ou adapter.

### Heurísticas de reparo silencioso

Evitar:

```python
if timestamp.year < 2010:
    timestamp.year = current_year
```

A correção deve ser explícita e auditável.

### Mega-schema prematuro

Não criar dezenas de campos para sensores nunca testados.

Adicionar contratos quando um problema real exigir.

### Adapter fazendo mapping/perception

Adapters apenas interpretam a fonte e produzem observações/calibração canônicas.

### Profile de dataset escolhendo ciência

Um dataset profile não deve selecionar SAM, DINO, CLIP, PTv3 ou outro backend por conveniência histórica.

### Fallback silencioso

Se o backend ou modalidade requerida não está disponível, falhar explicitamente.

### Genericidade sem teste

Uma abstração não é genérica porque possui muitos parâmetros.

Ela é genérica quando fontes distintas passam pelo mesmo contrato sem branches downstream.

## 23. Questões abertas

Estas perguntas devem permanecer abertas até existirem casos concretos.

1. O inventário genérico de sensores deve substituir `SourceTopicMapping` ou coexistir com uma forma simplificada?
2. Clock normalization pertence integralmente à Ingestion ou deve existir um primitive compartilhado de clock transform?
3. Qual conjunto mínimo de modelos de câmera é necessário para os datasets alvo?
4. TF dinâmico deve ser tratado como observação, calibração ou stream de transform separado?
5. Como representar sensores que mudam calibração durante uma sequência?
6. Como segmentar `SequenceArtifact` em streaming sem perder identidade contínua da sessão?
7. Até que ponto replay online e ingestão offline devem compartilhar implementação?
8. Quais campos extras de PointCloud2 devem ser preservados genericamente?
9. Como representar GNSS e georeferencing sem contaminar a geometria local?
10. Quando uma correção temporal deve ser considerada calibração de clock em vez de quirk de dataset?

Nenhuma dessas questões precisa ser resolvida para concluir a validação atual.

## 24. Definition of success de longo prazo

A direção estará funcionando quando for possível receber uma nova fonte e seguir um processo semelhante a:

```text
1. identificar sensores, clocks, frames e calibração
2. escolher ou implementar SourceAdapter
3. escrever source/rig/dataset profile
4. executar preflight
5. corrigir configuração, não código downstream
6. gerar SequenceArtifact
7. executar pipeline canônica
8. obter ContextMapArtifact
9. auditar toda decisão até a fonte
```

Um novo dataset pode exigir novo código na borda. Isso é aceitável.

O resultado indesejado seria precisar modificar geometric mapping, sensor association, semantic fusion ou semantic memory somente porque a fonte usa outro formato, outro nome de tópico, outro clock ou outro modelo de sensor.

## 25. Relação com o trabalho atual

Este documento não muda a prioridade atual.

No curto prazo:

- #554 resolve somente a necessidade real e observada de timestamp normalization;
- #176 continua responsável pela definição/congelamento do cenário canônico E2E;
- a validação da Solution 1 deve revelar quais flexibilidades futuras são realmente necessárias;
- novos knobs e abstrações não devem ser adicionados à pipeline apenas para antecipar datasets ainda não escolhidos.

A função deste documento é preservar a direção de longo prazo sem transformar possibilidades arquiteturais em dívida de implementação imediata.
