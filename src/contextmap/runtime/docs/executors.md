# Executores de estágio

Um executor roda **uma** capability para **um** estágio do DAG e grava só no diretório que a runtime lhe entrega. Ele é fino: abre as entradas por `StageRequest.directory_of(ref)`, chama o serviço público da capability com as políticas científicas com que foi construído, entrega o `StageRequest.output_dir` ao writer da capability e devolve um `ArtifactRef` com o `location`. Não calcula caminho, não aloca identidade e não decide ciência: todo limiar e toda política vêm de quem constrói o executor.

O módulo é `contextmap.runtime.executors` e é importado **explicitamente**, nunca por `import contextmap.runtime`: ele depende da raiz pública de cada capability que executa, e o pacote da runtime não pode carregar nenhuma delas (um teste de arquitetura garante). Só raízes públicas, nunca um backend.

## Executores existentes

| Estágio | Executor | Entradas | Políticas recebidas na construção |
|---|---|---|---|
| `ingestion` | `IngestionStageExecutor` (em `ingestion_service.py`) | — | o `IngestionRequest`; o `output_dir` e o `artifact_id` vêm do estágio |
| `state_estimation` | `StateEstimationExecutor` | `sequence` | o estimador composto e a prontidão que reporta para a geometria |
| `geometric_mapping` | `GeometricMappingExecutor` | `sequence`, `trajectory` | política de pose, correção de movimento, agregação opcional |
| `sensor_association` | `SensorAssociationExecutor` | `sequence`, `perception`, `trajectory`, `geometry` | oclusão, tolerâncias e política de pose; só o canal de geometria |
| `semantic_fusion` | `SemanticFusionExecutor` | `sequence`, `association`, `perception`, `geometry` | política de suporte e de acumulação baseline |
| `semantic_mapping` | `SemanticMappingExecutor` | `fusion`, `geometry` | política de materialização e o `SemanticMapId` |
| `entity_resolution` | `EntityResolutionExecutor` | `entities` | recuperação de candidatos, canais de comparação e política de resolução |
| `spatial_relations` | `SpatialRelationsExecutor` | `entities` (o run de resolução), `geometry` | o pacote `RelationsRunPolicies` (convenções de frame, candidatos, predicados geométricos e de contato) e a política de resumo da geometria |

## Identidade

O `run_id` que o writer grava vem de `StageRequest.identity()`: o estágio, o `config_digest` e o hash de conteúdo exato de cada entrada. Uma execução idêntica publica a **mesma** identidade e, portanto, o mesmo conteúdo; qualquer mudança de configuração ou de entrada muda a identidade. É isso que faz dois runs da mesma execução terem hashes contratuais idênticos e que mantém o reuso válido entre runs. O `run_index` é o número do run (`StageRequest.run_number()`). O `content_hash` do `ArtifactRef` é o digest do inventário contratual do manifest (`inventory_digest`).

## Cardinalidade

Um estágio produz **exatamente um** artifact, e um executor consome **exatamente um** run por entrada: um pedido com vários runs para a mesma entrada (`multiple`) é recusado com `ExecutorError`, sem escolher um em silêncio. Comparar dois runs de percepção ou dois braços de ablação é executar a runtime duas vezes e ler os dois artifacts pela referência.

## Topologia

Para que os executores tenham o que precisam, o DAG `canonical/1` dá `sequence` à fusão (os instantes de aquisição das observações físicas vêm da sequência) e `geometry` ao mapeamento semântico e às relações espaciais (o suporte 3D das entidades vem do mapa).

## Quem carrega as políticas

Os executores de Entity Resolution e de Spatial Relations recebem os pacotes de política na construção, como os demais: nenhum limiar é decidido pela runtime, e **nenhuma configuração do runtime os carrega hoje** (o perfil canônico só declara as capabilities e os componentes). Quem monta o executor (o teste, o script de validação ou um perfil futuro) escolhe as políticas, e elas entram no fingerprint do artifact. As convenções de frame de Spatial Relations precisam nomear o referencial em que o mapa geométrico está expresso; se divergirem, o estágio falha com a mensagem do frame, sem converter em silêncio (o mapa sintético do CI está em `odom`).

## O que ainda não existe

- **`visual_perception`** não tem executor: a percepção usa modelos e GPU e entra por **referência** a um run existente (`provided`). Um executor real de percepção é um trabalho à parte.
- **`context_map`**: o schema, o writer e o leitor do `ContextMapArtifact` existem, mas **não existe o passo de montagem** que transforma os runs de resolução e de relações em um `ContextMap` (só o builder de teste `assemble_from_runs`, que fixa à mão o estado semântico de cada entidade). Sem esse passo não há executor.
