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
| `entity_resolution` | `EntityResolutionExecutor` | `entities` | política de retrieval, o `MatchEvidenceBuilder` (canal de geometria obrigatório; semântico, temporal, aparência e representação só quando configurados) e a política de decisão conservadora |
| `spatial_relations` | `SpatialRelationsExecutor` | `entities`, `geometry` | convenções de eixo, política de candidatos e resumo de geometria, sempre presentes; avaliadores geométrico e de contato só quando configurados |

## Identidade

O `run_id` que o writer grava vem de `StageRequest.identity()`: o estágio, o `config_digest` e o hash de conteúdo exato de cada entrada. Uma execução idêntica publica a **mesma** identidade e, portanto, o mesmo conteúdo; qualquer mudança de configuração ou de entrada muda a identidade. É isso que faz dois runs da mesma execução terem hashes contratuais idênticos e que mantém o reuso válido entre runs. O `run_index` é o número do run (`StageRequest.run_number()`). O `content_hash` do `ArtifactRef` é o digest do inventário contratual do manifest (`inventory_digest`).

## Cardinalidade

Um estágio produz **exatamente um** artifact, e um executor consome **exatamente um** run por entrada: um pedido com vários runs para a mesma entrada (`multiple`) é recusado com `ExecutorError`, sem escolher um em silêncio. Comparar dois runs de percepção ou dois braços de ablação é executar a runtime duas vezes e ler os dois artifacts pela referência.

## Topologia

Para que os executores tenham o que precisam, o DAG `canonical/1` dá `sequence` à fusão: os instantes de aquisição das observações físicas vêm da sequência.

## Como um executor chega a existir

Construir um destes executores manualmente (juntar backend, políticas e classe do executor) é trabalho da [composition root](composition.md). `compose_executors(effective, ...)`, em `contextmap.runtime.composition`, faz exatamente isso a partir de uma `EffectiveConfig`: monta `StateEstimationExecutor`, `GeometricMappingExecutor`, `SensorAssociationExecutor`, `SemanticFusionExecutor`, `EntityResolutionExecutor` e `SpatialRelationsExecutor` (nessa ordem de dependência) e devolve um `dict[str, StageExecutor]` indexado por `stage_id`, sem fabricar nada para o que não pode compor de verdade. `IngestionStageExecutor` fica de fora dessa composição automática: ele precisa de um `IngestionRequest` concreto, que é entrada de uma execução (os flags de `contextmap ingest`), não parte de uma configuração — continua sendo construído e injetado explicitamente por quem chama (`main(executors=...)` ou `Runtime(executors=...)`), que é como o comando `ingest` e os testes deste módulo já o exercitam.

## O que ainda não existe

- **`visual_perception`** não tem executor: a percepção usa modelos e GPU e entra por **referência** a um run existente (`provided` ou `inputs.selections`). Um executor real de percepção é um trabalho à parte.
- **`point_representation`** também não tem executor, pelo mesmo motivo (backend dependente de modelo); é o único estágio opcional do canônico, então um run que não o habilita nunca sente essa lacuna.
- **`semantic_mapping`** faz parte da topologia de `canonical/2` (é a fonte de `entities` para `entity_resolution`; não existe em `canonical/1`, que termina em `semantic_fusion`), mas não tem componente de catálogo nem executor automático: seu artifact precisa ser **suprido** (`provided`/`inputs.selections`) para que `entity_resolution` o consuma, nunca produzido no mesmo run por `compose_executors`.
- **`context_map`** (a montagem do `ContextMapArtifact`) ainda não faz parte de nenhum preset: é trabalho à parte, tratado por outro milestone.
- **`entity_resolution.appearance`** e **`entity_resolution.representation`**, quando selecionados, precisam de um `FeatureVectorSource`/`RepresentationVectorSource` que só existe vinculado a runs de percepção/representação já abertos: `compose_executors` os obtém de um `provider` fornecido pelo chamador (o mesmo mecanismo de `RuntimeProvider` já usado por `sam3`/`qwen`/`gemini`), nunca os inventa.
