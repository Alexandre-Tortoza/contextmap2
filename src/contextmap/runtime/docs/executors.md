# Executores de estágio

Um executor roda **uma** capability para **um** estágio do DAG e grava só no diretório que a runtime lhe entrega. Ele é fino: abre as entradas por `StageRequest.directory_of(ref)`, chama o serviço público da capability com as políticas científicas com que foi construído, entrega o `StageRequest.output_dir` ao writer da capability e devolve um `ArtifactRef` com o `location`. Não calcula caminho, não aloca identidade e não decide ciência: todo limiar e toda política vêm de quem constrói o executor.

O módulo é `contextmap.runtime.executors` e é importado **explicitamente**, nunca por `import contextmap.runtime`: ele depende da raiz pública de cada capability que executa, e o pacote da runtime não pode carregar nenhuma delas (um teste de arquitetura garante). Só raízes públicas, nunca um backend.

## Executores existentes

| Estágio | Executor | Entradas | Políticas recebidas na construção |
|---|---|---|---|
| `ingestion` | `IngestionStageExecutor` (em `ingestion_service.py`) | — | o `IngestionRequest`; o `output_dir` e o `artifact_id` vêm do estágio |
| `visual_perception` | `VisualPerceptionExecutor` | `sequence` | os quatro backends compostos (`region_discovery`, `dense_features`, `region_features`, `semantic_interpreter`) |
| `state_estimation` | `StateEstimationExecutor` | `sequence` | o estimador composto e a prontidão que reporta para a geometria |
| `geometric_mapping` | `GeometricMappingExecutor` | `sequence`, `trajectory` | política de pose, correção de movimento, agregação opcional |
| `sensor_association` | `SensorAssociationExecutor` | `sequence`, `perception`, `trajectory`, `geometry` | oclusão, tolerâncias e política de pose; só o canal de geometria |
| `semantic_fusion` | `SemanticFusionExecutor` | `sequence`, `association`, `perception`, `geometry` | política de suporte e de acumulação baseline |

### `VisualPerceptionExecutor` (#507)

Diferente dos outros quatro, dois dos backends compostos (`dense_features`/`region_features`) não chegam prontos: são `FeatureFactory` (`Callable[[FeatureBuildScope], FeatureExtractor]`, definido em `composition.py`) porque um extrator de feature só pode ser construído depois que existe um run para gravar seus payloads. O executor:

1. cria um diretório de rascunho (irmão do `output_dir` final, nunca publicado) e materializa cada `ImageObservation` bruta da sequência num PNG decodificável (`PIL`/`numpy`, único formato de imagem convertido aqui — sem resize, crop ou retificação: é a mesma transformação identidade que `prepare_image(operations=())` já modela, o mesmo formato que `SensorAssociationExecutor._frame` usa para sua própria referência simbólica, não-decodificável);
2. constrói dois `FeatureBuildScope` (um por `feature_stage_id`, `dense_feature_extraction`/`region_feature_extraction`) apontando para um `_DeferredFeaturePayloadSink` — um encaminhador local que resolve o ciclo `PerceptionRunWriter` precisa de `configuration_digest()` (que só existe depois de `resolve_pipeline()` construir os backends) enquanto os extratores de feature precisam de um `payload_sink` já na própria construção; o encaminhador é vinculado ao writer real assim que ele existe, antes de qualquer observação ser processada; o escopo de `region_feature_extraction` sempre recebe um `mask_source` (`_RegionInlineMaskSource`) que recorta a máscara inline que a própria região já carrega (`region.mask`, em resolução de imagem inteira) para o retângulo local da caixa (`box_mask_shape`) — um backend condicionado a máscara (AlphaCLIP) usa isso; os demais (CLIP, DINOv2/v3) simplesmente o ignoram (#535);
3. chama `resolve_pipeline(CANONICAL_PRESET_V1, backend_factories=...)`, e por observação, `build_stage_graph()` / `execute_stage_graph()` / `assemble_perception_result()` — o mesmo caminho público que hoje só era exercitado manualmente ou em scripts de validação;
4. grava cada resultado com `PerceptionRunWriter.add_result()`/`add_stage_outcomes()` e finaliza; o diretório de rascunho é sempre removido (`finally`), esteja o run correto ou tenha falhado.

Um backend que ainda não satisfaz a forma que `CANONICAL_PRESET_V1` espera — por exemplo um `SemanticInterpreter` que só implementa a porta nova `interpret(request)` enquanto os estágios `scene_interpretation`/`region_interpretation` do preset canônico ainda despacham pela forma legada `interpret_scene`/`interpret_regions` — não recebe nenhum tratamento especial aqui: `execute_stage_graph()` isola essa falha por observação (`StageStatus.FAILED`, mensagem honesta) e os estágios independentes (region discovery, features densa/de região) continuam produzindo evidência real. O executor nunca contorna uma incompatibilidade backend/preset; isso é lacuna de capability, não de runtime.

## Identidade

O `run_id` que o writer grava vem de `StageRequest.identity()`: o estágio, o `config_digest` e o hash de conteúdo exato de cada entrada. Uma execução idêntica publica a **mesma** identidade e, portanto, o mesmo conteúdo; qualquer mudança de configuração ou de entrada muda a identidade. É isso que faz dois runs da mesma execução terem hashes contratuais idênticos e que mantém o reuso válido entre runs. O `run_index` é o número do run (`StageRequest.run_number()`). O `content_hash` do `ArtifactRef` é o digest do inventário contratual do manifest (`inventory_digest`).

## Cardinalidade

Um estágio produz **exatamente um** artifact, e um executor consome **exatamente um** run por entrada: um pedido com vários runs para a mesma entrada (`multiple`) é recusado com `ExecutorError`, sem escolher um em silêncio. Comparar dois runs de percepção ou dois braços de ablação é executar a runtime duas vezes e ler os dois artifacts pela referência.

## Topologia

Para que os executores tenham o que precisam, o DAG `canonical/1` dá `sequence` à fusão: os instantes de aquisição das observações físicas vêm da sequência.

## Como um executor chega a existir

Construir um destes executores manualmente (juntar backend, políticas e classe do executor) é trabalho da [composition root](composition.md). `compose_executors(effective, ...)`, em `contextmap.runtime.composition`, faz exatamente isso a partir de uma `EffectiveConfig`: monta `VisualPerceptionExecutor`, `StateEstimationExecutor`, `GeometricMappingExecutor`, `SensorAssociationExecutor` e `SemanticFusionExecutor` (nessa ordem de dependência) e devolve um `dict[str, StageExecutor]` indexado por `stage_id`, sem fabricar nada para o que não pode compor de verdade — `visual_perception` só entra quando os quatro pontos de variação (`region_discovery`, `dense_features`, `region_features`, `semantic_interpretation`) estão de fato selecionados e disponíveis; uma seleção parcial nunca gera um executor parcial. `IngestionStageExecutor` fica de fora dessa composição automática: ele precisa de um `IngestionRequest` concreto, que é entrada de uma execução (os flags de `contextmap ingest`), não parte de uma configuração — continua sendo construído e injetado explicitamente por quem chama (`main(executors=...)` ou `Runtime(executors=...)`), que é como o comando `ingest` e os testes deste módulo já o exercitam.

## O que ainda não existe

- **`point_representation`** não tem executor: depende de um backend com modelo/GPU (`ptv3`) sem um executor real ainda; é o único estágio opcional do canônico, então um run que não o habilita nunca sente essa lacuna. Continua entrando por **referência** a um run existente (`provided` ou `inputs.selections`), como `visual_perception` fazia antes de #507.
- **Estágios depois de `semantic_fusion`** (`semantic_mapping`, `entity_resolution`, `spatial_relations` e `context_map`) não fazem parte de `canonical/1`: as capabilities ainda não estão em `dev`. Um preset versionado posterior os declara, com os executores correspondentes.
