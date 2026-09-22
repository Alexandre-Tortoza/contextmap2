# Configuração do runtime

A configuração diz **o que executar e com quais implementações**; nunca diz como uma capability funciona. Ela separa as preocupações que a arquitetura mantém distintas e resolve as camadas de forma determinística.

## Documento

Arquivos `.json` ou `.toml` (não há YAML: seria uma dependência nova sem consumidor). Todas as seções são opcionais em um arquivo; o perfil fornece o restante.

```toml
[pipeline]
preset = "canonical/1"

[pipeline.stages]            # topologia: só estágios opcionais podem ser desligados/ligados
point_representation = true

[components.visual_perception.region_discovery]   # seleção de backend + parâmetros do backend
backend = "sam3"
[components.visual_perception.region_discovery.sam3]
checkpoint = "..."

[components.visual_perception.region_discovery.florence2]   # bloco de outro backend: ignorado
checkpoint = "..."

[inputs]                     # seleção/run inputs
sequence = "seq-01"
[inputs.selections]
state_estimation = "seq-01--run-0003"

[resources]                  # recursos/dispositivo
device = "cuda"
workspace = "workspace/run-a"

[policies]                   # política de debug
debug_level = "standard"     # none | standard | full
```

| Seção | Preocupação |
|---|---|
| `pipeline` | preset de topologia e quais estágios opcionais participam |
| `components` | backend escolhido para cada ponto de variação, com os parâmetros **somente desse backend** |
| `inputs` | sequência esperada e seleção explícita de runs/artifacts upstream por estágio: ids exatos, listas de runs, seleções nomeadas ou `latest` ([`selection.md`](selection.md)) |
| `resources` | dispositivo (repassado aos backends que declaram um parâmetro de dispositivo e não o definiram) e workspace |
| `policies` | nível de debug; o debug nunca é dependência contratual de um estágio downstream |

Não há campos de política de avaliação: nenhum consumidor existe ainda, e o schema não ganha campo sem consumidor.

## Precedência

Do menor para o maior, o último vence:

1. os padrões do **perfil** (`canonical/1`);
2. os **arquivos**, na ordem informada;
3. os **overrides** `dotted.path=valor`, na ordem informada.

Mapeamentos se mesclam chave a chave; qualquer outro valor substitui. O valor de um override é lido como literal JSON quando é um (`true`, `12`, `[1,2]`, `null`) e como texto caso contrário. O ambiente **não participa** da mescla: ele só carrega segredos.

## Configuração efetiva

`resolve_effective_config()` devolve uma `EffectiveConfig`. A resolução:

- valida a estrutura e devolve **todos** os problemas de uma vez (`ConfigurationError.problems`);
- lista todo estágio do preset com seu estado resolvido;
- mantém, por ponto de variação, apenas o bloco de parâmetros do **backend selecionado**: o bloco de outro backend não vaza para a execução nem entra no digest, o que permite trocar de backend por override sem editar o arquivo;
- descarta os componentes de um estágio desligado: um estágio não selecionado não é construído, então sua configuração não conta;
- repassa `resources.device` aos backends que o declaram, sem sobrescrever um valor explícito;
- rejeita combinações incompatíveis (hoje: o canal de evidência `point_representation` de Semantic Fusion exige o estágio `point_representation` habilitado).

O runtime **não escolhe backend em nome do usuário**: o perfil `canonical/1` fixa a topologia e deixa todo backend não selecionado. `check_selection()` reporta o que falta escolher.

## Digest e persistência

`digest` é o SHA-256 da forma JSON canônica (chaves ordenadas, sem espaços) do documento efetivo. Duas configurações que resolvem para os mesmos valores têm o mesmo digest, independentemente de quais arquivos, formatos ou overrides as produziram. `sources` registra as camadas (perfil, arquivos com hash do conteúdo, e a chave de cada override, nunca o valor).

`write_effective_config()` grava `effective_config.json` com `schema_version`, `digest`, `config` e `sources`. A escrita é atômica e nunca substitui um arquivo existente: repetir a mesma configuração é um no-op; uma configuração diferente é recusada, porque um run publicado é imutável. `read_effective_config()` recusa outra versão de schema, digest divergente e documento não resolvido.

## Segredos

Um segredo nunca faz parte da configuração:

- nomes de parâmetro que denotam credenciais (`api_key`, `password`, `authToken`, ...) são rejeitados na validação, com a orientação de usar o ambiente; `max_new_tokens` ou `keyframe_stride` não são confundidos com credenciais;
- cada backend declara no catálogo os nomes das variáveis de ambiente de que precisa;
- `resolve_secrets()` lê apenas esses nomes; `ResolvedSecrets` só entrega um valor por `get()`, e `repr`/`str` mostram nomes, nunca valores.

## Três níveis de verificação

| Função | Pergunta | Precisa de |
|---|---|---|
| `resolve_effective_config()` | o documento é válido e coerente? | nada |
| `check_selection()` | todo ponto de variação de um estágio habilitado tem backend? | nada |
| `check_availability()` | os módulos opcionais e os segredos dos backends selecionados existem? | consulta de módulos (sem importá-los) e ambiente |

Nenhuma delas carrega modelo. É o que permite um dry-run e uma inspeção de topologia antes de qualquer execução pesada.

## Catálogo

O catálogo (`contextmap.runtime.catalog`) é dado puro: nomeia estágios, pontos de variação e backends, e declara o que cada backend precisa (módulos opcionais do código **empacotado**, segredos, parâmetro de dispositivo). Um backend cujo runtime o chamador fornece não lista módulos: os do runtime fornecido são do chamador. Ele não importa backend, não carrega modelo e não define valor científico. Os parâmetros de um backend, como checkpoint e limiares, são validados por quem os possui; a composition root os valida pela configuração da própria capability e os passa a ela ([`composition.md`](composition.md)).

Pontos de variação:

| Componente | Backends |
|---|---|
| `ingestion.source_adapter` | `ros1_bag`, `ros2_bag` |
| `visual_perception.region_discovery` | `sam2`, `sam3`, `florence2` |
| `visual_perception.dense_features` | `dinov2`, `dinov3` |
| `visual_perception.region_features` | `clip`, `alphaclip` |
| `visual_perception.semantic_interpretation` | `qwen`, `gemini`, `florence2` |
| `state_estimation.estimator` | `external_pose`, `fast_lio` |
| `point_representation.encoder` | `geometric_descriptor`, `ptv3` |
| `semantic_fusion.support` | `geometry-jaccard-support-v1` |
| `semantic_fusion.accumulation` | `baseline-evidence-accumulation-v1`, `quality-aware-evidence-accumulation-v1` |
| `entity_resolution.retrieval`, `.resolution`, `.geometry_comparison` | políticas versionadas, obrigatórias |
| `entity_resolution.semantic_compatibility`, `.temporal_compatibility`, `.appearance`, `.representation` | políticas versionadas, **opcionais**: sem backend selecionado, o canal é `None`, nunca um padrão |
| `spatial_relations.frame_conventions`, `.candidate`, `.geometry_summary` | políticas versionadas, obrigatórias |
| `spatial_relations.geometric_predicate`, `.contact_predicate` | políticas versionadas, **opcionais**: sem backend selecionado, o avaliador é `None` |

Para uma política, o identificador de backend é a identidade que a própria capability já versiona.

Estágios de `canonical/1`: `ingestion`, `visual_perception`, `state_estimation`, `geometric_mapping`, `sensor_association`, `point_representation` (opcional, desligado por padrão), `semantic_fusion`, `semantic_mapping`, `entity_resolution`, `spatial_relations`. O preset termina em `spatial_relations`; `semantic_mapping` participa da topologia mas não tem componente nem executor automático (seu artifact precisa ser suprido). Só `context_map` (a montagem do `ContextMapArtifact`) continua fora e entra em um preset versionado posterior, quando a capability existir.

## Lacunas conhecidas

- **Perfil sem backends.** Escolher SAM3, DINOv3, Qwen/Gemini ou FAST-LIO como canônicos é uma decisão científica que pertence à validação end-to-end (milestone #19), não ao runtime. Até lá, um experimento fornece um arquivo de configuração que seleciona os backends.
- **Parâmetros por capability.** A validação dos parâmetros de cada backend (obrigatórios, tipos, faixas) acontece em `compose()`, instanciando a configuração da própria capability (`build_config()`); a resolução da configuração só garante valores JSON finitos e sem aparência de segredo. Políticas que hoje são apenas parâmetros (sincronização, voxelização, oclusão) ganham ponto de variação quando um executor de estágio as consumir.
- **Dois `canonical/1`.** O `canonical/1` do runtime é o preset de topologia global. O `CANONICAL_PRESET_V1` de Visual Perception é o preset **interno** da percepção, com identidade própria, e continua conservando temporariamente os estágios legados de cena/região até a política de construção de `SemanticInterpretationRequest` ser promovida para a topologia default. O runtime não altera esse preset: a composition root entrega os backends atrás dos ports e não monta o preset interno; a lacuna segue registrada em [`composition.md`](composition.md) e em [`docs/runtime-composition.md`](../../../../docs/runtime-composition.md).
- **`context_map`.** `pipeline.stages.context_map` é recusado como estágio desconhecido: `canonical/1` não o declara ainda; é trabalho de outro milestone (End-to-End Validation).
