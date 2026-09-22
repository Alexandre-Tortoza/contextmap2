# API pública do runtime

Uma CLI, uma TUI ou qualquer outro frontend precisa dos mesmos poucos casos de uso: **descobrir** o que o runtime sabe fazer, **resolver** configuração e topologia, **fazer preflight**, **executar** e **inspecionar** o que foi executado. `contextmap.runtime.Runtime` é essa superfície única. Ela é uma **fachada de aplicação**, não um novo motor de workflow: cada operação delega ao serviço que a possui.

```text
CLI / TUI
   -> contextmap.runtime.Runtime            (esta API)
      -> config / composição / pipeline / reuso / seleção / lifecycle
```

Um frontend importa **somente** de `contextmap.runtime` e nunca precisa conhecer os módulos internos, nenhum backend concreto e nenhuma capability.

## Operações

| Operação | O que faz | Delega a |
|---|---|---|
| `status()` | versão, versões de schema, perfis conhecidos, workspace, executores **injetados** na construção e verificador ligados | catálogo, schemas |
| `capabilities()` | cada estágio, seus pontos de variação e se cada backend pode ser usado **aqui**, com o motivo | catálogo + checagem de disponibilidade |
| `resolve_config(profile, files, overrides)` | configuração efetiva, com digest e camadas | `resolve_effective_config` |
| `resolve_plan(config, targets, provided, catalog)` | topologia, ordem, backends, entradas/saídas, escopo e edições permitidas | `resolve_plan`, `scope`, seleção |
| `preflight(config, …, reuse)` | todos os problemas de uma vez, avisos, identidades e previsão de reuso | `compose_executors`, `preflight`, `predict_reuse` |
| `run(config, …, reuse, resume, events, cancellation)` | executa o pipeline ou um subgrafo e persiste o run | `compose_executors`, `RunJournal`, `run_plan`, `resume_plan` |
| `reuse_policy(index, code_identity, force)` | política de reuso sobre um índice | `ReusePolicy`, `FileArtifactStore` |
| `list_runs()` / `inspect_run(run)` | lista e detalhe **lidos do registro persistido** | `read_run` |
| `ingestion(config)` | o serviço público de ingestion com o adapter composto | composition root, `IngestionService` |

Um `Runtime` é um objeto comum: recebe `workspace`, `executors`, `verifier`, `environ`, `module_available` e `clock`. **Não há registro global nem service locator**: dois `Runtime` não compartilham estado.

## Contratos retornados

Todos são dataclasses imutáveis com `to_document()` (JSON-compatível), sem classe de backend, objeto ROS ou tipo de biblioteca de UI.

| Contrato | Conteúdo |
|---|---|
| `RuntimeStatus` | identidade e ligação do runtime |
| `RuntimeCapability` → `RuntimeComponent` → `RuntimeBackend` | estágio, implementado ou não (e por quê), backends com `available` e `reasons` |
| `ResolvedPipelinePlan` (`RuntimePlanStage`, `RuntimePlanInput`, `RuntimeEdit`) | topologia, digests, escopo, seleção resolvida, problemas estruturais e edições |
| `RuntimePreflightReport` | `ok`, `problems`, `warnings`, identidades, `predicted_reuse`, `missing_executors` |
| `RuntimeExecutionEvent` | o `ExecutionEvent` do ciclo de vida (mesmo tipo dos eventos persistidos) |
| `RuntimeExecutionResult` | envolve o `RuntimeRunRecord` persistido; `status`, `ok`, `run_id`, `event_errors` |
| `RuntimeRunRecord` (`RuntimeRunStage`), `RuntimeRunSummary` | o run lido de volta: estado, linhagem exata, decisões, falha, eventos e notas |

## Descoberta sem carregar modelo

`capabilities()` consulta o **metadado de instalação** (`importlib.util.find_spec`) e o ambiente. **Nenhum módulo opcional é importado e nenhum modelo é carregado.** Um backend que não pode ser usado diz por quê: o módulo ausente com a dica de instalação, ou o segredo ausente **pelo nome, nunca pelo valor**. É a mesma checagem que o preflight usa, então descoberta e preflight não divergem.

O teste `test_discovery_and_inspection_import_no_optional_sdk` prova isso com pacotes falsos (`torch`, `transformers`, `PIL`, `rosbags`, `alpha_clip`) que gravam um arquivo sentinela **se forem executados**: a descoberta os encontra (`dinov2`, `alphaclip` e `ros1_bag` aparecem disponíveis) e nenhum sentinela é criado.

## Configuração e topologia

`resolve_config` devolve a `EffectiveConfig` da #161 (digest versionado, camadas perfil < arquivos < overrides). O `workspace` do `Runtime` vira `resources.workspace`, **exatamente como o `--workspace` da CLI**, e um override explícito vence. Por isso, o mesmo pedido pela CLI e por esta API tem o mesmo digest.

`resolve_plan` devolve a topologia já resolvida: ordem de execução, entradas tipadas ligadas ao produtor, saída, backend de cada ponto de variação, `optional`/`available`, `config_digest` do estágio, o que está no escopo da execução, o que foi fornecido, os estágios opcionais desligados e a seleção resolvida (com a origem de cada run e sua linhagem). O frontend **não infere o DAG da documentação nem do filesystem**: ele o lê.

`ResolvedPipelinePlan.editable` lista as **edições permitidas** como `RuntimeEdit(path, kind, allowed, current)`. `path` é um override válido: `f"{edit.path}={json.dumps(valor)}"` passado a `resolve_config` aplica a edição (um teste confere que reaplicar o valor `current` de cada edição reproduz o mesmo digest).

Problemas estruturais (ciclo, contrato incompatível, alvo desconhecido, seleção sem catálogo) **voltam no plano**, não como exceção.

## Preflight

`preflight` roda as mesmas checagens do início de `run` (topologia, contratos, seleções, seleção de backend, módulos opcionais, segredos, executores) e reporta **todos os problemas de uma vez**, cada um com o caminho. Não importa nem consulta nada além do que o metadado já diz, então é barato o bastante para uso interativo. Além dos problemas:

- `warnings`: o que não bloqueia, como a ausência de workspace e cada `latest` que foi resolvido (com o artifact escolhido);
- `predicted_reuse`, com uma política de reuso: o que seria reaproveitado ou recomputado, sem executar nada. O registro do run continua sendo a autoridade (a previsão é conservadora);
- `missing_executors`: só os estágios que **rodariam** e não têm executor, contando tanto os compostos automaticamente da configuração (`compose_executors`, ver [`composition.md`](composition.md)) quanto os injetados na construção do `Runtime`; um estágio que certamente será reaproveitado não precisa dele.

Estágio inexistente ou de capability ainda não implementada é **explícito**: `stages.context_map: the artifact capability is not implemented yet` (para um preset que declare um estágio assim); backend ou estágio desconhecido é recusado já na resolução.

## Execução

`run` percorre o caminho canônico: preflight, o executor do DAG, com o journal do run. **Todo desfecho esperado volta como resultado** lido do registro persistido: `completed`, `failed`, `blocked` e `cancelled`. Só levantam exceção o **uso incorreto** (sem workspace, `resume` sem política de reuso, workspace da configuração diferente do do runtime, `provided` junto com seleções configuradas, retomada impossível, que **não cria run**) e um **bug inesperado**. `KeyboardInterrupt` é registrado como cancelamento e propagado.

- **Eventos.** O consumidor recebe cada `ExecutionEvent` em ordem (`sequence` 1..n), **depois** de persistido. É o mesmo evento do journal, então o que a tela viu e o que ficou no log são iguais.
- **O consumidor não altera a ciência.** Se o sink levantar, o erro é registrado em `RuntimeExecutionResult.event_errors` (com segredos redigidos) e o run **continua**: uma tela quebrada não vira "falha inesperada do pipeline".
- **Cancelamento** cooperativo entre estágios com `CancellationToken`.
- **Segredos.** Por padrão, os segredos que os backends selecionados declaram são redigidos de eventos, falhas e status; nunca aparecem em resultado, evento ou arquivo.
- **Reuso e retomada.** `reuse=` reaproveita por identidade; `resume=` retoma um run falho, cancelado ou interrompido como um run **novo**, reaproveitando só o que passa nas checagens normais.
- **Sem substituição silenciosa.** Nada é executado no lugar de um estágio indisponível ou de um backend ausente.

## Inspeção de runs: a linhagem persistida é a autoridade

`list_runs()` lista `<dataset>/run-NNNN` ordenando por dataset e depois em ordem **numérica** (não lexicográfica), inclusive um registro ilegível, que aparece com o motivo em vez de sumir; cada `RuntimeRunSummary` traz o `dataset`, porque `run-0001` se repete entre datasets. `inspect_run(run)` aceita um id sob algum dataset do workspace (um id presente em mais de um dataset é ambíguo e é recusado: passe o diretório) ou um diretório, e devolve o que o registro contém:

- estado, `interrupted`, digests, backends por ponto de variação (da configuração persistida), alvos, artifacts fornecidos;
- por estágio: `outcome` (`pending`, `started`, `reused`, `completed`, `failed`), **entradas exatas** (ids), saída, **decisão de reuso** (com o artifact anterior exato) e tempo;
- falha (categoria, estágio, tipo), problemas de bloqueio, identidade de código, ambiente e os eventos.

**O que o registro não tem, a API não deduz.** As entradas de um estágio vêm do evento `stage_started` ou do `execution.json`; um estágio reaproveitado por um run que não concluiu não tem entradas registradas, então `inputs` é `None`. `backends` é `None` se a configuração persistida faltar. Cada ausência vira uma **nota** em `notes`. Não há "preenchimento" por inferência.

## Fronteiras

- Nenhuma classe de backend (SAM, DINO, VLM, FAST-LIO etc.), objeto ROS ou biblioteca de UI (Textual, Typer, Click, Rich) na API pública ou no núcleo.
- `api.py` importa só a biblioteca padrão e módulos do próprio runtime: nunca uma capability, um backend ou um adaptador (`tests/architecture/test_runtime_boundaries.py`).
- Capabilities nunca importam o runtime (`tests/architecture/test_boundaries.py`).
- Importar o runtime e usar a descoberta, o plano e o preflight não carrega SDK pesado.

## Métricas medidas

Medidas neste repositório (uma execução, mediana de 20 repetições, com o mundo falso de testes; não são um benchmark):

| Operação | Latência |
|---|---|
| `status()` | ~0,001 ms |
| `capabilities()` | ~0,6 ms |
| `resolve_config()` | ~0,3 ms |
| `resolve_plan()` | ~0,25 ms |
| `preflight()` | ~0,5 ms |
| `list_runs()` (1 run) | ~0,1 ms |
| `inspect_run()` | ~0,4 ms |

Módulos pesados carregados pela descoberta e pela inspeção: **0**. Os testes usam limites folgados (2 s) apenas como proteção contra regressões grosseiras, não como medida.

## Como usar

```python
from contextmap.runtime import Runtime

runtime = Runtime(workspace="ws", executors=executors, verifier=verify)

for capability in runtime.capabilities():
    print(capability.stage_id, capability.implemented, capability.reason)

config = runtime.resolve_config(files=["experiment.json"], overrides=["resources.device=cuda"])
plan = runtime.resolve_plan(config, targets=["semantic_fusion"])

report = runtime.preflight(config, targets=list(plan.run_stages))
if report.ok:
    result = runtime.run(config, targets=list(plan.run_stages), events=my_sink)
    print(result.status, result.record.stages)

for summary in runtime.list_runs():
    print(summary.run_id, summary.status)
```

`verifier` vem de quem possui os estágios: o runtime não sabe, sozinho, se um artifact indexado ainda existe. `executors` já não é a única fonte de executores: `preflight` e `run` mesclam o que foi passado aqui com o que [`compose_executors`](composition.md) consegue montar da própria `config` de cada chamada — o injetado aqui sempre vence, então `executors=` continua servindo para testes e para substituir ou completar um estágio que a composição não sabe montar sozinha (por exemplo `ingestion`).

## Lacunas conhecidas

- **A CLI ainda não usa esta fachada.** `contextmap` chama diretamente as mesmas funções (`resolve_effective_config`, `run_plan`, `RunJournal`…), então não há uma segunda especificação do pipeline, mas há duas costuras de código — ambas mesclam `compose_executors` com o injetado, do mesmo jeito. Migrar a CLI para `Runtime` é trabalho futuro; o comportamento já é equivalente (mesmo digest, mesmo registro).
- **Não há catálogo sobre os índices reais de run das capabilities.** `catalog=` recebe um `ArtifactCatalog` (por exemplo `StaticCatalog`/`load_catalog`); reconstruí-lo a partir dos índices exige leitores das capabilities, que o runtime não importa.
- **Descoberta por perfil.** Só o perfil `canonical/1` existe; `capabilities(profile=...)` já aceita outros.
- **Um run em andamento.** `inspect_run` de um run `running` mostra o que já foi persistido; o acompanhamento ao vivo é pelo sink de eventos, não por leitura repetida.
- **`ingestion`, `visual_perception`, `point_representation` e `semantic_mapping` sem composição automática.** `compose_executors` monta `state_estimation`, `geometric_mapping`, `sensor_association`, `semantic_fusion`, `entity_resolution` e `spatial_relations` da configuração; `ingestion` precisa de um `IngestionRequest` que não é parte de nenhuma configuração (injete um `IngestionStageExecutor`, ou rode `contextmap ingest` e forneça/selecione o artifact publicado), `visual_perception`/`point_representation` ainda não têm executor real (dependem de modelo/GPU), e `semantic_mapping` não tem componente de catálogo nem executor automático (seu artifact precisa ser suprido) — os testes desta fachada usam estágios falsos para essas quatro.
