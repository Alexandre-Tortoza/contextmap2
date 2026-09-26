# Experimentos, matrizes de ablação e execução

Comparar modelos, políticas, canais de evidência ou estágios opcionais do DAG só faz sentido quando **tudo, exceto a variável declarada, é idêntico**. Este módulo define o manifesto de experimento (`contextmap.evaluation.experiments`), a execução controlada (`contextmap.evaluation.experiment_runner`) e o esqueleto comum dos drivers de experimento sobre dados reais (`contextmap.evaluation.experiment_driver`, ver [Drivers de experimento e retenção dos braços](#drivers-de-experimento-e-retenção-dos-braços)).

Resolver uma configuração em um DAG é responsabilidade do `runtime`, que ainda não existe. A avaliação **recebe topologias já resolvidas** e apenas verifica que elas diferem só como declarado; o executor de cada arm é injetado (`ArmExecutor`).

## Manifesto

`ExperimentManifest` (schema `contextmap.experiment/v1`) é imutável, versionado e hasheado, e amarra:

| Campo | Conteúdo |
|---|---|
| `selection` | reference set (`id`, `version`, `digest`), scheme, split e a **lista ordenada exata** de amostras |
| `evaluated_stage` | o estágio cuja qualidade todo relatório mede |
| `base_configuration_digest` | digest da configuração efetiva base/default |
| `variables` | as variáveis sob teste, com tipo, estágios que podem tocar, valores e baseline |
| `mode` e `arms` | a matriz de ablação e, para cada célula, a **topologia resolvida** (estágios, backend, modelo, digest de configuração, dependências) |
| artefatos upstream | por estágio, o `ArtifactIdentity` (com digest `sha256:<64 hex>` válido) **reaproveitado em vez de recalculado** |
| `fixed_controls` | valores constantes entre arms (versão de prompt, seed, variante de evidência, …) |
| `quality_metrics` / `resource_capture` | métricas de qualidade e de performance exigidas em todo relatório (política de captura de runtime, memória, storage e throughput) |
| `registry` | identidade do registro de métricas ([`metrics.md`](metrics.md)) |
| `repetitions_per_sample` | inferências repetidas por amostra: evidência correlacionada, **nunca** novas amostras físicas |

## Só as variáveis declaradas variam

O manifesto só se constrói se for uma comparação controlada:

1. **Matriz.** Os arms são exatamente as células de `ablation_cells()`: `one_at_a_time` (baseline mais um arm por variável e valor não baseline) ou `full_factorial` (todas as combinações). O baseline segura todos os valores baseline.
2. **Diferenças declaradas.** Cada diferença entre um arm e o baseline precisa ser coberta por uma variável que o arm atribui a um valor não baseline, **de um tipo que a permita**:

   | Diferença | Tipo de variável exigido |
   |---|---|
   | estágio inserido, removido ou dependência religada | `topology` |
   | backend ou modelo de um estágio | `backend` |
   | digest de configuração | `backend`, `policy`, `configuration` ou `evidence_channels` |
   | artifact pinado de um estágio | qualquer variável que toque o estágio |
   | capability de um estágio | nunca (troque removendo e inserindo) |

   Uma mudança fora dessas regras é `undeclared change`. Uma variável com valor não baseline que **não muda nenhum** dos estágios que toca também é recusada.
3. **Entrada imutável.** Toda dependência upstream de um estágio tocado por uma variável é um **artifact pinado**, então todos os arms a compartilham comprovadamente. Um arm não pode trocar o artifact pinado de um estágio não tocado.

`ExperimentVariable.kind` cobre os tipos de variação pedidos: backends (Qwen × Gemini, backends de Region Discovery, CLIP × AlphaCLIP, Point Representation off × geométrico × PTv3), políticas (fusão uniforme × ciente de qualidade), subconjuntos de canais (Semantic Fusion e Entity Resolution) e topologia.

### Ablação de estágio do DAG

```text
Arm baseline:  DINO artifact X  ->  Sensor Association
Arm enhanced:  DINO artifact X  ->  FeatureResolutionEnhancement  ->  Sensor Association
```

A variável `feature_resolution` (`topology`) toca `feature_resolution_enhancement` e `sensor_association`. O manifesto só é válido se o estágio DINO estiver pinado com o **mesmo** artifact X nos dois arms; a comparação registra o artifact compartilhado.

## Validação contra reference set e registro

`validate_experiment_manifest()` confere:

- a seleção cita a versão/digest do reference set e **reproduz** exatamente a lista ordenada do split; o papel do split declarado confere;
- um experimento de **tuning nunca usa o split held-out** (`test`): hiperparâmetros não são ajustados em dados de avaliação retidos;
- o manifesto cita o registro de métricas usado; cada métrica existe, é de qualidade (ou de performance, em `resource_capture`), pertence ao `evaluated_stage` e o reference set tem os schemas de anotação que ela exige (`MetricCompatibilityError` caso contrário).

## Execução

`run_experiment(manifest, executor=…, registry=…, reference_set=…, root=…)`:

- exige um `ValidatedReferenceSet` ([`reference-integrity.md`](reference-integrity.md)): a integridade do reference set é pré-requisito, e valida o manifesto **antes** de executar qualquer arm;
- exige um `root` novo: reexecutar cria outra identidade, nunca sobrescreve;
- executa o baseline primeiro, depois os demais; um arm que falha **não interrompe** os outros.

```text
<root>/
├── experiment.json            # o manifesto, com digest
├── arms/<arm_id>/run.json     # run manifest do arm
├── arms/<arm_id>/report.json  # relatório no envelope comum (só arms concluídos)
└── comparison.json            # manifesto de comparação
```

O **run manifest** de cada arm preserva a topologia resolvida, backend, modelo, digest de configuração, os artifacts por estágio, a identidade do reference set e do registro, e o resultado. Não há timestamps: a mesma entrada produz os mesmos bytes.

### Arms que falham ou ficam indisponíveis

| Situação | Registro |
|---|---|
| o executor levanta `ArmUnavailableError` (backend, modelo ou API externa indisponível) | `unavailable`; **nenhum fallback** para outro backend |
| o executor levanta outro erro | `failed`, com tipo e mensagem |
| o resultado contradiz o manifesto | `failed` com `invalid_result` e o motivo |

Um resultado é inválido quando: avalia outro estágio; cita outro reference set ou registro; seu digest de configuração não é o da topologia do arm; não cita este manifesto exato entre as entradas; é inconsistente com o registro; não traz uma métrica de qualidade ou de recursos declarada; não reporta o artifact de cada estágio; ou **não reaproveitou um artifact pinado**.

Um arm que não concluiu não tem métricas, aparece na comparação com `comparable: false` e torna a comparação **incompleta** (`complete: false`). `require_complete_comparison()` recusa uma comparação incompleta.

## Comparação

O `ComparisonManifest` lista:

- os arms (assignments, digest da topologia, status, falha), as variáveis e os controles fixos;
- os **artifacts compartilhados**: os que todo arm concluído reporta idênticos para o mesmo estágio (a prova de que só o declarado difere; em uma ablação de DAG, o artifact upstream compartilhado);
- cada métrica declarada **lado a lado**, por estrato: valor, status (`value`/`not_applicable`/`unsupported`), `sample_count` e a diferença **para o baseline na mesma métrica**; `same_population` indica se todos os valores vieram do mesmo número de amostras e é `false` quando algum valor não informa `sample_count`, porque população desconhecida não é evidência de população comum;
- `physical_sample_count` (amostras físicas distintas) e `repetitions_per_sample`, separados.

**Não há score geral nem vencedor.** Métricas distintas nunca são agregadas, qualidade e performance ficam em `kind` diferentes, e nada ranqueia os arms; a decisão de manter, adiar ou promover uma configuração é humana e explícita.

## Drivers de experimento e retenção dos braços

Uma execução real sobre dados reais é feita por **drivers**: scripts de pesquisa sob `experiments/<experimento>/` que montam à mão o `StageRequest` de um estágio, ou de um braço de um estágio, e chamam o executor. Cada rodada copiava os drivers e editava neles as mesmas três coisas: o número do run embutido no diretório de saída, nas `location` dos `ArtifactRef` e no `config_digest`; o caminho do workspace; e o runner que mede tempo e memória de um processo filho (achados EV-05 e EV-06 da auditoria 2026-09, issue #620). `contextmap.evaluation.experiment_driver` é a versão única e testada dessas três coisas, e só delas. O executor, as políticas e as entradas de cada estágio continuam no driver, porque são o assunto de cada experimento. Não há framework, registry nem plugin.

As cópias históricas em `experiments/*/scripts*` e os reports existentes ficam **intactos**, como evidência do que cada rodada executou. Só experimentos novos usam o esqueleto.

### Layout de um run: `DriverRun`

| Valor | Para `DriverRun(experiment="e2e-real", run_number=3, outputs_root=R)` |
|---|---|
| `run_id` | `run-0003` |
| `output_dir(nome)` | `R/e2e-real/run-0003/<nome>`, o diretório que o writer da capability cria e finaliza |
| `location(nome)` | `e2e-real/run-0003/<nome>`, relativo a `R`: a `location` do `ArtifactRef` e o `run_dir` que o report registra |
| `config_digest(nome)` | `e2e-real/run-0003/<nome>` |

`<nome>` é o id do estágio ou, quando vários braços do mesmo estágio rodam lado a lado, o id do braço. O layout é o da runtime (`<workspace>/<dataset>/run-NNNN/<estágio>`, ver [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md)) com `R` como workspace: `StageRequest.run_number()` lê o número de volta do `output_dir`, e `StageRequest.directory_of()` abre a `location`. O experimento, o nome e o número do run são validados: um nome com separador, `.` ou `..`, um número que não é inteiro positivo e uma raiz relativa são recusados.

Uma rodada nova muda **só `run_number`**. O `config_digest` é um rótulo, porque um driver montado à mão não tem documento de configuração efetiva para hashear. Ele depende do run, para que uma reexecução tenha outra identidade. Depende também do nome do artifact porque a identidade de execução da runtime é estágio + `config_digest` + hashes das entradas: dois braços do mesmo estágio sobre as mesmas entradas, diferentes só na configuração, teriam a mesma identidade com um rótulo por run. Os drivers do run real canônico usavam um rótulo por run, o que bastava lá porque cada estágio era diferente.

Esboço de um driver novo:

```python
from pathlib import Path

from contextmap.evaluation import DriverRun
from contextmap.runtime import ArtifactRef, StageRequest

RUN = DriverRun.for_driver(Path(__file__), experiment="e2e-real", run_number=6)

request = StageRequest(
    stage_id="sensor_association",
    inputs={
        "trajectory": (
            ArtifactRef(
                stage_id="state_estimation",
                contract="StateEstimationRunArtifact",
                artifact_id=TRAJECTORY_ARTIFACT_ID,
                content_hash=TRAJECTORY_CONTENT_HASH,
                location=RUN.location("state_estimation"),
            ),
        ),
        # ... as demais entradas do estágio
    },
    components={},
    config_digest=RUN.config_digest("sensor_association"),
    output_dir=RUN.output_dir("sensor_association"),
    workspace=RUN.outputs_root,
)
```

### Retenção dos braços sob `outputs/`

Os artifacts dos braços de um experimento são **retidos**. Sem eles, o experimento pode ser reexecutado, mas não reinspecionado: auditar as associações de um braço custaria a execução inteira de novo. Foi o que aconteceu com `experiments/sensor-association-scaling-20260925/`, cujo `report.json` aponta para diretórios temporários que não existem mais.

A convenção é a do run real canônico (`experiments/e2e-real-canonical-run-20260923/`):

- **Onde.** Sob o `outputs/` do checkout que contém o driver, a raiz padrão de `DriverRun.for_driver()`. O checkout é reconhecido pelo layout, nunca por um caminho absoluto de máquina: é o pai do ancestral mais próximo do driver chamado `experiments`, desde que esse pai tenha um `pyproject.toml`. Um driver fora disso não tem raiz padrão e precisa de uma explícita.
- **Diretório temporário só com raiz explícita.** `outputs_root=` aceita outra raiz, por exemplo um diretório temporário para um smoke run. Nada escolhe um diretório temporário implicitamente.
- **O que o report registra.** O `run_dir` de cada braço é `run.location(<braço>)`, relativo ao `outputs/`. Ele continua válido em outra máquina que tenha os artifacts no mesmo lugar e não expõe caminho pessoal.
- **Tempo de vida.** `outputs/` é ignorado pelo Git: os artifacts não são versionados nem entram no bundle leve do experimento (ver `experiments/` em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md)) e duram enquanto o `outputs/` daquele checkout durar. Um *worktree* do Git é outro checkout, e o `outputs/` dele some junto com ele. Um experimento cujos braços precisam ser retidos roda do checkout principal ou recebe explicitamente o `outputs/` dele.
- **Contrato.** Nenhum contrato público de artifact muda: cada braço é um artifact comum do seu estágio, finalizado pelo writer da capability.

### Medição por processo filho: `measure_child_process()`

`measure_child_process(command)` executa um comando e devolve um `ChildProcessMeasurement` com o código de saída, o wall time (relógio monotônico, incluindo a partida do interpretador do filho), o pico de RSS em KiB e a saída capturada. Um filho que falha é medido, não levantado. Um comando vazio, um comando que nem começa e uma plataforma sem unidade conhecida são erro (`ExperimentDriverError`).

O pico vem de `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`, o método do perfil de recursos do #181. Esse valor é um **máximo corrente sobre todos os filhos que um processo já esperou**: lido de um pai de vida longa, ele atribuiria a cada estágio ou braço o pico de todos os anteriores. Por isso a função nunca o lê no processo de quem chama. Um processo de medição novo executa o comando como **único** filho e reporta a leitura, e esse processo recusa medir se já tiver esperado qualquer outro filho. Chamar a função várias vezes do mesmo pai, como faz um orquestrador de braços, é seguro.

- O pico é o do **maior processo isolado** entre o filho e os descendentes que ele esperou, não a soma de uma árvore de processos.
- `ru_maxrss` é KiB só no Linux (no macOS é byte). Em outra plataforma a função recusa em vez de inflar o pico 1024 vezes.
- `to_record()` devolve exatamente os campos que os dois runners anteriores registravam: `returncode`, `wall_time_seconds` com 3 casas e `peak_rss_mb` = KiB / 1024 com 1 casa. Os números novos continuam comparáveis com os do #181 e do #564.
- O comando herda o ambiente e o diretório de trabalho. Para rodar um braço sob o `src/` de outra revisão, a variável vai no próprio comando (`["env", "PYTHONPATH=<src da revisão>", python, "arm.py", ...]`): `env` se substitui pelo alvo, que continua sendo o filho medido.

## Limitações e lacunas

- **Sem `PipelineConfig`/DAG do runtime.** A topologia é a resolvida que se fornece; quando o runtime existir (milestone 17), o manifesto passa a citar a configuração e o DAG que ele resolve, sem mudar as regras de verificação.
- **Sem executor real.** O executor é injetado. A conexão com o runtime, com o reaproveitamento de artifacts imutáveis e com os backends reais virá com essa milestone.
- O registro de métricas v1 não tem uma métrica de custo de modelo externo; ela entra com uma nova versão do registro quando for necessária.
