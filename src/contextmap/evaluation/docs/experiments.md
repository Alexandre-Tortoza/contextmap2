# Experimentos, matrizes de ablação e execução

Comparar modelos, políticas, canais de evidência ou estágios opcionais do DAG só faz sentido quando **tudo, exceto a variável declarada, é idêntico**. Este módulo define o manifesto de experimento (`contextmap.evaluation.experiments`) e a execução controlada (`contextmap.evaluation.experiment_runner`).

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

## Limitações e lacunas

- **Sem `PipelineConfig`/DAG do runtime.** A topologia é a resolvida que se fornece; quando o runtime existir (milestone 17), o manifesto passa a citar a configuração e o DAG que ele resolve, sem mudar as regras de verificação.
- **Sem executor real.** O executor é injetado. A conexão com o runtime, com o reaproveitamento de artifacts imutáveis e com os backends reais virá com essa milestone.
- O registro de métricas v1 não tem uma métrica de custo de modelo externo; ela entra com uma nova versão do registro quando for necessária.
