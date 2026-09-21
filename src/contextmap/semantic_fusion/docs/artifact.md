# `SemanticFusionRunArtifact`

Este documento descreve `src/contextmap/semantic_fusion/run_artifact.py` e `serialization.py`. As regras gerais de artifacts (imutabilidade, atomicidade, inventário, índice de run) estão em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) e são implementadas uma única vez em `contextmap.shared.run_directory`.

Um run persiste os **suportes de fusão** e a **evidência fundida** sobre cada um, com a linhagem, as métricas e a evidência pulada. É imutável, autodescritivo e **abre sem NumPy, sem runtime de percepção e sem biblioteca de modelo**. Toda hipótese fundida volta, depois de reabrir, à evidência e à proveniência exatas de origem. Nada a montante é duplicado: claims, scores, features, qualidade e representações 3D são só referenciados, e a geometria é guardada como deltas posicionais.

## O que nunca se perde

- **Todas as hipóteses**, com alternativas, conflitos, abstenções e evidência sem score: o artifact nunca guarda só a hipótese primária.
- **Frames físicos e resultados de inferência** continuam distintos (grupos por suporte, contagens e métricas separadas).
- **Ausência explícita:** um score não pontuado (`None`) volta como `None`, nunca como zero.
- **A ponderação por qualidade**, quando existe, com fatores por componente, por contribuição e por hipótese (antes e depois).

## Layout

```text
workspace/runs/semantic-fusion/<sequence>/
├── runs.json                                   # registry reconstruível
└── run-000N__<selection>__<policy>/
    ├── README.md
    ├── manifest.json                           # identidade, linhagem, políticas e inventário
    ├── outputs/                                # contratual
    │   ├── fusion-supports.jsonl               # um FusionSupport por linha
    │   ├── fused-evidence.jsonl                # um FusedEvidence por linha (autocontido)
    │   ├── support-observation-index.jsonl     # suporte → observações, offsets nos dois arquivos
    │   ├── hypothesis-evidence-index.jsonl     # hipótese → evidência exata (claim, stance, frame, resultado, run)
    │   ├── physical-observation-groups.jsonl   # grupos por frame físico de cada suporte
    │   ├── contribution-index.jsonl            # contribuição → suporte, observação, resultado, run, claims
    │   └── excluded-observations.jsonl         # observações que não entraram em suporte algum, com o motivo
    ├── metrics/                                # contratual
    │   ├── counts.json
    │   ├── distributions.json
    │   ├── payload.json
    │   └── runtime.json                        # somente quando medido
    └── debug/                                  # nunca contratual
```

Não existem `config.yaml`, `lineage.json`, `environment.json` nem `events.jsonl` separados: a configuração efetiva e a linhagem ficam no `manifest.json`, e a evidência pulada em `excluded-observations.jsonl`. Criá-los sem produtor real violaria YAGNI (mesma decisão de Sensor Association e Point Representation).

`fused-evidence.jsonl` é a fonte autoritativa; os índices são derivados e existem para acesso preguiçoso.

## Escrita

`SemanticFusionRunWriter.write(outcomes, excluded=..., warnings=..., runtime=...)` consome os `FusionOutcome` (um suporte e a evidência fundida sobre ele) **como fluxo**, gravando cada registro assim que ele chega (`open_binary`, com hash durante a escrita), então o tamanho do run não é limitado pela memória para os arquivos grandes. O run é publicado de forma atômica ao fim do fluxo.

O writer recusa, com `FusionRunArtifactError` e sem deixar run visível, quando:

- os suportes não chegam estritamente ordenados por identidade (o que também recusa duplicatas);
- um suporte lista observações espaciais diferentes das contribuições da sua evidência;
- um suporte é sobre outro mapa que o da linhagem, ou uma contribuição vem de uma run de percepção que a linhagem não lista, ou uma referência de estrutura vem de uma run de Point Representation que ela não lista;
- suporte ou fusão usam política ou configuração diferentes do resto do run: **um run guarda uma política**;
- já existe um run no caminho (um run finalizado nunca é sobrescrito; reexecutar cria outro índice).

## Linhagem (manifest)

`FusionRunLineage` é a seleção **explícita** das runs a montante: a sequência canônica, o `GeometricMapArtifact`, as runs de Sensor Association, as runs de percepção e, se selecionadas, as de Point Representation (a seleção, não o uso: numa ablação de canais todo braço lista as mesmas). Nunca se mescla automaticamente tudo o que existe. O manifest também registra:

- as políticas: agrupamento por observação física, construção de suporte (com fingerprint) e fusão (com fingerprint);
- as identidades que alimentaram cada canal (interpretadores, scorers, espaços de embedding, versão das definições de qualidade, espaços de representação, mapa);
- contagens, avisos, o código e o inventário com hash de cada arquivo contratual.

Um run sem suportes é válido e explícito (as políticas ficam `null`).

## Métricas

Separadas, sem um escalar único:

- `counts.json`: suportes, contribuições, hipóteses, frames físicos e resultados de inferência, ambos **distintos no run inteiro** e contados à parte (um resultado de percepção com várias regiões em suportes diferentes conta uma vez; o valor por suporte está em `distributions.json`), claims (total, pontuadas, não pontuadas, sem hipótese), sinais de scorer, claims em abstenção, stances, incerteza por tipo, suportes com incerteza, observações excluídas e avisos;
- `distributions.json`: por suporte, frames físicos, resultados de inferência, contribuições e hipóteses (contagem, mínimo, mediana, máximo);
- `payload.json`: o tamanho de cada arquivo contratual;
- `runtime.json`: tempo e memória, **só** quando quem chama os mediu, e nunca misturados às métricas de qualidade.

Uma claim de um suporte sem nenhuma hipótese (só abstenções) não tem sinais persistidos em hipótese alguma; por isso aparece em `without_hypothesis`, e sua referência fica nos registros de `INSUFFICIENT_EVIDENCE`.

## Leitura

`SemanticFusionRunReader(run_dir)` abre um run só pelo seu diretório:

- `support(id)` e `fused_evidence(id)` leem um registro por deslocamento, **sem carregar os outros**;
- `iter_outcomes()` percorre tudo em ordem; `support_ids()` lista as identidades;
- `support_of_observation(id)` diz em que suporte uma observação foi acumulada (`None` se excluída);
- `contribution(id)` resolve uma contribuição às suas referências exatas a montante;
- `excluded_observations()` devolve a evidência pulada;
- `read_record()` lê JSON de `outputs/` e `metrics/` e **recusa `debug/`**;
- `verify_integrity()` confere o inventário (arquivo ausente, tamanho, hash).

Registro malformado, truncado ou adulterado vira `FusionRunArtifactError` explícito, e uma escrita adulterada em um suporte não impede ler os outros. `allocate_fusion_run_index` e `rebuild_fusion_run_registry` seguem as regras dos outros artifacts: o índice é monotônico e calculado dos runs válidos no disco, e o `runs.json` é só conveniência.

## Debug

`SemanticFusionDebugLevel.NONE` (padrão) não grava nada; **desligar o debug não remove nenhum dado que Semantic Mapping precise**.

- `standard`: para uma amostra de suportes (os primeiros 20), `debug/supports/<support-id>/` com `summary.json`, `hypotheses.json`, `conflicts.json` e `physical-observations.json`.
- `full`: o mesmo para todos os suportes, mais `contribution-trace.jsonl` e `geometry-summary.json`.

O `summary.json` responde: qual hipótese foi sustentada, por quantos frames físicos, quais resultados de inferência contribuíram, que evidência conflitou ou se absteve, e quais canais de score e feature foram usados. `debug/` nunca é inventariado nem autoritativo.

## Limitações

- O escritor mantém em memória as linhas dos índices (proporcionais ao número de observações e de evidências), não a evidência inteira.
- Cada contribuição repete a sua geometria (como deltas) além da união no suporte; o custo cresce com o número de pontos por região. Uma tabela binária compartilhada, como a de Sensor Association, é uma otimização futura sem consumidor hoje.
- Não há CLI; a composição das entradas e das runs a montante pertence ao `runtime`.
