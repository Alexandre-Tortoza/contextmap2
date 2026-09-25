# StateEstimationRunArtifact

Este documento descreve `src/contextmap/state_estimation/run_artifact.py`. As regras gerais de artifacts (imutabilidade, atomicidade, inventário, índice de run) estão em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) e são implementadas uma única vez em `contextmap.shared.run_directory`.

## Por que existe

Quando um mapa ou uma associação câmera-LiDAR sai errada, é preciso descobrir se a causa está na qualidade da trajetória, na semântica de frames, no lookup de timestamps, na configuração do estimador, nos pré-requisitos de calibração ou em um transform downstream. O run persiste a trajetória canônica junto com métricas legíveis por máquina e evidência de auditoria, e abre sem ROS, sem FAST-LIO e sem NumPy.

## Layout

O writer grava o artifact **exatamente** no `output_dir` que o chamador entrega; ele não calcula caminho, não aloca índice e não mantém registro. No runtime, `output_dir` é `<workspace>/<dataset>/<run>/state_estimation/` ([`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md)).

```text
<output_dir>/
├── README.md
├── manifest.json
├── outputs/                           # contratual
│   ├── trajectory.json                # metadados da trajetória (sem poses)
│   ├── poses.jsonl                    # uma pose por linha
│   ├── pose-index.jsonl               # id, timestamp_ns, offset e tamanho de cada pose
│   ├── frame-summary.json             # frames dinâmicos, frames/arestas estáticas, convenções
│   └── quality.json                   # amostragem, gaps, contagens
├── metrics/                           # contratual
│   ├── preflight.json                 # relatório completo do preflight de geometria
│   ├── motion.json                    # distribuições de translação, rotação e velocidades
│   ├── runtime.json                   # somente quando o tempo foi medido
│   └── diagnostics.jsonl              # eventos do backend, somente quando existem
└── debug/                             # nunca contratual
```

Não existem `config.yaml`, `lineage.json`, `environment.json` nem `events.jsonl` separados: a linhagem e a configuração efetiva (`estimator.configuration_fingerprint`) ficam no `manifest.json` e os eventos em `metrics/diagnostics.jsonl`. Criar arquivos sem produtor real violaria YAGNI, o mesmo critério adotado por `PerceptionRunArtifact`.

## `manifest.json`

Identifica o run, o que ele consumiu e quem o produziu: `run_id`, `run_index`, `sequence_name`, `sequence_artifact_id`, `selection_id`, `auxiliary_sequence_artifact_id`/`auxiliary_selection_id` (issue #555: a sequência de pose auxiliar realmente incorporada na trajetória, ou `None` quando nenhuma contribuiu — inclusive quando uma foi configurada, mas descartada pela salvaguarda de ground truth), `trajectory_id`, `estimator` (`backend_id`, `backend_version`, `configuration_fingerprint`), `calibration_identity`, `code_version`, frames dinâmicos (`reference_frame`, `body_frame`), `clock_id`, limites de tempo, contagens (poses, observações consumidas/rejeitadas, gaps), `diagnostic_counts` por código, `preflight_status`, `debug_level`, a semântica de `interpolation` usada por `TrajectoryLookup`, `schema_version` e `created_at`. `file_inventory` lista cada arquivo contratual com tamanho e SHA-256, sem o manifest, o README e o `debug/`.

Os dois campos de auxiliar existem para que este artifact seja autoportável: um consumidor que só tenha o `StateEstimationRunArtifact` (sem a lineage própria da runtime) ainda consegue abrir a sequência de pose auxiliar nomeada aqui e fechar a proveniência de cada pose que veio dela (ver `runtime/docs/composition.md`, seção da bridge do #555).

Um run cujo preflight de geometria estava `BLOCKED` nunca é persistido: o writer recusa.

## Acesso a uma pose

`StateEstimationRunReader` abre um run somente pelo seu diretório. `pose(estimate_id)` e `pose_at(timestamp)` usam `pose-index.jsonl` para ler apenas os bytes daquela pose: não carregam as demais nem qualquer arquivo de debug. `pose_at` exige o mesmo domínio de clock (`ClockDomainMismatchError` caso contrário) e devolve `None` quando nenhuma pose tem exatamente aquele timestamp; para pose mais próxima ou interpolada use `TrajectoryLookup(reader.trajectory())`.

`read_record()` lê JSON de `outputs/` e `metrics/` e recusa `debug/`, para que nenhum estágio downstream dependa dele por engano.

## Investigar anomalias sem reexecutar

- gaps e amostragem: `outputs/quality.json` (contagem, intervalos mínimo/mediano/máximo, `sample_rate_hz`, lista de gaps);
- saltos e paradas: `metrics/motion.json` (distribuições de deslocamento, rotação e velocidades por intervalo);
- amostras rejeitadas e avisos: `metrics/diagnostics.jsonl` e `diagnostic_counts`;
- causa de bloqueio ou de pré-requisito downstream ausente: `metrics/preflight.json`.

Qualidade e tempo de execução ficam separados: `metrics/runtime.json` não entra nas medidas de qualidade.

## Níveis de debug

| Nível | Conteúdo em `debug/` |
| --- | --- |
| `none` | nada; os outputs contratuais, a linhagem e as métricas exigidas continuam completos |
| `standard` | `pose-deltas.jsonl`, `timestamp-gaps.jsonl` (quando há gaps), `trajectory-xy.csv`, `trajectory-xz.csv` |
| `full` | acrescenta `preflight-checks.jsonl` e `backend-diagnostics/events.jsonl` |

Os projetos XY/XZ são CSV, não imagens: o artifact não exige nenhuma tecnologia de visualização. Arquivos de debug são escritos mas nunca entram no inventário, então removê-los não invalida o run.

## Integridade, imutabilidade e identidade

- a escrita acontece em um diretório temporário e o run só aparece no caminho final depois de a checagem de inventário passar; uma escrita interrompida não pode parecer um run válido;
- um run finalizado nunca é sobrescrito: o writer recusa um `output_dir` que já exista, e reexecutar grava em outro diretório;
- `verify_integrity()` detecta arquivo ausente, tamanho diferente e hash diferente; um schema desconhecido levanta `RunArtifactError` e um diretório sem manifest levanta `IncompleteRunArtifactError`.

`run_id` e `run_index` são entregues pelo chamador e gravados como recebidos; o writer nunca os aloca. O `run_index` é um ordinal legível, mas não substitui identidade nem hash.
