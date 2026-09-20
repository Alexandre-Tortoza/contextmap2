# GeometricMapArtifact

Este documento descreve `src/contextmap/geometric_mapping/run_artifact.py`.

Um `GeometricMapArtifact` é um diretório imutável e autodescritivo com o mapa que **um run** construiu a partir de um trecho selecionado de uma sequência canônica e de uma trajetória do State Estimation, junto de linhagem, configuração efetiva, métricas e evidência de debug. Ele abre **sem ROS, sem FAST-LIO, sem biblioteca de modelo e sem NumPy**, e uma `GeometryReference` resolve depois de fechar e reabrir para o mesmo XYZ autoritativo e a mesma proveniência.

## Layout

```text
workspace/runs/geometric-mapping/<sequência>/run-000N__<seleção>__<perfil>/
├── README.md
├── manifest.json
├── lineage.json
├── config.json
├── environment.json
├── outputs/
│   ├── geometry.bin           payload empacotado (ver accumulation.md)
│   ├── source-index.jsonl     um ScanRecord por scan: origem → geometria, cadeia e limites
│   └── map-metadata.json      GeometricMap: identidade, frame, limites, tempo, índice, proveniência
├── metrics/
│   ├── input-plan.json        scans selecionados/aceitos/recusados, política de correção, lookups
│   ├── mapping.json           contagens, redução, limites, tempo
│   └── runtime.json           só quando medido (tempo e memória)
└── debug/                     evidência humana; nunca inventariada, nunca uma dependência
```

`outputs/`, `metrics/`, `lineage.json`, `config.json` e `environment.json` são **contratuais**: entram no inventário do manifesto com tamanho e SHA-256, então a perda ou a alteração de qualquer um é detectada. `debug/` nunca entra: removê-lo não invalida o run, e nenhum estágio a jusante pode depender dele. O `DebugLevel.NONE` não remove nada que o Sensor Association ou etapas posteriores precisem.

A escrita usa `AtomicRunDirectory`: um run interrompido não parece um run finalizado, um run finalizado nunca é sobrescrito, e rodar de novo cria outro run. A geometria é escrita **em fluxo** (`open_binary`) e hasheada durante a escrita, então um mapa maior que a memória pode ser gravado.

## Identidade

`map_id = <sequência>--<run_id>`; toda `GeometryReference` carrega esse `map_id`. As referências continuam válidas para o mesmo artefato e são locais a ele.

## Manifesto

`manifest.json` identifica o run (`run_id`, `run_index`, `sequence_name`), o mapa (`map_id`, `map_frame`), as entradas (sequência canônica, seleção, trajetória, run do State Estimation, identidade da calibração), a configuração (`configuration_fingerprint`), o código (`code_version`), as contagens (pontos persistidos, pontos medidos antes da agregação, scans acumulados e recusados), a regra de agregação, o índice espacial, o intervalo de tempo e o inventário de arquivos contratuais.

## Linhagem, configuração e ambiente

- `lineage.json`: sequência canônica, a **seleção** (não só a identidade), o run do State Estimation, a trajetória, a identidade da calibração, os frames do mapa e do corpo, o código, as observações que entraram no mapa e as recusadas.
- `config.json`: a configuração **efetiva**: política de lookup de pose, política de scans não corrigidos, agregação, o formato do registro empacotado e o índice espacial, mais o nível de debug e o `configuration_fingerprint`. O fingerprint é o SHA-256 da configuração que muda a geometria (o nível de debug não entra) e muda com a política de lookup, a de correção, a agregação e o formato de armazenamento. É JSON, não YAML: não há dependência de YAML no projeto.
- `environment.json`: versões do Python e do NumPy (lidas dos metadados de instalação, sem importar a biblioteca), plataforma e versão do código.

## Métricas

- `metrics/input-plan.json`: scans selecionados, aceitos e recusados **com o motivo**; observações ignoradas por modalidade; lookups de pose (contagens por resultado e deltas de tempo) e os recusados; a política de correção, a contagem por estado e por decisão e os avisos.
- `metrics/mapping.json`: scans acumulados e sem geometria, pontos medidos antes da agregação e persistidos, pontos descartados por não serem finitos, razão de redução, limites, intervalo de tempo e tamanho do payload.
- `metrics/runtime.json`: tempo e memória de pico, **só** quando o chamador os mediu, e nunca misturados às métricas de qualidade.

## Leitura

`GeometricMapArtifactReader(run_dir)` lê o manifesto e abre a geometria sob demanda:

- `geometry()` devolve o `PackedGeometry` (um `GeometrySource`) sobre o payload **mapeado em memória**, sem lê-lo; `close()` (ou o `with`) libera o mapeamento;
- `read_record(caminho)` lê um JSON contratual e recusa `debug/`, o payload e qualquer caminho fora dos registros contratuais;
- `verify_integrity()` confere o inventário (arquivo ausente, tamanho, hash) e, por padrão, recalcula os limites derivados a partir da geometria e os compara com os registrados;
- `geometry().trace(reference)` reconstrói **como um ponto persistido chegou à sua posição global** (a cadeia com a pose e a calibração usadas) sem reexecutar o mapeamento.

Arquivo ausente ou payload truncado viram um `MapArtifactError` explícito. `allocate_map_run_index` e `rebuild_map_run_registry` seguem as mesmas regras do State Estimation: o índice é monotônico, calculado dos runs válidos no disco (manifesto e tamanhos, sem reler gigabytes), e o `runs.json` é só uma conveniência.

## Debug

- `none`: nada.
- `standard`: `scans-by-time.jsonl`, `trajectory-over-map.csv` (a pose usada em cada scan), `bounds-summary.json`, `warnings.jsonl` (scans recusados e avisos de correção) e `selected-transform-traces.jsonl`: traces de uma amostra determinística de pontos persistidos, **lidos de volta do payload** (5 scans, 2 pontos por scan).
- `full`: acrescenta uma amostra mais densa (até 50 scans, 3 pontos por scan) e `trace-residuals.jsonl` com o resíduo de reconstrução de cada trace.

Visualizações de debug não são geometria autoritativa e não exigem uma tecnologia de visualização: são arquivos padrão (JSONL, CSV, JSON).

## Diferenças em relação ao layout sugerido na issue

- `config.json` no lugar de `config.yaml` (sem dependência de YAML);
- sem `events.jsonl` separado: rejeições e avisos vivem em `metrics/input-plan.json` e em `debug/warnings.jsonl`;
- sem arquivos `geometry-index.*` nem `map-bounds.*`: a identidade é posicional (o índice é aritmético) e os limites estão em `map-metadata.json`; o índice de origem é `source-index.jsonl`;
- sem arquivo de índice espacial derivado separado: os limites por scan vivem no índice de origem (ver [`spatial-access.md`](spatial-access.md));
- os resíduos de alinhamento scan-mapa e a cobertura entre scans vizinhos pertencem à validação, não a este artefato: aqui há os resíduos de reconstrução dos traces.

## Limitações

- O tempo e a memória de pico são medidos por quem chama; o artefato só os registra.
- `verify_integrity()` com o índice lê o payload inteiro (cerca de 1,4 s por 3 milhões de pontos).
