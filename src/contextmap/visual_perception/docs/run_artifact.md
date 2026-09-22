# Artefato de run de percepção

Este documento descreve `src/contextmap/visual_perception/run_artifact.py` e `serialization.py`. Para as convenções globais de artifact/lineage/immutability, ver [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md).

## Layout do artefato

O writer grava o run **exatamente** no `output_dir` que o chamador entrega; ele não calcula caminho, não aloca índice e não mantém registro. No runtime, `output_dir` é `<workspace>/<dataset>/<run>/visual_perception/` ([`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md)).

```text
<output_dir>/
├── README.md                                          # gerado, legível por humano
├── manifest.json                                      # ponto autoritativo
├── outputs/
│   ├── results.jsonl                                   # um PerceptionResult por linha
│   ├── semantic-interpretations.jsonl                   # execução semântica auditável
│   ├── semantic-views/                                  # pixels exatos enviados ao backend
│   │   └── <view-payloads>
│   └── features/                                       # quando payloads são persistidos
│       ├── feature-index.jsonl
│       └── <observation-scope>/*.npy
├── metrics/
│   ├── stage-timings.jsonl                             # um StageOutcome por linha
│   └── feature-extraction.jsonl                        # quando há diagnóstico de feature
└── debug/
    ├── 30-feature-extraction/                           # somente standard/full
    └── 40-semantic-interpretation/<request-id>/
        ├── request.json / prompt.txt / parsed-response.json
        ├── diagnostics.json / semantic-claims.json
        └── raw-response.txt                             # somente full
```

## Fluxo de persistência e leitura

```mermaid
flowchart LR
    INPUT["PerceptionResult[] + StageOutcome[]"] --> WRITER["PerceptionRunWriter"]
    WRITER --> TMP["diretório temporário irmão"]
    TMP --> FILES["manifest.json<br/>outputs/results.jsonl<br/>outputs/semantic-interpretations.jsonl<br/>metrics/stage-timings.jsonl<br/>README.md"]
    FILES --> CHECK["checagem interna de consistência<br/>tamanho + hash + ownership"]
    CHECK -->|válido| FINAL["output_dir"]
    FINAL --> READER["PerceptionRunReader"]
    READER --> RESULT["result() / list_results()"]
```

`manifest.json` e os arquivos inventariados no próprio run formam a fonte de verdade; não existe registro nem índice ao lado do run.

`run_id` e `run_index` são entregues pelo chamador e gravados como recebidos; o writer nunca os aloca. O `run_index` é um ordinal legível do chamador (a runtime usa o número de `run-NNNN`), nunca timestamp, e não substitui identidade nem hash.

## Decisões desta issue (v0)

- **`outputs/results.jsonl`, não `outputs/results.parquet`.** Mesma decisão e mesmo motivo da issue #39 de Ingestion: nenhuma dependência de runtime nova (`pyarrow`/`pandas`) se justifica ainda; JSON Lines é inspecionável com ferramentas de texto padrão. Revisitar se o volume de resultados tornar leitura linha-a-linha um gargalo real.
- **`outputs/results.jsonl` continua canônico para metadata; `outputs/features/feature-index.jsonl` indexa apenas payloads numéricos opt-in.** Cada `PerceptionResult` carrega suas `regions`/`features`/`claims`; o feature index não duplica esse contrato, apenas liga a chave `(source_observation_id, feature_id)` ao arquivo `.npy`, hash e metadata necessária para leitura lazy. `finalize()` valida essa referência cruzada antes de publicar o artifact.
- **Views semânticas são outputs contratuais.** Cada `SemanticVisualView` possui SHA-256 obrigatório e referencia um arquivo abaixo de `outputs/semantic-views/`. `add_semantic_view_payload()` valida o hash antes de enfileirar os bytes (na persistência; a inferência já verifica o mesmo hash em cada runtime, ver [Integridade das views](semantic-interpretation.md#integridade-das-views-na-inferência)); `finalize()` exige que toda view de toda execução possua payload inventariado e rejeita payloads sem request correspondente.
- **`debug/` só existe quando há conteúdo real.** Feature Extraction
  materializa previews conforme seu nível. `SemanticDebugLevel.NONE` não grava
  diagnostics humanos, `STANDARD` grava request/prompt/parsing/final outputs e
  `FULL` acrescenta a resposta bruta. `outputs/semantic-interpretations.jsonl`,
  hashes e outputs canônicos permanecem suficientes para leitura quando debug
  está desabilitado. Campos de credencial conhecidos são redigidos antes de
  qualquer serialização.
- **`config.yaml`, `lineage.json`, `environment.json`, `events.jsonl` não são escritos no v0.** Nenhum destes tem produtor real ainda (configuração efetiva de backend, lineage de artefatos upstream, ambiente de execução, eventos granulares) — `manifest.json` já cobre a metadata mínima autoritativa (run_id, índice, sequência, seleção, capabilities, contagens). Adicionar esses arquivos vazios/parciais agora seria estrutura sem conteúdo real.

## Escrita atômica

Mesmo padrão de `contextmap.ingestion.sequence_artifact`: `PerceptionRunWriter.finalize()` escreve em um diretório temporário irmão de `output_dir` (`.tmp-<nome-de-output_dir>-<random>/`), roda uma checagem de consistência interna, e só então renomeia para `output_dir` — um run interrompido nunca aparenta ser válido. Um `output_dir` que já exista é recusado com `RunArtifactError`, sem alterar o run que está lá, e uma falha remove o temporário e não deixa nada. O writer não escreve registro nem `runs.json` e não toca em nenhum outro diretório.

`add_result()` rejeita evidência pertencente a outro `run_id`, a outro `sequence_artifact_id` ou uma segunda evidência para o mesmo `source_observation_id`. Assim, o arquivo final preserva exatamente um resultado por observação e nunca mistura ownership de runs ou sequências.

O writer também exige, antes de publicar, que cada diagnostic de feature `SUCCEEDED`/`WARNING` descreva exatamente uma feature do resultado da mesma observação (`feature_id`), com scope, shape, dtype, normalização, `payload_reference`, proveniência do backend e fingerprint do `EmbeddingSpace` iguais. Para features densas, `(grid_height, grid_width)` precisa ser `feature.shape[:2]` e `source_artifact_id` precisa ser o `run_id` do próprio run; um segundo diagnostic para a mesma feature é rejeitado. `metrics/feature-extraction.jsonl` carrega a geometria densa contratual, então ela não pode descrever outro payload. Detalhes em [`feature_diagnostics.md`](feature_diagnostics.md).

Antes de publicar, o writer também exige que cada `SemanticInterpretationExecution` resolva para exatamente um `PerceptionResult` por `perception_result_id` e observação. Todas as `parsed.claims` precisam estar materializadas nesse resultado e o `parsed.scene_context`, quando presente, precisa coincidir integralmente com o contexto persistido.

Essa reconciliação inclui inputs mesmo quando a execução abstém: `region_id`
deve resolver em `regions`; cada `visual_feature` deve coincidir com uma feature
do resultado e possuir payload no feature store; e `scene_context_reference`
deve resolver pelo `PerceptionResultId` para um contexto da mesma observação.
`claim_count` soma tanto `PerceptionResult.claims` quanto as claims aninhadas em
`SceneContext`.

## Leitura isolada

`PerceptionRunReader(run_dir)` abre um run **apenas com seu próprio diretório**. `manifest.json` e o inventário de `outputs/` fornecem os resultados e registros de execução contratuais; `debug/` não é dependência de leitura.

## `serialization.py`

Funções `encode_x`/`decode_x` simétricas para cada tipo de `models.py` (`BackendProvenance`, `BoundingBox2D`, `Region2D`, `VisualFeature`, `SemanticClaim`, `SceneContext`, `PerceptionResult`). Reaproveitadas por `run_artifact.py` para persistir `outputs/results.jsonl`, mas não dependem do layout do artefato — qualquer chamador que precise de uma view JSON de um desses contratos pode usá-las diretamente.

## Reprodutibilidade do pipeline resolvido (`schema_version` 0.4.0)

`manifest.json` também persiste `pipeline_preset` (o `PipelinePreset` resolvido — ver [`pipeline.md`](pipeline.md) — codificado por `encode_pipeline_preset()`) e `configuration_digest` (o fingerprint determinístico de `ResolvedPipeline.configuration_digest()`). Isso torna o grafo de estágios e as identidades de backend efetivamente usados por um run inspecionáveis a partir do próprio manifest, sem precisar reabrir `outputs/results.jsonl` e agregar a proveniência de cada evidência individualmente.

O schema `0.4.0` incorpora a nova forma de `SemanticClaim` e os registros
completos de execução semântica. Cada registro preserva request, prompt
renderizado, resposta/hash, parsing, diagnósticos e configuração efetiva
redigida. A cópia humana da resposta bruta só é inventariada em debug `FULL`.
Esta é uma quebra pré-1.0: artifacts `0.3.0` são rejeitados na abertura.
