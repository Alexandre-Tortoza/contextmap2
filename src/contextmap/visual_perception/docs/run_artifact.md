# Artefato de run de percepção

Este documento descreve `src/contextmap/visual_perception/run_artifact.py` e `serialization.py`. Para as convenções globais de artifact/lineage/immutability, ver [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md).

## Layout no workspace local

```text
workspace/
└── runs/
    └── visual-perception/
        └── <sequence-name>/
            ├── runs.json                                              # registro reconstruível, não fonte de verdade
            ├── run-0001__frames-0120-0260__sam3-dinov2-gemini/
            │   ├── README.md                                          # gerado, legível por humano
            │   ├── manifest.json                                      # ponto autoritativo
            │   ├── outputs/
            │   │   └── results.jsonl                                   # um PerceptionResult por linha
            │   ├── metrics/
            │   │   ├── stage-timings.jsonl                             # um StageOutcome por linha
            │   │   └── feature-extraction.jsonl                        # quando há diagnóstico de feature
            │   └── debug/
            │       └── 30-feature-extraction/                          # somente standard/full
            └── run-0002__frames-0120-0260__sam3-dinov2-qwen/
                └── ...
```

Nomeação por índice monotônico (`run-0001`, `run-0002`, ...), nunca timestamp — o maior índice é o run mais recente nesse escopo sequência+capability. `allocate_run_index()` calcula o próximo índice escaneando os diretórios de run **reais** (nunca `runs.json`), então um diretório interrompido/incompleto (sem `manifest.json` válido) nunca é contado.

## Decisões desta issue (v0)

- **`outputs/results.jsonl`, não `outputs/results.parquet`.** Mesma decisão e mesmo motivo da issue #39 de Ingestion: nenhuma dependência de runtime nova (`pyarrow`/`pandas`) se justifica ainda; JSON Lines é inspecionável com ferramentas de texto padrão. Revisitar se o volume de resultados tornar leitura linha-a-linha um gargalo real.
- **`outputs/results.jsonl` continua canônico para metadata; `outputs/features/feature-index.jsonl` indexa apenas payloads numéricos opt-in.** Cada `PerceptionResult` carrega suas `regions`/`features`/`claims`; o feature index não duplica esse contrato, apenas liga a chave `(source_observation_id, feature_id)` ao arquivo `.npy`, hash e metadata necessária para leitura lazy. `finalize()` valida essa referência cruzada antes de publicar o artifact.
- **`debug/` só existe quando há conteúdo e nível `standard`/`full`.** Feature Extraction possui produtores concretos de metadata, suporte e previews (#72), então `PerceptionRunWriter` integra `FeatureExtractionDiagnostic` e `FeatureDiagnosticPreview` em `debug/30-feature-extraction/`. Nível `none` mantém apenas métricas obrigatórias; nenhum diretório vazio é materializado. Ver [`feature_diagnostics.md`](feature_diagnostics.md).
- **`config.yaml`, `lineage.json`, `environment.json`, `events.jsonl` não são escritos no v0.** Nenhum destes tem produtor real ainda (configuração efetiva de backend, lineage de artefatos upstream, ambiente de execução, eventos granulares) — `manifest.json` já cobre a metadata mínima autoritativa (run_id, índice, sequência, seleção, capabilities, contagens). Adicionar esses arquivos vazios/parciais agora seria estrutura sem conteúdo real.

## Escrita atômica

Mesmo padrão de `contextmap.ingestion.sequence_artifact`: `PerceptionRunWriter.finalize()` escreve em um diretório temporário irmão, roda uma checagem de consistência interna, e só então renomeia para o path final — um run interrompido nunca aparenta ser válido. Depois de renomear, `runs.json` é reconstruído a partir de todos os diretórios de run válidos (incluindo o recém-criado).

## Leitura isolada, sem `runs.json`

`PerceptionRunReader(run_dir)` abre um run **apenas com seu próprio diretório** — `manifest.json` + `outputs/results.jsonl` bastam. `runs.json` nunca é necessário para abrir ou entender um run individual; `rebuild_run_registry()` pode reconstruí-lo do zero a qualquer momento a partir dos manifests.

## `serialization.py`

Funções `encode_x`/`decode_x` simétricas para cada tipo de `models.py` (`BackendProvenance`, `BoundingBox2D`, `Region2D`, `VisualFeature`, `SemanticClaim`, `SceneContext`, `PerceptionResult`). Reaproveitadas por `run_artifact.py` para persistir `outputs/results.jsonl`, mas não dependem do layout do artefato — qualquer chamador que precise de uma view JSON de um desses contratos pode usá-las diretamente.

## Reprodutibilidade do pipeline resolvido (issue #55, `schema_version` 0.2.0)

`manifest.json` também persiste `pipeline_preset` (o `PipelinePreset` resolvido — ver [`pipeline.md`](pipeline.md) — codificado por `encode_pipeline_preset()`) e `configuration_digest` (o fingerprint determinístico de `ResolvedPipeline.configuration_digest()`). Isso torna o grafo de estágios e as identidades de backend efetivamente usados por um run inspecionáveis a partir do próprio manifest, sem precisar reabrir `outputs/results.jsonl` e agregar a proveniência de cada evidência individualmente.

Esta é uma quebra de schema pré-1.0 (`0.1.0` → `0.2.0`, ambos os campos são obrigatórios): nenhum leitor para manifests `0.1.0` é mantido, seguindo a mesma postura de todo o resto deste milestone (schema versionado, mas sem compromisso de compatibilidade retroativa antes de 1.0).
