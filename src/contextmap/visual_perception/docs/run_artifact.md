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
            │   └── metrics/
            │       └── stage-timings.jsonl                             # um StageOutcome por linha
            └── run-0002__frames-0120-0260__sam3-dinov2-qwen/
                └── ...
```

Nomeação por índice monotônico (`run-0001`, `run-0002`, ...), nunca timestamp — o maior índice é o run mais recente nesse escopo sequência+capability. `allocate_run_index()` calcula o próximo índice escaneando os diretórios de run **reais** (nunca `runs.json`), então um diretório interrompido/incompleto (sem `manifest.json` válido) nunca é contado.

## Decisões desta issue (v0)

- **`outputs/results.jsonl`, não `outputs/results.parquet`.** Mesma decisão e mesmo motivo da issue #39 de Ingestion: nenhuma dependência de runtime nova (`pyarrow`/`pandas`) se justifica ainda; JSON Lines é inspecionável com ferramentas de texto padrão. Revisitar se o volume de resultados tornar leitura linha-a-linha um gargalo real.
- **Um único `outputs/results.jsonl`, não arquivos separados `regions.jsonl`/`semantic-claims.jsonl`/`feature-index.jsonl`.** Cada `PerceptionResult` já carrega suas próprias `regions`/`features`/`claims` aninhadas (issue #48) — duplicar essa informação em índices paralelos seria redundância sem um caso de uso real ainda (YAGNI). `PerceptionRunReader.result(source_observation_id)` já permite lookup direto por observação sem escanear o diretório inteiro, satisfazendo o requisito real da issue.
- **`debug/` por frame não é criado por este writer.** A estrutura ordenada `debug/frames/<frame>/00-input/ ... 90-output/` sugerida pela issue depende de conteúdo que ainda não existe neste milestone (overlays de região, crops, respostas de modelo — dependem de Region Discovery/Feature Extraction/Semantic Interpretation reais). Criar as pastas vazias antecipando esse conteúdo violaria YAGNI (`docs/development.md`). O writer já registra `StageOutcome` com timing/erro por estágio (`metrics/stage-timings.jsonl`), que é o auditável mínimo desta issue; backends futuros podem estender o writer para popular `debug/` quando tiverem conteúdo real.
- **`config.yaml`, `lineage.json`, `environment.json`, `events.jsonl` não são escritos no v0.** Nenhum destes tem produtor real ainda (configuração efetiva de backend, lineage de artefatos upstream, ambiente de execução, eventos granulares) — `manifest.json` já cobre a metadata mínima autoritativa (run_id, índice, sequência, seleção, capabilities, contagens). Adicionar esses arquivos vazios/parciais agora seria estrutura sem conteúdo real.

## Escrita atômica

Mesmo padrão de `contextmap.ingestion.sequence_artifact`: `PerceptionRunWriter.finalize()` escreve em um diretório temporário irmão, roda uma checagem de consistência interna, e só então renomeia para o path final — um run interrompido nunca aparenta ser válido. Depois de renomear, `runs.json` é reconstruído a partir de todos os diretórios de run válidos (incluindo o recém-criado).

## Leitura isolada, sem `runs.json`

`PerceptionRunReader(run_dir)` abre um run **apenas com seu próprio diretório** — `manifest.json` + `outputs/results.jsonl` bastam. `runs.json` nunca é necessário para abrir ou entender um run individual; `rebuild_run_registry()` pode reconstruí-lo do zero a qualquer momento a partir dos manifests.

## `serialization.py`

Funções `encode_x`/`decode_x` simétricas para cada tipo de `models.py` (`BackendProvenance`, `BoundingBox2D`, `Region2D`, `VisualFeature`, `SemanticClaim`, `SceneContext`, `PerceptionResult`). Reaproveitadas por `run_artifact.py` para persistir `outputs/results.jsonl`, mas não dependem do layout do artefato — qualquer chamador que precise de uma view JSON de um desses contratos pode usá-las diretamente.
