# Diagnósticos auditáveis de Feature Extraction

Este documento descreve `src/contextmap/visual_perception/feature_diagnostics.py` e sua integração atômica com `PerceptionRunWriter`.

## Separação entre contrato, métrica e debug

```text
outputs/features/                         payloads contratuais + índice
metrics/feature-extraction.jsonl          métricas mínimas obrigatórias
debug/30-feature-extraction/              inspeção humana, nunca dependência downstream
```

Desabilitar debug não remove `VisualFeature`, payload, índice nem métricas de status/timing/memória. Consumidores nunca dependem de `debug/`.

## Schema comum

`FeatureExtractionDiagnostic` registra de forma backend-agnostic:

- observação e imagem preparada (referência/dimensões);
- stage, status e backend/model/checkpoint/configuração;
- feature, escopo, shape, dtype, normalização e payload;
- `EmbeddingSpace` completo e fingerprint;
- preprocessing ordenado;
- timing, memória, warnings, falha ou abstention.

`DenseFeatureDiagnostic` acrescenta artifact de origem, grid, origem (`origin_x`, `origin_y`), stride, suporte e transformação espacial. Essa geometria é contratual: ela vai em `metrics/feature-extraction.jsonl` (chave `dense`, com o tamanho da imagem preparada) em qualquer nível de debug, e reconstrói exatamente o `DenseFeatureSampling` do mapa. Sem ela o payload denso não vira mapa espacial reaberto de um run; uma execução real com DINOv3 mostrou que antes a geometria só existia em `debug/` e sem a origem, obrigando o leitor a supor `(0, 0)`. `RegionFeatureDiagnostic` acrescenta região/box, view de suporte, mask reference/hash, transformação e estatísticas de pooling quando aplicáveis.

Falha ou abstention pode existir sem inventar `feature_id`/payload. Eventos bem-sucedidos ou com warning exigem metadata completa. Estatísticas de pooling são um grupo coerente: todas presentes ou todas ausentes.

## Níveis de debug

### `none`

Persiste apenas `metrics/feature-extraction.jsonl` (que inclui a geometria densa), além dos outputs contratuais normais. Nenhum diretório `debug/` é criado.

### `standard`

Adiciona:

```text
debug/30-feature-extraction/
├── feature-summary.json
├── feature-map-meta.json              # quando há feature densa
├── embedding-spaces.json
├── extraction-events.jsonl
├── dense/                             # previews fornecidos pelo produtor
└── regions/<region-id>/
    ├── feature-meta.json
    └── support-preview.*              # quando fornecido
```

O summary agrupa contagens por escopo, status e backend, além de duração total e pico máximo, permitindo comparar DINOv2, DINOv3, CLIP, AlphaCLIP ou fakes pelo mesmo schema.

### `full`

Inclui tudo de `standard`, previews marcados como full e `regions/<id>/pooling.json` com política, células contribuintes, peso e cobertura.

## Previews

`FeatureDiagnosticPreview` recebe bytes já renderizados pelo produtor e um caminho descritivo iniciado por `dense/`, `regions/` ou `diagnostics/`. O writer não interpreta imagem e não transforma canais arbitrários de embedding em heatmaps. Paths absolutos/escape, duplicatas e previews acima de 5 MiB são rejeitados.

A existência de previews é opcional: só devem ser produzidos quando houver método determinístico e significado científico claro. Payload denso bruto nunca deve ser copiado para debug.

## Imutabilidade e inventário

`PerceptionRunWriter.add_feature_diagnostic()` e `add_feature_preview()` apenas enfileiram conteúdo. `finalize()` escreve tudo no diretório temporário, inclui cada arquivo no `file_inventory`, verifica hashes e só então publica o run por rename atômico. Após finalização, novos diagnósticos/previews são rejeitados.

A integração de diagnostics não exige campos dedicados no `RunArtifactManifest`; o `file_inventory` aberto já representa esses arquivos. O schema global do `PerceptionRunArtifact` está atualmente em `0.4.0` por mudanças posteriores nos contratos semânticos e em sua persistência, não por causa do layout de diagnostics.

Ver [`feature_store.md`](feature_store.md) para payloads contratuais e [`run_artifact.md`](run_artifact.md) para o artefato completo.
