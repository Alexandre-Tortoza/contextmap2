# Persistência de payload de máscara

Este documento descreve `src/contextmap/visual_perception/mask_store.py`: onde e como a máscara de um `Region2D` (referenciada apenas por `mask_reference`, nunca inline em `outputs/results.jsonl` — `contracts.md`, #378) é persistida, indexada e carregada sob demanda.

## Problema, decisão e justificativa (#378)

- **Problema**: `encode_region()` inlinava `mask.to_dict()` — um array JSON de um inteiro por pixel cobrindo a imagem inteira, mesmo quando a caixa delimitadora da região era pequena. Uma máscara 640x480 custava cerca de 0,92 MB de JSON por região; um frame com dezenas de regiões virava uma única linha de dezenas de MB, e reabrir o run materializava cada máscara como `tuple[bool, ...]` em memória.
- **Justificativa**: `outputs/results.jsonl` deve escalar com metadados de região (KB por frame), e payloads de pixel devem ser compactos e carregados sob demanda, do mesmo jeito que payloads de feature (`feature_store.py`) já são.
- **Entrada**: `Region2D.mask` (um `InlineMask` cobrindo a imagem inteira) para cada região de cada resultado enfileirado em `PerceptionRunWriter`.
- **Processamento**: bit-packing (`numpy.packbits`) do array booleano completo — não recorte pela caixa delimitadora — persistido como `.npy`, indexado por `(source_observation_id, region_id)`.
- **Saída**: `outputs/masks/<source_observation_id>/<region_id>.npy` (payload compacto) + `outputs/masks/mask-index.jsonl` (hash, shape, referência); a região em `outputs/results.jsonl` carrega só `mask_reference`.
- **Métrica**: tamanho de `encode_region()`/`outputs/results.jsonl` deixa de escalar com resolução de imagem; round trip bit-exato via hash de conteúdo.
- **Teste**: `tests/visual_perception/test_mask_store.py` (writer/reader/integridade), `tests/visual_perception/test_serialization.py::test_encode_region_never_inlines_a_full_frame_mask`, `tests/visual_perception/test_run_artifact.py::test_full_frame_mask_is_persisted_compactly_and_lazily_loadable`.

## Por que bit-packing sem recorte pela caixa delimitadora

A issue sugeria RLE ou um recorte bit-packed pela caixa delimitadora como alternativas compactas. Bit-packing da imagem inteira foi escolhido por ser a correção menor e mais simples que já resolve o alvo numérico pedido:

- 8x de redução sobre um byte por pixel (640x480 empacotado = 38.400 bytes, independente do conteúdo) e cerca de 24x sobre o array JSON que substitui;
- mover o payload para fora de `outputs/results.jsonl` já domina a redução pedida (o arquivo volta a escalar só com metadados) — o bit-packing reduz ainda mais o `outputs/masks/` sem depender do formato do conteúdo (ao contrário de RLE);
- mantém o shape do payload idêntico a `(image_height, image_width)`: nenhuma contabilidade de offset de recorte, nenhuma transformação de coordenada para inverter na leitura, e nenhuma divergência com o invariante de imagem inteira que `InlineMask`/`_validate_geometry_bounds` já impõe (`region_models.py`).

Um recorte pela caixa delimitadora foi considerado e rejeitado como complexidade desnecessária para o tamanho que esta issue efetivamente pede (AGENTS.md §13/§14, KISS/YAGNI). `numpy.unpackbits` (limitado à contagem exata de bits original) garante round trip bit-exato.

## NumPy nunca na fronteira pública, mas livre internamente

Como em `feature_store.py`, `import numpy as np` só acontece **dentro** de `MaskStoreWriter.write()` e `MaskStoreReader.load()` — abrir um run, listar `region_keys()`, ou ler um `MaskPayloadEntry` nunca importa NumPy.

## Layout no artefato de run

```text
outputs/
└── masks/
    ├── mask-index.jsonl              # header de schema + um MaskPayloadEntry por linha
    └── frame-0001/
        └── region-0001.npy
```

O caminho de cada payload é `<source_observation_id>/<region_id>.npy`, relativo à raiz do mask store; `Region2D.mask_reference` grava o caminho completo relativo ao artefato (`outputs/masks/<source_observation_id>/<region_id>.npy`).

## Metadados sem carregar o payload

`MaskStoreReader.open(root)` lê apenas `mask-index.jsonl`. A primeira linha identifica explicitamente `record_type="mask_index"` e `schema_version`; versões desconhecidas são rejeitadas antes de decodificar entradas. `entry(source_observation_id, region_id)` devolve um `MaskPayloadEntry` (width, height, hash, tamanho) sem tocar o arquivo `.npy`. Só `load(source_observation_id, region_id)` lê, verifica o hash e desempacota o payload.

A chave do store é `(source_observation_id, region_id)`, porque `RegionId` é local a um `PerceptionResult` e se repete em frames diferentes do mesmo run (`normalization.py` numera regiões por frame, ex. `region-0000`, `region-0001`, ...). Writer e reader rejeitam chaves duplicadas e colisões de `payload_reference`.

## Integridade

`load()` verifica, nesta ordem: arquivo presente; hash de conteúdo (`sha256`) igual ao indexado; arquivo `.npy` válido (erro decodificando -> `MaskPayloadIntegrityError`); contagem de bits desempacotados igual a `width * height`. Qualquer divergência levanta `MaskPayloadIntegrityError` com diagnóstico legível, nunca uma máscara silenciosamente incorreta.

## Fronteira do artefato

Toda referência de caminho é resolvida e validada como estritamente dentro do root do mask store (`_resolve_within_root`) — um `payload_reference` malicioso ou malformado (ex. contendo `../`) nunca escreve/lê fora do artefato.

## Integração com `PerceptionRunArtifact`

Diferente de payloads de feature, persistir a máscara **não é opt-in**: `PerceptionRunWriter._write_masks()` persiste automaticamente, no `finalize()`, a máscara de toda região de todo resultado enfileirado que tenha `region.mask is not None` — não existe um `add_mask_payload()` separado, porque a máscara já chega dentro de `PerceptionResult.regions`, sem risco de payload órfão. `encode_region()` nunca inlina pixels; o `mask_reference` efetivo (apontando para o payload recém-persistido) é atribuído à região antes da codificação, sem mutar o `Region2D` original em memória.

`PerceptionRunReader.mask_store()` abre o `MaskStoreReader` de um run; quando nenhuma região carregou uma máscara, devolve um reader vazio (não é erro). `PerceptionRunReader.list_results()` nunca materializa pixels: `Region2D.mask` decodificado é sempre `None`.

## Consumo por Sensor Association

`sensor_association.membership.associate_regions()` aceita um `mask_loader: RegionMaskLoader | None` opcional — um Protocol local (`load(source_observation_id, region_id) -> InlineMask`) que `MaskStoreReader` já satisfaz estruturalmente. Sem `mask_loader`, uma região sem máscara inline continua marcada `SkipReason.NO_INLINE_MASK`, exatamente como antes de #378. `SensorAssociationExecutor` (runtime) passa `PerceptionRunReader(...).mask_store()` como `mask_loader` ao montar o `SensorAssociationRequest`, já que a esse ponto o run de Visual Perception referenciado sempre foi reaberto do disco.

O `schema_version` do manifest de `PerceptionRunArtifact` foi incrementado para `0.5.0` (ver `run_artifact.md`), uma quebra pré-1.0 aceitável.
