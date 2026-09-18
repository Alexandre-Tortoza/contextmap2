# Persistência de payload de feature

Este documento descreve `src/contextmap/visual_perception/feature_store.py`: onde e como o array numérico de um `VisualFeature` (referenciado apenas por `payload_reference`, nunca inline — `contracts.md`, #48) é persistido, indexado, e carregado sob demanda.

## Formato: `.npy`, não Parquet/pickle

Cada payload é um arquivo `.npy` (formato nativo do NumPy) — preserva shape/dtype/valores exatos (incluindo byte order não nativo, já codificado na própria string de dtype), não exige nenhuma dependência de runtime nova além do NumPy, e dá acesso aleatório trivial já que cada feature é seu próprio arquivo. `allow_pickle=False` em toda leitura/escrita: nenhum payload pode conter código Python arbitrário.

## NumPy nunca na fronteira pública, mas livre internamente

`docs/shared-primitives.md` estabelece que primitivas públicas compartilhadas não devem exigir `numpy.ndarray` como identidade do contrato — e `VisualFeature` já obedece isso (`shape`/`dtype` são primitivos). Este módulo é a exceção esperada: cálculos/implementação internos de uma capability podem usar NumPy livremente. Para minimizar o alcance dessa dependência, `import numpy as np` só acontece **dentro** de `FeatureStoreWriter.write()` e `FeatureStoreReader.load()` — abrir um run, listar `feature_ids()`, ou ler um `FeaturePayloadEntry` nunca importa NumPy.

## Layout no artefato de run

```text
outputs/
└── features/
    ├── feature-index.jsonl              # um FeaturePayloadEntry por linha
    └── frame-000120/
        ├── dense-<feature-id>.npy
        └── region-0001-<feature-id>.npy
```

O caminho exato de cada payload é `feature.payload_reference` — o mesmo valor já presente no `VisualFeature` persistido em `outputs/results.jsonl`. Este módulo nunca inventa o caminho; ele apenas grava no caminho que o produtor da feature já declarou.

## Metadados sem carregar o array

`FeatureStoreReader.open(root)` lê apenas `feature-index.jsonl`. `entry(feature_id)` devolve um `FeaturePayloadEntry` (shape, dtype, embedding_space_id, hash, proveniência) sem tocar o arquivo `.npy`. Só `load(feature_id)` lê e decodifica o payload — e persistir um payload é opt-in por feature: um `VisualFeature` sem `add_feature_payload()` correspondente ainda aparece normalmente em `outputs/results.jsonl`, só não tem array carregável nesse run.

## Integridade

`load()` verifica, nesta ordem: arquivo presente; hash de conteúdo (`sha256`) igual ao indexado; arquivo `.npy` válido (erro decodificando -> `FeaturePayloadIntegrityError`); shape decodificado igual ao indexado; dtype decodificado igual ao indexado. Qualquer divergência levanta `FeaturePayloadIntegrityError` com diagnóstico legível, nunca um array silenciosamente incorreto.

`write()` valida shape/dtype do array recebido contra os metadados do `VisualFeature` **antes** de gravar — um array incompatível com o que a feature declara nunca chega a ser persistido.

## Fronteira do artefato

Toda referência de caminho é resolvida e validada como estritamente dentro do root do feature store (`_resolve_within_root`) — um `payload_reference` malicioso ou malformado (ex. contendo `../`) nunca escreve/lê fora do artefato.

## Integração com `PerceptionRunArtifact`

`PerceptionRunWriter.add_feature_payload(feature, source_observation_id, array)` enfileira o payload (mesmo padrão de `add_result`/`add_stage_outcomes`: nada toca o disco antes de `finalize()`, preservando a escrita atômica já estabelecida em `run_artifact.md`). `finalize()` grava cada payload e `feature-index.jsonl`, e adiciona cada arquivo ao `file_inventory` do manifest — `verify_integrity()` já cobre os payloads de feature pelo mesmo mecanismo genérico usado para `outputs/results.jsonl`, sem precisar de uma checagem separada.

`PerceptionRunReader.feature_store()` abre o `FeatureStoreReader` de um run; quando nenhum payload foi persistido, devolve um reader vazio (não é erro).

Nenhuma mudança de `schema_version` do manifest foi necessária — `file_inventory` já era uma lista aberta.
