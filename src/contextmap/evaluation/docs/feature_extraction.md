# Avaliação de Feature Extraction

Este documento descreve `src/contextmap/evaluation/feature_extraction.py` e o protocolo determinístico usado na CI da milestone Feature Extraction.

## O que faz

`evaluate_feature_payload()` recebe um `VisualFeature`, o `EmbeddingSpace` completo, o payload numérico e, para scope denso, seu `DenseFeatureMap`. Antes de emitir `FeatureEvaluationReport`, valida:

- fingerprint do espaço de embedding e igualdade da metadata de normalização entre feature e espaço;
- shape, dtype e dimensão vetorial;
- finitude de todos os valores;
- norma unitária quando o produtor declara `normalization="l2"`;
- presença e correspondência da geometria `DenseFeatureMap` para features densas.

O hash do payload inclui dtype, shape e valores em ordem C. Assim, alteração de metadata ou de valores vira uma regressão observável mesmo quando a dimensão permanece igual.

## Decisão principal: três eixos independentes

```text
FeatureEvaluationReport
├── numerical   # valores, finitude, normas
├── spatial     # imagem, grid, origin, stride, support, transform
└── cost        # runtime, memória, bytes, throughput
```

Não existe `quality_score`. Tempo menor não implica feature melhor; resolução maior não implica consistência espacial; mesma dimensão não implica espaço compatível. Cada conclusão deve usar a métrica correspondente.

## Repetibilidade

`assert_repeatable_feature_outputs()` compara duas execuções sobre o mesmo `FeatureEvaluationContext`. Backend/model/configuração, `EmbeddingSpace`, shape, dtype, normalização, transformação espacial e hash dos valores devem coincidir.

`feature_id`, referência de payload, artifact/run de saída e custo podem variar legitimamente entre reexecuções imutáveis, portanto não definem repetibilidade numérica.

## Native versus enhanced

`compare_feature_map_resolutions()` aceita grids diferentes somente quando:

- o `FeatureEvaluationContext` é idêntico;
- ambos são mapas densos com geometria explícita;
- representam a mesma imagem preparada;
- o `EmbeddingSpace` é exatamente compatível.

O contexto preserva o artifact/modelo de origem, o conjunto de regiões e a configuração downstream mantidos fixos. Cada variante preserva separadamente grid, transformação, valores e custo. A função não requer um backend de enhancement e, por isso, a CI usa mapas sintéticos.

## Fixtures e regressões cobertas

Os testes usam arrays pequenos e calculáveis manualmente para cobrir:

- dense, global e region features pelo mesmo validador numérico;
- incompatibilidade de espaço mesmo com payload dimensionalmente plausível;
- grids native e enhanced consumidos pelo mesmo `pool_region_feature()`;
- aspect ratio não quadrado, região minúscula e clipping na borda;
- máscara conhecida, suporte vazio e transformações origin/stride/support (complementando os testes proprietários de `visual_perception`);
- persistência, índice e carregamento lazy antes da avaliação;
- runtime, memória, payload e throughput separados da corretude.

Testes de GPU/checkpoint real permanecem integração opcional: a CI canônica não baixa pesos nem exige um backend de enhancement.

## Trade-offs explícitos

- A verificação L2 usa tolerância absoluta versionada no código (`1e-5`); não tenta inferir outras normalizações a partir dos valores.
- `peak_memory_bytes` é uma medição fornecida pelo runner/diagnóstico. O avaliador não escolhe CUDA, RSS ou `tracemalloc`, pois essas fontes possuem semânticas diferentes.
- `payload_size_bytes` mede o array em memória (`nbytes`), não o tamanho comprimido ou o container persistido.
- O encoder produz valores JSON-compatible, mas esta issue não cria um schema global de artifact de avaliação; nenhum consumidor atual exige esse envelope.

## O que deliberadamente não faz

- não executa DINO, CLIP, AlphaCLIP ou enhancement;
- não compara vetores de `EmbeddingSpace` diferentes;
- não escolhe thresholds de qualidade científica sem dataset/hipótese;
- não transforma o relatório em output do pipeline avaliado;
- não classifica regiões nem altera máscaras.

Ver [`../../visual_perception/docs/dense_region_association.md`](../../visual_perception/docs/dense_region_association.md) para pooling e transformação espacial, [`../../visual_perception/docs/embedding_space.md`](../../visual_perception/docs/embedding_space.md) para compatibilidade e [`../../visual_perception/docs/feature_store.md`](../../visual_perception/docs/feature_store.md) para persistência lazy.
