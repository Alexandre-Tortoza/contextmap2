# Aumento opcional da resolução de features densas

Este documento descreve `src/contextmap/visual_perception/feature_resolution_enhancement.py`, o port `FeatureResolutionEnhancement` e a lineage opcional adicionada a `DenseFeatureMap`.

## Contrato do estágio

```text
DenseFeatureMap nativo
    -> FeatureResolutionEnhancement
    -> DenseFeatureMap enhanced
```

O contrato não se chama LoftUp, FeatUp ou pelo nome de outro modelo. Um backend concreto implementa `enhance(source)` e devolve o mesmo tipo público consumido por pooling e, futuramente, Sensor Association. `enhance_feature_resolution()` aplica as invariantes comuns sem carregar SDK ou checkpoint.

O mapa de origem é uma dataclass imutável. A saída deve possuir novos `feature_id`, `source_artifact_id` e `payload_reference`; corrigir ou sobrescrever o artifact nativo não é permitido.

## Lineage obrigatória

Um mapa nativo possui `DenseFeatureMap.enhancement=None`. Um mapa enhanced carrega `FeatureResolutionEnhancementProvenance`, que registra:

- feature, artifact, payload reference, hash e tamanho do payload de origem;
- `EmbeddingSpace`, dtype, normalização e transformação espacial de origem;
- backend/modelo/configuração efetivamente selecionados;
- dimensões dos grids de entrada e saída;
- dimensões da imagem preparada;
- `EmbeddingSpace` declarado para a saída;
- device, precision, runtime e peak memory;
- hash e tamanho do payload de saída.

O `DenseFeatureMap` de saída continua registrando shape, dtype, normalização, payload, backend e `DenseFeatureSampling` completos. O wrapper rejeita lineage que não identifica exatamente o input, provenance divergente do backend selecionado, dimensões espaciais inconsistentes, reutilização das identidades imutáveis de origem ou um grid que não aumente ao menos um eixo sem reduzir o outro.

Se o enhancement alterar a semântica da representação, o backend deve declarar outro `output_embedding_space_id`. Preservar o fingerprint é permitido somente quando os vetores continuam no mesmo espaço; alterar a normalização mantendo o mesmo fingerprint é rejeitado.

## Estágio opcional no DAG

`pipeline.py` conhece a capability explícita `feature_resolution_enhancement`. Ela só é construída quando um `StageSpec` de um preset alternativo a seleciona:

```text
native_dense_feature (source)
    -> feature_resolution_enhancement (optional=True)
```

`CANONICAL_PRESET_V1` permanece inalterado e não inclui o estágio. Ausência de dependência/checkpoint concreto não afeta o caminho padrão; quando um preset seleciona o estágio, a falta de sua `StageBackendFactory` falha na resolução antes da execução.

O preset resolvido, backend e parâmetros entram no `configuration_digest` já persistido pelo `PerceptionRunArtifact`, deixando a inserção/remoção do estágio auditável sem introduzir infraestrutura genérica de plugins.

## Consumo comum

`pool_region_feature()` recebe mapas native e enhanced pela mesma assinatura e usa apenas `DenseFeatureSampling`. Não há `if loftup`, `if enhanced` ou heurística baseada no tamanho do grid. Cada transformação mantém `coordinate_transform_id` próprio, então a associação espacial continua determinística.

## Backend aprendido

Esta issue não materializa um backend LoftUp-style. Não existe no repositório uma dependência/checkpoint validado nem uma execução com pesos que demonstre compatibilidade e benefício no dataset do projeto. Inventar um wrapper sem essa validação transformaria uma hipótese da literatura em backend aparentemente suportado.

O port, a composição opcional, a lineage e os testes fake deixam a fronteira pronta para um backend real posterior. Esse backend deverá permanecer em `visual_perception/backends/`, falhar apenas quando selecionado e ser avaliado pelo protocolo de `contextmap.evaluation` antes de integrar qualquer preset canônico.

## Trade-offs explícitos

- O port trabalha com referências de artifact/payload, não com `numpy.ndarray` na identidade pública. O backend selecionado é responsável pelo I/O e pela escrita atômica de seu artifact.
- `peak_memory_bytes` preserva o valor medido, mas a semântica da fonte (VRAM, RSS etc.) deve ser documentada pelo backend; sinais heterogêneos não são somados.
- A validação exige aumento geométrico do grid, mas não afirma ganho de qualidade. Qualidade, estabilidade e custo permanecem métricas separadas.
- O estágio aceita um novo `EmbeddingSpace` explícito, mas nunca assume compatibilidade por dimensão.

## O que deliberadamente não faz

- não altera a resolução produzida pelo backend DINO nativo;
- não adiciona o estágio ao preset canônico;
- não executa/download pesos nem escolhe LoftUp como arquitetura pública;
- não produz labels, máscaras ou associação LiDAR;
- não implementa cache/reuse global; apenas preserva identidades para o runtime fazê-lo;
- não mistura custo com qualidade em um score único.

Ver [`dense_region_association.md`](dense_region_association.md) para o contrato espacial/consumidor comum, [`pipeline.md`](pipeline.md) para composição declarativa e [`../../evaluation/docs/feature_extraction.md`](../../evaluation/docs/feature_extraction.md) para o protocolo native versus enhanced.
