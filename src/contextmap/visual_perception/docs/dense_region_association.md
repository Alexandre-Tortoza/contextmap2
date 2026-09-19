# Associação de feature densa a região

Este documento descreve `src/contextmap/visual_perception/dense_region_association.py`: como uma `Region2D` congelada em coordenadas da imagem preparada é associada a um payload denso e reduzida a um vetor de região sem assumir um backbone ou uma resolução específica.

## Contrato `DenseFeatureMap`

`DenseFeatureMap` conecta um `VisualFeature` de escopo `DENSE` à sua `DenseFeatureSampling` e ao `source_artifact_id` do run/artefato que o possui. Esta identidade é obrigatória porque `FeatureId` é local a um `PerceptionResult`. O `VisualFeature` continua sendo a fonte canônica de `embedding_space_id`, shape, dtype, normalização, referência de payload e proveniência do backend. Mapas nativos possuem `enhancement=None`; mapas produzidos pelo estágio opcional carregam `FeatureResolutionEnhancementProvenance` sem mudar o tipo consumido downstream. O contrato de sampling torna explícitos:

- dimensões do grid e da imagem preparada de origem;
- origem espacial do primeiro suporte;
- stride horizontal e vertical;
- largura e altura do suporte de cada célula;
- identidade da transformação de coordenadas.

Stride e suporte são campos distintos. Assim, o mesmo contrato representa grids regulares sem overlap, receptive fields sobrepostos e mapas produzidos por um estágio opcional de aumento de resolução. A associação não contém branches por backend (`dinov2`, `dinov3`, LoftUp etc.).

## Política determinística de pooling

`pool_region_feature()` usa a política versionada `mask_weighted_mean_preserve_l2_v2`:

1. calcula as células cujo suporte espacial intersecta o bounding box;
2. recorta cada suporte contra o box e os limites da imagem preparada;
3. conta os pixels verdadeiros da máscara local cobertos por cada célula;
4. usa essa contagem como peso na média dos vetores das células;
5. quando o espaço declara `normalization="l2"`, renormaliza a média para norma unitária e rejeita um resultado zero ou não finito.

Quando não há máscara, uma máscara implícita totalmente verdadeira com o envelope rasterizado do box é usada. Para boxes com dimensões fracionárias, esse envelope possui shape `(ceil(height), ceil(width))`. Portanto regiões com máscara, regiões somente com box, mapas nativos e mapas enhanced percorrem exatamente a mesma implementação.

A máscara recebida é um array booleano já decodificado, em coordenadas locais do bounding box e com shape `(ceil(box.height), ceil(box.width))`. Este módulo não interpreta `Region2D.mask_reference`: resolver e decodificar o payload pertence ao chamador que selecionou o artefato.

## Casos vazios, pequenos e de borda

- máscara vazia ou região sem interseção válida com o mapa: `EmptyRegionSupportError`;
- shape/dtype do payload divergente do `VisualFeature`: `RegionAssociationError`;
- shape de máscara diferente do box ou máscara não booleana: `RegionAssociationError`;
- box parcialmente fora da imagem: o suporte é recortado e `coverage_fraction` evidencia a cobertura parcial;
- região de um pixel: associa normalmente à célula cujo suporte a cobre.

Não se produz vetor zero nem NaN silencioso para suporte vazio.

## Diagnóstico e proveniência

`RegionPoolingDiagnostics` preserva intervalos de células candidatas, número de células contribuintes, peso total e fração dos pixels solicitados cobertos por ao menos uma célula válida. Em geometrias com suportes sobrepostos, `total_weight` pode ser maior que o número de pixels únicos; `coverage_fraction` usa a união dos pixels cobertos e permanece entre zero e um.

`RegionPoolingProvenance` registra explicitamente o artefato, `feature_id` e `payload_reference` de origem, `region_id`, referência e hash de conteúdo da máscara decodificada (quando usada), política de pooling, identidade da transformação e fingerprint determinístico da configuração completa. O `RegionPoolingResult` preserva `embedding_space_id`, dtype e a semântica de normalização do mapa denso; para `l2`, isso exige a renormalização explícita da média. Esta operação não compara nem combina espaços de embedding distintos.

## Trade-off explícito na rasterização

Suportes com bordas fracionárias são convertidos para índices inteiros com `floor` no início e `ceil` no fim. Isso inclui todo pixel tocado pelo suporte, mas pode contar um pixel em mais de uma célula quando uma fronteira fracionária atravessa esse pixel. A política é determinística e auditável; trocar por interseção de área contínua seria uma mudança científica e exigiria uma nova versão de `POOLING_POLICY` e avaliação própria.

## O que este módulo não faz

- não altera, repara ou reamostra a geometria congelada de `Region2D`;
- não resolve nem persiste payloads de máscara ou feature;
- não produz label ou `SemanticClaim`;
- não compara features de `EmbeddingSpace` diferentes;
- não conhece SDK, checkpoint ou backend concreto;
- não cria uma nova capability: Feature Extraction permanece sob ownership de `visual_perception`.

Ver [`embedding_space.md`](embedding_space.md) para compatibilidade de espaços, [`feature_store.md`](feature_store.md) para persistência dos payloads, [`feature_resolution_enhancement.md`](feature_resolution_enhancement.md) para lineage native→enhanced e [`contracts.md`](contracts.md) para ownership de `Region2D` e `VisualFeature`.
