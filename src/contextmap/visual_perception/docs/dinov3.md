# Backend de features densas DINOv3

Este documento descreve `src/contextmap/visual_perception/backends/dinov3.py`, o adapter que converte patch tokens DINOv3 em `DenseFeatureMap` canônico sem expor tensores ou objetos do SDK.

## Contrato e fluxo

`DinoV3DenseFeatureBackend` satisfaz `FeatureExtractor` com escopo `DENSE`. `extract()` publica o `VisualFeature` canônico e envia o payload a um sink compatível com `PerceptionRunWriter.add_feature_payload()`. `extract_dense()` devolve no mesmo passe o `DenseFeatureMap`, `EmbeddingSpace`, array NumPy nativo e a contagem de register tokens excluídos.

O mapa preserva resolução nativa. Nenhum upsampling/downsampling de features acontece no backend; aumento de resolução é um estágio separado (#194).

O composition root fornece `feature_stage_id`. Seu SHA-256 participa de cada
`FeatureId` e referência de payload, evitando colisão com outros extractors
executados no mesmo `PerceptionResult`.

## CLS, registers e patch tokens

A saída `last_hidden_state` DINOv3 contém, nesta ordem:

```text
CLS | register tokens | spatial patch tokens
```

O runtime consulta `model.config.num_register_tokens`, remove exatamente `1 + num_register_tokens` posições e só então reshapeia os tokens restantes para `(grid_height, grid_width, channels)`. Register tokens não possuem coordenada de imagem e nunca entram no mapa espacial. A contagem removida permanece no resultado e na identidade da transformação para auditoria.

O `EmbeddingSpace.layer` é `last_hidden_state.patch_tokens_after_registers`, distinguindo explicitamente esta escolha de outras camadas ou projeções.

## Preprocessamento e sampling

O adapter faz resize bilinear direto para `input_width × input_height` no **Pillow**, sem center crop, e usa o processor apenas para rescale e normalização (`do_resize=False`). O resize é do adapter, não do processor: a implementação de resize do processor depende do backend instalado (torchvision ou PIL) e dá pixels diferentes (até 1,75e-2) e features diferentes (cosseno mínimo por patch 0,981) para o mesmo checkpoint, config e imagem. Com o resize no Pillow, os dois backends concordam a 2,4e-7 (issue #339). O patch size vem da configuração real do modelo. Origem, stride e suporte de cada célula são mapeados por escala aos pixels da `PreparedImage`; dimensões, transformações anteriores, patch size, register count e fingerprint efetivo entram no `coordinate_transform_id`.

Resize direto pode distorcer aspect ratio se as dimensões configuradas não forem coerentes com a imagem preparada. Essa escolha é explícita, reproduzível e deve ser controlada pelo experimento.

## Identidade e falhas

`DinoV3Config` registra checkpoint, revisão, device, precisão, tamanho de entrada, política local/download, normalização, caminho de payload e versão do adapter. `revision` deve ser o SHA Git completo de 40 caracteres; referências móveis como `main` são rejeitadas antes do carregamento. `local_files_only=True` impede download implícito por default.

Falhas são específicas e nunca acionam fallback:

- `DinoV3DependencyError` — PyTorch, Transformers ou Pillow ausente, incluindo pacotes que o processor ou o modelo importam (por exemplo `torchvision`, exigido pelo `transformers` 5.x; requer `transformers` >= 4.56 por causa de `dtype=`);
- `DinoV3DeviceError` — device/precisão indisponível;
- `DinoV3ModelLoadError` — checkpoint/revision não carregável;
- `DinoV3InferenceError` — erro de payload, preprocessamento, token count, shape ou inferência.

Valores não finitos e vetores de norma zero quando `l2_normalize=True` falham
antes da persistência.

O espaço `family="dinov3"` não é compatível automaticamente com DINOv2, CLIP, AlphaCLIP ou outro checkpoint DINOv3, mesmo quando a dimensão coincide.

## Validação desta implementação

Os testes de CI injetam um runtime determinístico e cobrem o port, persistência, metadata, exclusão de registers, mapping espacial, normalização, determinismo, pooling comum, falhas sem rede ou modelo real e a reconstrução do mapa denso a partir das métricas persistidas.

Execução real (issue #69, 2026-09-21; RTX 3060, torch 2.14+cu130, transformers 5.17, 20 frames reais de `corridor-02` da seleção `sha256:dc641b34…`): `facebook/dinov3-vitb16-pretrain-lvd1689m` (revisão `5931719e…`, patch 16, 4 registers, 768 canais) e `facebook/dinov3-vits16-pretrain-lvd1689m` (revisão `114c1379…`, 384 canais), ambos com entrada 448×336 e `float32`. O relatório completo, com identidade, números e limites, está em [`dinov3-validation.md`](dinov3-validation.md). Em resumo:

- **carga:** o adapter carrega o checkpoint fixado por revisão em cache local (safetensors, sha256 conferido contra o Hub) e produz `(21, 28, 768)` com stride e suporte de `16 × 640/448 ≈ 22,86` px e origem `(0, 0)`; 1,3–1,6 s incluindo a carga, 336 MiB de GPU (ViT-B) e mediana de 42 ms por frame;
- **layout de tokens:** a contagem de registers coincide entre `config`, o parâmetro real do modelo e a aritmética de tokens; a saída é idêntica (diferença máxima 8e-6) a um forward cru independente. Um bloco magenta colado em posição conhecida é localizado com IoU 0,88–1,0 e erro de centroide de 2–9 px (célula de 16–23 px) usando só o `sampling`; sem remover os registers o bloco aparece deslocado exatamente 4 células (IoU 0), e a leitura em ordem de colunas o desloca dezenas a centenas de pixels. O pico de equivariância à translação cai no deslocamento correto (0,4) e (3,0) células;
- **repetibilidade:** diferença 0,0 na mesma instância, depois de outro frame e em instância nova; os 20 payloads têm hash idêntico entre dois processos; `assert_repeatable_feature_outputs` passa;
- **pooling:** as 429 regiões reais do SAM2 (20 frames) passam por `pool_region_feature`; contra uma reimplementação independente da política a diferença máxima é 2,4e-7 e pesos e contagem de células coincidem nas 429;
- **persistência:** um run imutável reabre com integridade íntegra, payload idêntico e, com `FeatureDebugLevel.NONE`, o `DenseFeatureSampling` é reconstruído exatamente de `metrics/feature-extraction.jsonl`.

A execução real revelou que a geometria densa só existia em `debug/` e sem a origem da grade; ela agora é persistida nas métricas obrigatórias com a origem (ver [`feature_diagnostics.md`](feature_diagnostics.md)), com testes de regressão.

Observações que não foram alteradas neste adapter (detalhes no relatório): entradas que não são múltiplos do patch deixam uma faixa sem célula (o `sampling` é fiel e o pooling informa `coverage_fraction`; o DINOv2 se comporta igual); `coordinate_transform_id` inclui o fingerprint da configuração (device e precisão); `float32` em GPU Ampere usa TF32 no conv do patch por default do cuDNN (CPU contra GPU: cosseno mínimo 0,9999999998).

Esses resultados validam carregamento, layout espacial, contratos e estabilidade numérica dessas configurações; não constituem avaliação científica comparativa da qualidade dos embeddings.

## O que este backend não faz

- não gera labels ou claims;
- não associa LiDAR;
- não interpola o feature grid;
- não usa register tokens como evidência espacial;
- não compara espaços de embedding;
- não oferece fallback silencioso.

Ver [`dense_region_association.md`](dense_region_association.md), [`embedding_space.md`](embedding_space.md) e [`feature_store.md`](feature_store.md).
