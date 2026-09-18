# Backend de features densas DINOv3

Este documento descreve `src/contextmap/visual_perception/backends/dinov3.py`, o adapter que converte patch tokens DINOv3 em `DenseFeatureMap` canônico sem expor tensores ou objetos do SDK.

## Contrato e fluxo

`DinoV3DenseFeatureBackend` satisfaz `FeatureExtractor` com escopo `DENSE`. `extract()` publica o `VisualFeature` canônico e envia o payload a um sink compatível com `PerceptionRunWriter.add_feature_payload()`. `extract_dense()` devolve no mesmo passe o `DenseFeatureMap`, `EmbeddingSpace`, array NumPy nativo e a contagem de register tokens excluídos.

O mapa preserva resolução nativa. Nenhum upsampling/downsampling de features acontece no backend; aumento de resolução é um estágio separado (#194).

## CLS, registers e patch tokens

A saída `last_hidden_state` DINOv3 contém, nesta ordem:

```text
CLS | register tokens | spatial patch tokens
```

O runtime consulta `model.config.num_register_tokens`, remove exatamente `1 + num_register_tokens` posições e só então reshapeia os tokens restantes para `(grid_height, grid_width, channels)`. Register tokens não possuem coordenada de imagem e nunca entram no mapa espacial. A contagem removida permanece no resultado e na identidade da transformação para auditoria.

O `EmbeddingSpace.layer` é `last_hidden_state.patch_tokens_after_registers`, distinguindo explicitamente esta escolha de outras camadas ou projeções.

## Preprocessamento e sampling

Como no adapter DINOv2, o processor faz resize direto configurado para `input_width × input_height`, sem center crop. O patch size vem da configuração real do modelo. Origem, stride e suporte de cada célula são mapeados por escala aos pixels da `PreparedImage`; dimensões, transformações anteriores, patch size, register count e fingerprint efetivo entram no `coordinate_transform_id`.

Resize direto pode distorcer aspect ratio se as dimensões configuradas não forem coerentes com a imagem preparada. Essa escolha é explícita, reproduzível e deve ser controlada pelo experimento.

## Identidade e falhas

`DinoV3Config` registra checkpoint, revisão, device, precisão, tamanho de entrada, política local/download, normalização, caminho de payload e versão do adapter. `local_files_only=True` impede download implícito por default.

Falhas são específicas e nunca acionam fallback:

- `DinoV3DependencyError` — PyTorch, Transformers ou Pillow ausente;
- `DinoV3DeviceError` — device/precisão indisponível;
- `DinoV3ModelLoadError` — checkpoint/revision não carregável;
- `DinoV3InferenceError` — erro de payload, preprocessamento, token count, shape ou inferência.

O espaço `family="dinov3"` não é compatível automaticamente com DINOv2, CLIP, AlphaCLIP ou outro checkpoint DINOv3, mesmo quando a dimensão coincide.

## Validação desta implementação

Os testes injetam um runtime determinístico e cobrem o port, persistência, metadata, exclusão de registers, mapping espacial, normalização, determinismo, pooling comum e falhas sem rede ou modelo real.

Por solicitação explícita do usuário, pesos reais não foram baixados nem executados neste ambiente. A issue #69 só deve ser encerrada depois de uma execução controlada na máquina de inferência confirmar carregamento, layout de tokens e repetibilidade numérica do checkpoint selecionado.

## O que este backend não faz

- não gera labels ou claims;
- não associa LiDAR;
- não interpola o feature grid;
- não usa register tokens como evidência espacial;
- não compara espaços de embedding;
- não oferece fallback silencioso.

Ver [`dense_region_association.md`](dense_region_association.md), [`embedding_space.md`](embedding_space.md) e [`feature_store.md`](feature_store.md).
