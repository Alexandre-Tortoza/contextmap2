# Backend de features SigLIP2

Este documento descreve `src/contextmap/visual_perception/backends/siglip2.py`, o adapter que expõe o vision encoder SigLIP2 **diretamente** como `FeatureExtractor` substituível, com identidade de espaço de embedding própria, para experimentos de features compatíveis com o Eagle 2.5 (issue #572).

O Eagle 2.5 usa SigLIP2 como vision encoder, mas estados ocultos de um VLM generativo não são um espaço de features canônico. Este adapter nunca passa pelo decoder de linguagem do Eagle: ele roda só o encoder de visão, com contrato explícito, para comparar SigLIP2 com os caminhos DINO/CLIP. Ele não afirma que SigLIP2 seja melhor que DINO ou CLIP.

## Um escopo por instância

`Siglip2Config.scope` escolhe o único escopo que a instância produz:

| Escopo | Saída do encoder | Contrato publicado |
| --- | --- | --- |
| `DENSE` | `last_hidden_state` (patch tokens depois do `post_layernorm`) | `VisualFeature` denso + `DenseFeatureMap` na grade nativa |
| `GLOBAL` | `pooler_output` (cabeça de attention pooling, "MAP") | `VisualFeature` global `(channels,)` |

`REGION` é recusado pela configuração: não há estratégia de região definida para SigLIP2. `extract()` segue o port (`regions` é ignorado fora de `REGION`) e devolve a única feature do escopo configurado. `extract_dense()` e `extract_global()` devolvem no mesmo passe o array e o `EmbeddingSpace`; chamar o método do outro escopo levanta `Siglip2ScopeError` antes de qualquer inferência. Um runtime que devolva a saída do outro escopo (vetor para `DENSE`, grade para `GLOBAL`) falha com `Siglip2InferenceError`.

O `GLOBAL` é o embedding de imagem que o SigLIP usa contra texto, mas este adapter é estritamente visual: não codifica texto, não calcula similaridade e não escolhe labels.

## Caminho de preprocessamento escolhido: FixRes

SigLIP2 é publicado em duas variantes no Hugging Face:

- **FixRes** (`google/siglip2-*-patch{14,16}-{224,256,384,512}`): `model_type="siglip"`, carregado por `SiglipVisionModel`, entrada quadrada de resolução fixa, compatível com SigLIP;
- **NaFlex** (`google/siglip2-*-naflex`): `model_type="siglip2"`, carregado por `Siglip2VisionModel`, cujo processor faz um resize que preserva o aspect ratio até `max_num_patches`, transforma a imagem em sequência de patches com padding, `pixel_attention_mask` e `spatial_shapes` por imagem.

O adapter implementa **só FixRes**, pelas razões:

- é o caminho do Eagle 2.5: o `extract_feature` do Eagle 2.5 instancia `SiglipVisionModel` com os pesos `Siglip2-So400m-Patch16-512` e usa `last_hidden_state` (`vision_select_layer -1`);
- a grade é fixa e quadrada, então a geometria de amostragem é explícita e igual para todo frame;
- o NaFlex entrega o resize ao processor, com grade variável por imagem; o repositório já decidiu que o resize é do adapter, não do processor (issue #339).

O runtime carrega primeiro só a configuração do checkpoint (`AutoConfig`) e recusa, **antes de carregar pesos**, um checkpoint cujo `model_type` não seja `"siglip"` (NaFlex) ou cuja resolução nativa (`vision_config.image_size`) difira de `input_size`: position embeddings nunca são interpolados. O adapter não distingue um checkpoint SigLIP 1 de um SigLIP2 FixRes (a arquitetura é idêntica); a identidade que importa é o checkpoint fixado por revisão.

O pixel path é:

1. decodificar a imagem preparada em RGB e conferir o tamanho com a metadata da `PreparedImage`;
2. resize direto no **Pillow** para `input_size × input_size`, sem center crop, com o filtro `resize_filter`: `bicubic` (default, a transform de avaliação do Eagle 2.5) ou `bilinear` (o `resample` que o `preprocessor_config.json` do SigLIP2 declara);
3. o processor do checkpoint faz só rescale `1/255` e normalização (média e desvio 0,5 nos checkpoints SigLIP2 publicados), com `do_resize=False`, pelo mesmo `preprocess_pixel_values` dos adapters DINO/CLIP.

O resize direto distorce o aspect ratio de uma imagem não quadrada (640×480 vira 512×512). É uma escolha explícita, reproduzível e registrada na geometria. O tiling dinâmico do Eagle 2.5 (várias tiles mais thumbnail) **não** é reproduzido: o adapter produz uma única view da imagem inteira.

## Geometria densa

SigLIP não tem token CLS nem registers: todos os tokens de `last_hidden_state` são patches, em ordem row-major. O runtime confere que há exatamente `grid_width × grid_height` tokens (com `grid = floor(input_size / patch_size)`) e só então reshapeia para `(grid_height, grid_width, channels)`. Nenhum upsampling ou downsampling acontece no backend; aumento de resolução é o estágio separado `FeatureResolutionEnhancement`. Uma grade devolvida pelo runtime que não corresponda à geometria entrada/patch falha, nunca é reamostrada.

O `DenseFeatureSampling` mapeia a grade para os pixels da `PreparedImage` com origem `(0, 0)` e, por eixo, passo e suporte iguais a `patch × imagem / input_size` (uma só divisão, para que a geometria sintética seja reproduzida exatamente). Para 640×480 com `input_size=512` e patch 16, a grade é 32×32 com células de 20×15 px. Quando `input_size` não é múltiplo do patch (patch 14 sobre 384 px), 27 células cobrem 378 px e a faixa final fica sem célula; o `sampling` é fiel a isso, como no DINO.

O `coordinate_transform_id` cobre tamanho e transformações da imagem preparada, a entrada do modelo, a identidade de preprocessamento, o patch e a política de tokens; device e precisão não entram, porque não mudam a geometria.

O `DenseFeatureMap` resultante é o mesmo contrato backend-neutral dos mapas DINO: `pool_region_feature()` faz o pooling por região e `sample_dense_features()` de Sensor Association faz a amostragem 2D→3D sem nenhum branch por backend. Um teste amostra a mesma grade nativa produzida pelo SigLIP2 e pelo DINOv3 e obtém as mesmas células (`tests/sensor_association/test_dense_sampling_across_backends.py`).

## Normalização

`l2_normalize` é explícito e vale para cada vetor persistido (cada patch no `DENSE`, o vetor pooled no `GLOBAL`); o default é `False` (`normalization="none"`). A declaração `l2` só é gravada depois que o vetor foi de fato normalizado. Valores não finitos e vetores de norma zero com `l2_normalize=True` falham antes da persistência.

## Espaço de embedding

| Campo | Valor |
| --- | --- |
| `family` | `"siglip2"` |
| `model` | checkpoint |
| `version` | revisão |
| `checkpoint` | `"<checkpoint>@<revisão>"` |
| `layer` | `"<saída>\|preprocessing=<identidade>"` |
| `dimension` | canais reais |
| `normalization` | `"none"` ou `"l2"` |

A saída é `last_hidden_state.post_layernorm_patch_tokens` (`DENSE`) ou `pooler_output.attention_pooling_head` (`GLOBAL`). A identidade de preprocessamento é `pillow_direct_<filtro>_resize_<N>x<N>_processor_rescale_normalize_no_crop_v1`.

O issue #572 exige que revisão e preprocessamento quebrem a compatibilidade de embeddings. `EmbeddingSpace` não tem um campo próprio de preprocessamento, então a identidade de preprocessamento **qualifica o `layer`**: a mesma saída do encoder lida sob outro filtro de resize ou outra resolução de entrada é outro espaço, e `ensure_compatible_features()` recusa a comparação com `EmbeddingSpaceMismatchError`. Os adapters DINO/CLIP não incluem o preprocessamento no espaço; um campo dedicado em `EmbeddingSpace`, válido para todos os backends, seria uma mudança de contrato compartilhado fora do escopo deste adapter.

| Muda o `embedding_space_id` | Muda só o fingerprint da configuração |
| --- | --- |
| checkpoint, revisão, escopo (saída), `resize_filter`, `input_size`, `l2_normalize` | `device`, `precision`, `local_files_only`, `payload_prefix`, `code_version` |

A precisão fica fora do espaço, como no DINO: o `VisualFeature` já declara o `dtype` do payload. O espaço `siglip2` nunca é compatível automaticamente com DINOv2, DINOv3, CLIP, AlphaCLIP ou outro checkpoint SigLIP2, mesmo com dimensão igual; `DENSE` e `GLOBAL` do mesmo checkpoint também são espaços distintos.

## Persistência

O backend envia cada payload validado ao sink compatível com `PerceptionRunWriter.add_feature_payload()`, com `payload_reference` em `features/<feature_id>.npy`. O `FeatureId` vem do `feature_stage_id` fornecido pelo composition root. A geometria densa é persistida pelas métricas obrigatórias de `FeatureExtractionDiagnostic`, exatamente como no DINO: um teste finaliza um run imutável, reabre o payload pelo `FeatureStoreReader` e reconstrói o `DenseFeatureMap` a partir de `metrics/feature-extraction.jsonl`.

## Seleção pelo runtime

O runtime compõe o SigLIP2 como backend `siglip2` do componente `visual_perception.dense_features`. Esse slot **fixa** `scope = dense`: um documento que peça `scope = "global"` ali é recusado por `BackendConfigurationError` na composição. O `GLOBAL` fica disponível para consumidores diretos (por exemplo, a avaliação de features), já que o preset canônico não tem estágio de features globais.

```toml
[components.visual_perception.dense_features]
backend = "siglip2"

[components.visual_perception.dense_features.siglip2]
checkpoint = "google/siglip2-so400m-patch16-512"
revision = "<SHA Git completo de 40 caracteres>"
input_size = 512
resize_filter = "bicubic"
```

`device` herda `resources.device` quando não é declarado. O loader Hugging Face empacotado é lazy (`torch`, `transformers`, `Pillow`); um `resources.providers` pode fornecer outro runtime que satisfaça `Siglip2Runtime`.

## Identidade e falhas

`Siglip2Config` registra checkpoint, revisão, escopo, `input_size`, device, precisão, filtro de resize, política local/download, normalização, caminho de payload e versão do adapter. `revision` deve ser o SHA Git completo; referências móveis como `main` são recusadas antes do carregamento. `local_files_only=True` impede download implícito por default.

Falhas são específicas e nunca acionam fallback:

- `Siglip2DependencyError` — PyTorch, Transformers ou Pillow ausente, incluindo pacotes que o processor ou o modelo importam (`torchvision` com `transformers` 5.x; `dtype=` requer `transformers` >= 4.56);
- `Siglip2DeviceError` — device/precisão indisponível (`float16` em CPU é recusado);
- `Siglip2ModelLoadError` — checkpoint/revisão não carregável, checkpoint NaFlex ou resolução nativa diferente de `input_size`;
- `Siglip2InferenceError` — payload inexistente ou fora da raiz, imagem com tamanho divergente, contagem de tokens, grade, shape, dtype, valores inválidos, checkpoint sem cabeça de pooling no `GLOBAL`;
- `Siglip2ScopeError` — extração de um escopo que a instância não produz.

## O que o Eagle 2.5 propõe, o que adaptamos e o que é nosso

- **Eagle 2.5:** SigLIP2-So400m-Patch16-512 como vision tower (`SiglipVisionModel`), imagens em tiles quadradas redimensionadas com bicúbico e normalizadas com média/desvio 0,5, `last_hidden_state` seguido de pixel shuffle e de um conector MLP para o LLM.
- **Adaptado:** o mesmo vision tower e a mesma saída antes do pixel shuffle, com a mesma transform bicúbica e normalização.
- **Original do ContextMap2:** uma única view da imagem inteira, a geometria explícita do `DenseFeatureMap`, a saída `GLOBAL` da cabeça de pooling como escopo separado, a identidade de espaço que inclui o preprocessamento e a persistência auditável. Pixel shuffle, conector MLP e decoder de linguagem não são usados.

## Validação desta implementação

Os testes de CI injetam um encoder determinístico e cobrem o port, a geometria exata da grade para dimensões sintéticas (incluindo células anisotrópicas e a faixa sem célula do patch 14), a ausência de reamostragem, o fingerprint determinístico, a quebra de compatibilidade por revisão/checkpoint/filtro/resolução/normalização/escopo, a validação de escopo, a normalização, a persistência pelo run artifact, o pooling comum e a amostragem 2D→3D pelo mesmo caminho do DINOv3. O runtime Hugging Face é exercitado com módulos de SDK falsos: reshape row-major dos tokens, filtro de resize, vetor pooled, ausência da cabeça de pooling e recusa de NaFlex/resolução divergente antes de carregar pesos.

**Ainda não houve execução real** com pesos SigLIP2. Pendentes de um run em GPU sobre a fatia de referência congelada: validade e repetibilidade do payload, consistência/recuperação entre views com correspondências anotadas, separabilidade de regiões, latência, tamanho de payload, memória de pico e cobertura da amostragem 2D→3D. Até lá, o adapter não deve ser tratado como validado numericamente.

## O que este backend não faz

- não extrai features pelo decoder de linguagem do Eagle 2.5 nem usa seus estados ocultos;
- não suporta checkpoints NaFlex nem interpola position embeddings;
- não produz features de região;
- não gera labels, claims nem scores contra texto;
- não interpola o feature grid;
- não compara espaços de embedding;
- não oferece fallback silencioso.

Ver [`feature-extraction.md`](feature-extraction.md), [`dense_region_association.md`](dense_region_association.md), [`embedding_space.md`](embedding_space.md), [`feature_store.md`](feature_store.md) e [`dinov3.md`](dinov3.md).
