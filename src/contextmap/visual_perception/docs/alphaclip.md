# Backend de features de região AlphaCLIP

Este documento descreve `src/contextmap/visual_perception/backends/alphaclip.py`, o adapter de `FeatureExtractor` que produz vetores de região condicionados pela máscara congelada.

## Contrato de máscara

`Region2D.mask_reference` permanece uma referência opaca. Um `RegionMaskSource` injetado resolve esse payload e entrega uma máscara booleana em coordenadas locais do bounding box, com shape exato `(box.height, box.width)`. O backend valida dtype, shape e suporte não vazio; não corrige nem resegmenta a região.

A máscara local é posicionada explicitamente no canvas da `PreparedImage`. Regiões parcialmente fora da imagem são recortadas; regiões sem pixels verdadeiros válidos falham. `AlphaClipView` preserva:

- `region_id`, `mask_reference` e hash do conteúdo decodificado;
- dimensões da imagem preparada;
- crop/view exato;
- dimensões de entrada do modelo;
- interpolação RGB (`bicubic`) e de máscara (`nearest`);
- identidade determinística da transformação completa.

Máscara sempre usa nearest-neighbor para não fabricar valores intermediários antes da normalização alpha esperada pelo modelo.

## Políticas de view

- `full_image`: RGB e máscara permanecem no contexto completo da imagem;
- `context_box:<fraction>`: o box é expandido pela fração configurada e recortado na imagem, com warning quando toca a borda.

RGB e alpha percorrem a mesma view e chegam às mesmas dimensões de entrada. A `Region2D` original nunca é mutada.

Depois do resize geométrico explícito, RGB recebe somente conversão
HWC→CHW e normalização fotométrica CLIP; o preprocess retornado pelo SDK não
executa uma segunda transformação geométrica. O runtime valida batch e
dimensões espaciais de RGB/alpha antes da inferência.

## Espaço de embedding

O contrato usa `family="alphaclip"` e `layer="alpha_conditioned_image_projection"`, com fingerprint combinado dos checkpoints base+alpha, dimensão real e normalização. Essa identidade é deliberadamente diferente de CLIP comum e DINO, mesmo quando a dimensão coincide.

O `AlphaClipRegionFeatureBackend` não codifica texto, não calcula similaridade e
não escolhe labels. O `AlphaClipSemanticScorer` implementado é um adapter
separado e só compara claims regionais com a feature da mesma região congelada.

## Runtime oficial e checkpoints locais

`OfficialAlphaClipRuntime` envolve o pacote oficial `alpha_clip`: `model.visual(image, alpha)`. PyTorch, AlphaCLIP e Pillow são importados apenas na primeira execução.

Para impedir download implícito, a configuração exige dois caminhos locais sob `checkpoint_root`:

- base CLIP checkpoint;
- AlphaCLIP vision checkpoint.

Antes de carregar, o runtime calcula SHA-256 combinado e compara com `checkpoint_fingerprint`. Caminho ausente, escape do root ou hash divergente falha com `AlphaClipModelLoadError`. Device, dependência e inferência possuem erros próprios; não há fallback.

## Persistência e diagnósticos

Cada vetor é enfileirado pelo sink compatível com
`PerceptionRunWriter.add_feature_payload()` e reabre pelo `FeatureStoreReader`
normal. `AlphaClipDiagnostics` preserva tempo, pico de memória quando disponível
e warnings; a persistência comum de métricas e previews já existe via
`FeatureExtractionDiagnostic` e `PerceptionRunWriter`.

Valores não finitos e vetores de norma zero sob normalização L2 são rejeitados
antes do sink. Durações não finitas também não são aceitas como diagnóstico.

O composition root fornece `feature_stage_id`; seu SHA-256 participa do
`FeatureId` e da referência de payload para impedir colisões com CLIP, DINO
ou outro feature stage no mesmo `PerceptionResult`.

## Validação desta implementação

Testes com runtime e mask source fakes cobrem transformações, bordas, erros de máscara, proveniência, identidade incompatível com CLIP, persistência/reabertura e diagnósticos sem baixar pesos.

Por instrução do usuário, a execução real não ocorreu neste ambiente. A issue #71 permanece aberta até uma run controlada na máquina de inferência confirmar os checkpoints, preprocessamento oficial, uso de memória e repetibilidade numérica.

## O que este backend não faz

- não altera ou cria máscaras;
- não exige AlphaCLIP para todas as features de região;
- não gera semântica ou scores;
- não presume compatibilidade com CLIP;
- não projeta evidência em 3D;
- não usa arquivos de debug como dependência.

Ver [`embedding_space.md`](embedding_space.md), [`feature_store.md`](feature_store.md) e [`dense_region_association.md`](dense_region_association.md).
