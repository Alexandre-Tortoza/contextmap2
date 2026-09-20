# Backend de features visuais CLIP

Este documento descreve `src/contextmap/visual_perception/backends/clip.py`, o papel exclusivamente visual de CLIP como `FeatureExtractor` global ou de região.

## Dois modos configurados, um port

`ClipConfig.scope` seleciona um dos modos:

- `GLOBAL`: a imagem preparada inteira gera um `VisualFeature` global;
- `REGION`: cada `Region2D` aceita e congelada gera um `VisualFeature` com seu `region_id`.

O modo faz parte da configuração do backend, não de um branch downstream. `extract()` retorna metadata canônica e enfileira cada vetor pelo mesmo sink de payload do `PerceptionRunWriter`. `extract_visual()` também devolve arrays, views, `EmbeddingSpace` e diagnósticos no mesmo passe.

O composition root fornece `feature_stage_id`. Seu SHA-256 cria o namespace
dos `FeatureId` e das referências de payload, de modo que CLIP global/região
possa coexistir com DINO ou outro feature stage no mesmo `PerceptionResult`.

## View de região e proveniência

CLIP não recebe a geometria abstrata: recebe uma view RGB concreta. `ClipView` registra para cada vetor:

- região de origem, ou `None` no modo global;
- crop exato em pixels da imagem preparada;
- política `tight_box` ou `context_box:<fraction>`;
- dimensões da imagem fonte e da entrada do modelo;
- identidade determinística da transformação imagem→crop→resize.

`context_box` expande cada lado por uma fração configurada da largura/altura da região e recorta nos limites da imagem. Clipping gera warning. Região sem suporte na imagem falha explicitamente. A `Region2D` original nunca é modificada.

A proveniência de cada `VisualFeature` combina o fingerprint do modelo/configuração com a transformação daquela view. Duas regiões do mesmo frame não perdem a diferença de suporte apenas porque usam o mesmo checkpoint.

## Espaço de embedding

O espaço canônico registra `family="clip"`, checkpoint+revision, `layer="image_projection"`, dimensão real e normalização (`l2` por default). Global e região produzidos pelo mesmo checkpoint/projeção/normalização pertencem ao mesmo `EmbeddingSpace`: crop/contexto muda a evidência visual, não a definição matemática do espaço.

O espaço é distinto de DINO, AlphaCLIP e checkpoints CLIP diferentes. A identidade preservada é suficiente para um futuro `SemanticScorer` verificar compatibilidade com text embeddings produzidos pelo mesmo modelo, mas este adapter não codifica texto, não calcula similaridade e não escolhe labels.

## Runtime Hugging Face

O runtime lazy decodifica a imagem uma vez, materializa os crops declarados, faz o resize bicúbico direto configurado no **Pillow**, sem center crop, usa o processor apenas para rescale e normalização (`do_resize=False`), chama somente `CLIPModel.get_image_features()` e converte o resultado projetado para NumPy. O resize é do adapter, não do processor: a implementação de resize do processor depende do backend instalado (torchvision ou PIL) e dá pixels diferentes (até 1,75e-2) e features diferentes (cosseno mínimo por patch 0,981) para o mesmo checkpoint, config e imagem. Com o resize no Pillow, os dois backends concordam a 2,4e-7 (issue #339). Dependências: `torch`, `transformers` (>= 4.56, que introduziu `dtype=`) e `Pillow`; com `transformers` 5.x o processor padrão importa também `torchvision`, que precisa estar instalado; um pacote ausente vira `ClipDependencyError`. PyTorch, Transformers e Pillow permanecem em `backends/` e só são importados na primeira execução.

`local_files_only=True` é o default; nenhuma inferência baixa pesos implicitamente. `revision` exige o SHA Git completo de 40 caracteres e rejeita referências móveis como `main` antes do carregamento. Dependência, device, checkpoint e inferência possuem erros separados e não acionam fallback.

O adapter rejeita valores não finitos e vetores nulos antes de declarar
normalização L2 ou enfileirar o payload.

## Diagnósticos

`ClipDiagnostics` preserva tempo de decode/preprocess/inferência, quantidade de views e warnings de runtime/crop. A persistência geral de evidência/diagnóstico será integrada pela issue #72; nenhum arquivo de debug é dependência downstream.

## Validação desta implementação

Os testes com runtime fake cobrem modos global/região, crops com contexto e borda, proveniência por view, persistência, normalização, compatibilidade de espaço, validação e ausência de scoring.

Execução controlada com pesos reais (issue #70, 2026-09-20; RTX 3060, torch 2.14, transformers 5.17, frames reais de `corridor-02`): `openai/clip-vit-large-patch14` no escopo global produz `(1, 768)` com norma L2 1,0 e repetição idêntica. O vetor está no espaço conjunto imagem-texto: para um frame de corredor, "a photo of an indoor corridor" pontua 0,229 contra 0,137 (cat), 0,118 (forest) e 0,109 (beach). No escopo de região com crops `context_box`, há um vetor por região com `feature_id` único e `region_id` preservado, e uma região não aceita é rejeitada com `ClipInferenceError`. fp16 contra fp32 tem cosseno 0,999994.

Antes da correção do pré-processamento (issue #339) o adapter tinha cosseno 0,99995 (diferença máxima 1,2e-3) contra uma referência independente; depois dela a diferença máxima é 3,6e-7 e os backends torchvision e PIL do processor coincidem.

## O que este backend não faz

- não gera labels;
- não codifica texto nem calcula scores;
- não altera região/máscara;
- não presume compatibilidade com DINO ou AlphaCLIP;
- não faz associação 2D↔3D;
- não oferece fallback silencioso.

Ver [`embedding_space.md`](embedding_space.md), [`feature_store.md`](feature_store.md) e [`ports.md`](ports.md).
