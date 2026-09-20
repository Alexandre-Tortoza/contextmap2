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

O espaço é distinto de DINO, AlphaCLIP e checkpoints CLIP diferentes. Essa
identidade permite que o `ClipSemanticScorer` implementado verifique
compatibilidade com text embeddings produzidos pelo mesmo modelo. O
`ClipVisualFeatureBackend` continua estritamente visual: não codifica texto,
não calcula similaridade e não escolhe labels; scoring é responsabilidade do
adapter separado em `backends/semantic_scoring.py`.

## Runtime Hugging Face

O runtime lazy decodifica a imagem uma vez, materializa os crops declarados, faz resize bicúbico direto configurado sem center crop, chama somente `CLIPModel.get_image_features()` e converte o resultado projetado para NumPy. A interpolação é explícita e não depende do default carregado pelo processor. PyTorch, Transformers e Pillow permanecem em `backends/` e só são importados na primeira execução.

`local_files_only=True` é o default; nenhuma inferência baixa pesos implicitamente. `revision` exige o SHA Git completo de 40 caracteres e rejeita referências móveis como `main` antes do carregamento. Dependência, device, checkpoint e inferência possuem erros separados e não acionam fallback.

O adapter rejeita valores não finitos e vetores nulos antes de declarar
normalização L2 ou enfileirar o payload.

## Diagnósticos

`ClipDiagnostics` preserva tempo de decode/preprocess/inferência, quantidade de
views e warnings de runtime/crop. A persistência comum de métricas e previews já
existe via `FeatureExtractionDiagnostic` e `PerceptionRunWriter`; nenhum arquivo
de debug é dependência downstream.

## Validação desta implementação

Os testes com runtime fake cobrem modos global/região, crops com contexto e borda, proveniência por view, persistência, normalização, compatibilidade de espaço, validação e ausência de scoring.

O backend também foi executado com pesos reais em frames de `corridor-02` durante a correção #339. Depois que o resize bicúbico passou a ser explícito em Pillow, os caminhos de processor PIL e torchvision diferiram no máximo `3.7e-7` e ambos coincidiram com uma referência Pillow independente. Isso valida o preprocessamento e a estabilidade numérica dessa configuração, não a qualidade semântica do embedding nem um benchmark científico do modelo.

## O que este backend não faz

- não gera labels;
- não codifica texto nem calcula scores;
- não altera região/máscara;
- não presume compatibilidade com DINO ou AlphaCLIP;
- não faz associação 2D↔3D;
- não oferece fallback silencioso.

Ver [`embedding_space.md`](embedding_space.md), [`feature_store.md`](feature_store.md) e [`ports.md`](ports.md).
