# Backend de features visuais CLIP

Este documento descreve `src/contextmap/visual_perception/backends/clip.py`, o papel exclusivamente visual de CLIP como `FeatureExtractor` global ou de região.

## Dois modos configurados, um port

`ClipConfig.scope` seleciona um dos modos:

- `GLOBAL`: a imagem preparada inteira gera um `VisualFeature` global;
- `REGION`: cada `Region2D` aceita e congelada gera um `VisualFeature` com seu `region_id`.

O modo faz parte da configuração do backend, não de um branch downstream. `extract()` retorna metadata canônica e enfileira cada vetor pelo mesmo sink de payload do `PerceptionRunWriter`. `extract_visual()` também devolve arrays, views, `EmbeddingSpace` e diagnósticos no mesmo passe.

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

O runtime lazy decodifica a imagem uma vez, materializa os crops declarados, faz resize direto configurado sem center crop, chama somente `CLIPModel.get_image_features()` e converte o resultado projetado para NumPy. PyTorch, Transformers e Pillow permanecem em `backends/` e só são importados na primeira execução.

`local_files_only=True` é o default; nenhuma inferência baixa pesos implicitamente. Dependência, device, checkpoint e inferência possuem erros separados e não acionam fallback.

## Diagnósticos

`ClipDiagnostics` preserva tempo de decode/preprocess/inferência, quantidade de views e warnings de runtime/crop. A persistência geral de evidência/diagnóstico será integrada pela issue #72; nenhum arquivo de debug é dependência downstream.

## Validação desta implementação

Os testes com runtime fake cobrem modos global/região, crops com contexto e borda, proveniência por view, persistência, normalização, compatibilidade de espaço, validação e ausência de scoring.

Por instrução do usuário, nenhum checkpoint real foi baixado ou executado. A issue #70 requer uma execução controlada na máquina de inferência para confirmar a projeção/dimensão do checkpoint, tolerância numérica e comportamento do processor antes do merge.

## O que este backend não faz

- não gera labels;
- não codifica texto nem calcula scores;
- não altera região/máscara;
- não presume compatibilidade com DINO ou AlphaCLIP;
- não faz associação 2D↔3D;
- não oferece fallback silencioso.

Ver [`embedding_space.md`](embedding_space.md), [`feature_store.md`](feature_store.md) e [`ports.md`](ports.md).
