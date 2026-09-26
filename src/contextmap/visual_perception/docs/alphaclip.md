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

Zero regiões aceitas é um resultado legítimo de Region Discovery (um frame sem nada saliente), não um erro (#380): `extract()`/`extract_masked()` retornam zero features sem invocar o runtime nem exigir máscara, `AlphaClipExtraction.embedding_space` fica `None` (nenhum request foi codificado) e `array` fica com shape `(0, 0)`.

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

Os requests de uma chamada são montados e codificados em lotes consecutivos de no máximo `max_batch_size` (default 32). Com `full_image`, cada região replica a imagem inteira como tensor próprio, então sem esse limite uma cena densa definiria sozinha a memória da inferência. As linhas da saída seguem a ordem dos requests, o resultado dos lotes é o mesmo do lote único e o pico de memória cobre todos os lotes. `max_batch_size` entra no fingerprint da configuração, porque em GPU o tamanho do lote pode mudar o arredondamento em ponto flutuante (#617).

Para impedir download implícito, a configuração exige dois caminhos locais sob `checkpoint_root`:

- base CLIP checkpoint;
- AlphaCLIP vision checkpoint.

O pacote oficial importa `loralib` e `pkg_resources` (setuptools < 81) ao ser importado; quando um deles falta, `AlphaClipDependencyError` nomeia o módulo ausente em vez de apenas pedir `alpha_clip`.

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

Testes com runtime e mask source fakes cobrem transformações, bordas, erros de máscara, proveniência, identidade incompatível com CLIP, persistência/reabertura, diagnósticos e a mensagem de dependência ausente sem baixar pesos. **Isto é validação de contrato, não execução real.**

**Não há execução real do AlphaCLIP e a issue #71 permanece aberta.** Nada foi baixado nem executado: não foi obtido nenhum checkpoint AlphaCLIP de origem e integridade verificáveis. Fatos apurados em 2026-09-21:

- **Fonte oficial existe, mas sem âncora de integridade.** O repositório `SunzeY/AlphaCLIP` (Apache-2.0; commit `ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff` lido) é apontado pelo artigo (arXiv 2312.03818) e pela página do projeto. Seu `model-zoo.md` publica os `.pth` em Google Drive e no OpenXLab (`download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/*.pth`). Nem os autores nem as plataformas publicam SHA-256; a integridade só seria fixada por primeiro uso (o `checkpoint_fingerprint` do adapter é esse mecanismo).
- **Disponibilidade.** `download.openxlab.org.cn` não resolve nesta máquina (DNS, `curl` sai com 6); o Google Drive responde a página de visualização, e arquivos grandes exigem o fluxo de confirmação. Nenhum download foi tentado.
- **Hugging Face.** A conta dos autores (`Zery`, referenciada pelo README oficial) só tem spaces e datasets que apontam para o OpenXLab; os únicos repositórios `alphaclip` do Hub são de terceiros (`chouss/alpha_clip_final`, `Elise-hf/alphaClip`, `kasiv008/CLIP-16-alpha`), sem origem nem licença verificáveis. Não foram usados.
- **Formato e licença.** Os pesos são `.pth` com pickle (`alpha_clip.load` chama `torch.load`; com torch >= 2.6 o default `weights_only=True` recusa objetos arbitrários, mas o arquivo nunca foi inspecionado). Código Apache-2.0; dados CC BY-NC 4.0; checkpoints "para uso de pesquisa" e restritos à licença do CLIP.
- **Base CLIP.** O `ViT-L-14.pt` da OpenAI é público e verificável (SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` no caminho da URL do próprio pacote; `HEAD` responde 200). Não foi baixado: sem o checkpoint alpha não há o que validar.

**Auditoria estática do pacote oficial** (leitura do texto no commit acima; nada instalado nem executado). É compatível com as premissas do adapter: `alpha_clip.load(caminho_local, alpha_vision_ckpt_pth=..., device=...)` devolve `(model, preprocess)`, `model.visual(image, alpha)` devolve a projeção `(B, output_dim)`, a normalização RGB usa as constantes do CLIP e o alpha usa `(m - 0,5) / 0,26`, como o README. Riscos e diferenças:

1. o README redimensiona o alpha com interpolação bilinear (sobre tensor) e o adapter usa vizinho mais próximo por decisão explícita; qual das duas o modelo treinado prefere só se decide com o checkpoint real e uma medida;
2. `import alpha_clip` exige `loralib` e `pkg_resources`; o ambiente de auditoria tem `pkg_resources` (com aviso de remoção) e não tem `loralib`. O adapter mascarava isso com uma mensagem genérica (corrigido, com teste);
3. sem o checkpoint alpha, `build_model` zera `conv1_alpha`: o modelo se comporta como o CLIP comum e ignora o alpha em silêncio (o default `alpha_vision_ckpt_pth="None"` não falha). O adapter sempre exige e passa o caminho alpha, e o `checkpoint_fingerprint` cobre os dois arquivos, então esse caso não ocorre pelo caminho suportado;
4. `torch.load(alpha_vision_ckpt_pth)` não recebe `map_location`.

**Decisões que cabem ao revisor** para liberar a execução real:

1. aceitar Google Drive/OpenXLab da model zoo oficial como fonte, mesmo sem SHA-256 publicado, e autorizar um único download (por exemplo `clip_l14_grit20m_fultune_2xe.pth`), fixando o SHA-256 obtido como âncora;
2. aceitar a licença de pesquisa/CC BY-NC dos pesos para este repositório;
3. autorizar um venv próprio com o pacote oficial pinado no commit acima, `loralib` e `setuptools < 81`, e inspecionar os opcodes do pickle antes do `torch.load`;
4. decidir a política de interpolação do alpha (bilinear oficial contra vizinho mais próximo) por ablação com o checkpoint real.

Com a autorização, o procedimento é o do DINOv3 ([`dinov3-validation.md`](dinov3-validation.md)): as máscaras congeladas dos runs SAM2 sobre os mesmos 20 frames, confronto com um forward independente, repetibilidade entre processos, persistência pelo `PerceptionRunWriter` e diagnósticos.

## O que este backend não faz

- não altera ou cria máscaras;
- não exige AlphaCLIP para todas as features de região;
- não gera semântica ou scores;
- não presume compatibilidade com CLIP;
- não projeta evidência em 3D;
- não usa arquivos de debug como dependência.

Ver [`embedding_space.md`](embedding_space.md), [`feature_store.md`](feature_store.md) e [`dense_region_association.md`](dense_region_association.md).
