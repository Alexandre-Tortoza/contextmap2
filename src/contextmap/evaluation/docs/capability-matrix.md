# Matriz de capacidades e compatibilidade (#522)

`contextmap.evaluation.capability_matrix` congela, como dados tipados, o que cada família de modelo do experimento do milestone #22 faz nativamente, que papel do ContextMap2 ela ocupa, se já é selecionável pelo runtime e quais composições produtor → consumidor são admissíveis. A versão atual é `1.1.0` (`CAPABILITY_MATRIX_VERSION`); repositórios e model cards upstream foram consultados em 2026-09-25.

A matriz descreve composições experimentais admissíveis; ela **não** escolhe um pipeline canônico. Um modelo só ocupa vários papéis por contratos de evidência explícitos de cada capability: geometria, hipótese semântica e crença persistente nunca se fundem só porque saem da mesma inferência.

## Modelo de dados

| Tipo | Conteúdo |
|---|---|
| `Capability` | uma operação nativa de uma família (ou uma operação do próprio ContextMap2 que a consome): backend, operação nativa, papel e adapter, status, `RuntimeBinding`, evidência de entrada e saída, geometria, controles de prompt/tarefa e de modelo/runtime, campos de proveniência, limitações, papéis compatíveis a montante e a jusante, grupo de comparação, família de métricas, motivo do status e issue |
| `RuntimeBinding` | estágio do preset canônico, componente e backend e os `settings` que selecionam a operação nativa (`task` do Florence-2, `strategy` do SAM3, `policy_id` de uma query do LocateAnything) |
| `Control` | um parâmetro (ou grupo reservado) e seu status; controles ainda não disponíveis citam a issue que os entrega |
| `Composition` | uma aresta produtor → consumidor com status, nota, issue, grupo de comparação e família de métricas |

### Status de uma capability

| Status | Significado |
|---|---|
| `supported` | selecionável hoje pela configuração do runtime: um teste confere que o estágio existe no preset canônico e que o par componente/backend está no catálogo e na composition root |
| `planned` | uma issue aberta do milestone vai torná-la selecionável; se a entrada já nomeia a identidade futura, um teste confere que ela **ainda não** existe (quando existir, a entrada precisa virar `supported`) |
| `blocked` | depende de código ou pesos upstream que não estão publicados |
| `out_of_scope` | nativa do modelo, mas fora do experimento; não ocupa papel nem grupo de comparação |

### Status de uma composição

| Status | Significado |
|---|---|
| `supported` | o runtime entrega hoje a evidência do produtor ao consumidor |
| `planned` | uma issue do milestone vai entregá-la; até lá é excluída |
| `blocked` | espera assets upstream |
| `incompatible` | o contrato do consumidor não usa a evidência do produtor |
| `not_scientifically_comparable` | executa, mas o resultado não substitui o caminho que parece substituir |

Uma aresta **não declarada** não é admissível. `require_supported_composition(produtor, consumidor)` recusa, antes de carregar qualquer modelo, toda aresta que não seja `supported`, com status, issue e motivo; é o guard que os manifestos do #527 consomem.

## Grupos de comparação

O grupo codifica **semântica da tarefa + condicionamento da entrada + geometria de saída**. `comparable(a, b)` só é verdadeiro no mesmo grupo; uma entrada sem grupo não se compara com nada. Emitir caixa ou máscara nunca basta:

- LocateAnything `category_detection` (caixas condicionadas por categorias da query) e Florence-2 `<OD>` (caixas com rótulo gerado pelo vocabulário de treino) **não** se comparam;
- SAM2 automático (máscaras sem condicionamento) e SAM3 texto (máscaras de todas as instâncias de um conceito) **não** se comparam, nem SAM2 automático com `<REGION_PROPOSAL>` (máscara × caixa);
- SAM3 texto e a composição LocateAnything categoria → refinamento SAM2 compartilham `category_conditioned_instance_masks`: é a comparação de topologia do #521;
- Qwen, Gemini e Eagle 2.5 compartilham os grupos `*_interpretation.instruction_following`, sob a mesma política de prompt, views e schema (#545); as tasks do Florence-2 têm prompt nativo e ficam em grupos `*.task_native`;
- DINOv2, DINOv3 e SigLIP2 denso compartilham `dense_visual_features`; CLIP (crop) e AlphaCLIP (máscara) têm condicionamento diferente e grupos diferentes, assim como os dois scorers, cujas similaridades vivem em espaços distintos.

Comparar operações de grupos diferentes (por exemplo, caixa × máscara numa métrica de caixa) exige declarar a diferença de tarefa ou geometria como fator do experimento.

## Como o plano do #521 sai da matriz

| Fase do #521 | O que a matriz fornece |
|---|---|
| 1. grade de capability/backend | grupos com dois ou mais membros `supported`: `dense_visual_features`, `scene_interpretation.*`, `region_interpretation.*` |
| 2. tarefa/estratégia nativa | as tasks do Florence-2, a estratégia do SAM3 e as políticas do LocateAnything são capabilities distintas (`RuntimeBinding.settings`); compará-las é ablação de tarefa, não de backend |
| 3. grounding/refinamento | LocateAnything → refinamento SAM2 (`supported`, #568) × SAM3 texto; caixa sem máscara → associação é `incompatible` |
| 4. features densas/de região | `dense_visual_features`; amostragem 2D→3D ainda `planned` |
| 5. intérprete semântico | Qwen × Gemini × Eagle 2.5 no mesmo grupo; Florence-2 à parte |
| 6–9. prompt, views, contexto, orçamento | controles: `prompt_policy` e orçamentos visuais (`min_pixels`/`max_pixels` do Qwen, tiles do Eagle 2.5) `supported`; views (#524), contexto de cena (#529) e política de requisição (#544) `planned` |
| 10–11. composições e confirmação downstream | as arestas `supported` até Semantic Fusion |

## Achados que as fases seguintes precisam saber

- **Caixa não entra na associação 2D→3D.** O pertencimento é definido pela máscara: uma região só com caixa (LocateAnything, `<OD>`, `<REGION_PROPOSAL>`, ...) vira `NO_INLINE_MASK` e não sustenta nenhum ponto. A composição "grounding → associação direta" (#521 B, #528 P1, #577 P1) precisa de uma política de pertencimento por caixa que nenhuma issue do milestone implementa.
- **Regiões de grounding chegam tarde.** O executor de percepção roda descoberta, features e interpretação e só depois acrescenta as regiões do grounding ao resultado; por isso elas não recebem features de região nem interpretação. As arestas LocateAnything → CLIP e → views estão `planned` sob o #577, sem issue de implementação dona.
- **Amostragem densa não está ligada.** `sample_dense_features` funciona com DINO e SigLIP2, mas o `SensorAssociationExecutor` não passa canais densos.
- **Scorers fora do runtime.** `ClipSemanticScorer` e `AlphaClipSemanticScorer` existem, mas só no grafo interno da percepção; o slot de features de região fixa o escopo `region`, então a feature CLIP global do scorer de cena não é produzida.
- **Ponto não vira caixa.** `locateanything.pointing` fica no stream de grounding; não há contrato canônico de ponto.
- **Bloqueados:** LocateAnything com prompt visual (#574; os pesos publicados não suportam) e LocateAnything3D (#575; o repositório público só tem o título do README e não há pesos da NVIDIA publicados).
- **Florence-2 `<OPEN_VOCABULARY_DETECTION>`** devolve caixas ou polígonos conforme a resposta, então sua associação é `not_scientifically_comparable`.

## Fontes consultadas (2026-09-25)

- SAM2: README de `facebookresearch/sam2` (gerador automático de máscaras, predictor de imagem por pontos/caixas, predictor de vídeo).
- SAM3: README de `facebookresearch/sam3` (segmentação por conceito com frase curta ou exemplares, detector e tracker de vídeo, checkpoint com acesso controlado).
- Florence-2: `processing_florence2.py` de `microsoft/Florence-2-large` (lista de tasks e tipo de pós-processamento de cada uma).
- LocateAnything: `NVlabs/Eagle/Embodied/README.md` e o model card `nvidia/LocateAnything-3B` (detecção, grounding de frase, pontos, texto/GUI/layout, modos fast/slow/hybrid, prompt visual sem suporte nos pesos publicados).
- Eagle 2.5: `NVlabs/Eagle/Eagle2_5/README.md` (SigLIP2-So400m-Patch16-512 + Qwen2.5-7B, imagens em alta resolução, até 512 frames de vídeo).
- SigLIP2: documentação do adapter (#572), variantes FixRes e NaFlex.
- LocateAnything3D: `NVlabs/LocateAnything3D` (README só com o título) e busca de modelos no Hugging Face.

## Tabelas

As tabelas abaixo são geradas de `CAPABILITY_MATRIX`; `tests/evaluation/test_capability_matrix.py` falha, mostrando o texto esperado, quando elas divergem dos dados. Controles, evidência, proveniência e limitações de cada entrada estão no módulo.

### Capabilities

<!-- capabilities:start (gerado de CAPABILITY_MATRIX; não editar à mão) -->
| id | backend | operação nativa | papel | status | issue | grupo de comparação |
|---|---|---|---|---|---|---|
| `sam2.automatic_mask_generation` | SAM2 | automatic mask generation (point-grid prompts, IoU/stability filter) | region_discovery | supported | — | automatic_region_proposal.mask |
| `sam2.box_prompt_refinement` | SAM2 | promptable image segmentation from a box or points (image predictor) | region_refinement | supported | — | prompted_mask_refinement |
| `sam2.video_tracking` | SAM2 | video object segmentation with memory (masklets across frames) | — | out_of_scope | — | — |
| `sam3.text_concept_segmentation` | SAM3 | promptable concept segmentation of every instance of a noun phrase | region_discovery | supported | — | category_conditioned_instance_masks |
| `sam3.exemplar_prompts` | SAM3 | concept segmentation from visual exemplars (positive/negative boxes) | — | out_of_scope | — | — |
| `sam3.video_tracking` | SAM3 | concept detection and tracking in video (video predictor) | — | out_of_scope | — | — |
| `florence2.region_proposal` | Florence-2 | <REGION_PROPOSAL> class-agnostic boxes | region_discovery | supported | — | automatic_region_proposal.box |
| `florence2.object_detection` | Florence-2 | <OD> boxes with a generated category name | region_discovery | supported | — | generated_label_detection.box |
| `florence2.dense_region_caption` | Florence-2 | <DENSE_REGION_CAPTION> boxes with a generated description | region_discovery | supported | — | dense_region_caption.box |
| `florence2.open_vocabulary_detection` | Florence-2 | <OPEN_VOCABULARY_DETECTION> boxes or polygons for a text input | region_discovery | supported | — | open_vocabulary_detection.box_or_mask |
| `florence2.caption_phrase_grounding` | Florence-2 | <CAPTION_TO_PHRASE_GROUNDING> boxes for the phrases of a caption | region_discovery | supported | — | caption_phrase_grounding.box |
| `florence2.referring_expression_segmentation` | Florence-2 | <REFERRING_EXPRESSION_SEGMENTATION> polygon mask of a text referent | region_discovery | supported | — | referring_expression_segmentation.mask |
| `florence2.region_to_segmentation` | Florence-2 | <REGION_TO_SEGMENTATION> polygon mask of an input box | region_discovery | supported | — | — |
| `florence2.scene_caption` | Florence-2 | <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION> scene text | semantic_interpretation | supported | — | scene_caption.task_native |
| `florence2.region_category` | Florence-2 | <REGION_TO_CATEGORY> category of a region | semantic_interpretation | supported | — | region_category.task_native |
| `florence2.region_description` | Florence-2 | <REGION_TO_DESCRIPTION> description of a region | semantic_interpretation | supported | — | region_description.task_native |
| `florence2.ocr` | Florence-2 | <OCR>, <OCR_WITH_REGION>, <REGION_TO_OCR> text reading | — | out_of_scope | — | — |
| `dinov2.dense_patch_features` | DINOv2 | self-supervised patch tokens (CLS and registers removed) | dense_features | supported | — | dense_visual_features |
| `dinov3.dense_patch_features` | DINOv3 | self-supervised patch tokens (CLS and 4 registers removed) | dense_features | supported | — | dense_visual_features |
| `siglip2.dense_patch_features` | SigLIP2 | vision-encoder patch tokens (last_hidden_state, FixRes checkpoints) | dense_features | supported | — | dense_visual_features |
| `siglip2.global_embedding` | SigLIP2 | attention-pooled image embedding (pooler_output) | — | out_of_scope | — | — |
| `clip.region_embedding` | CLIP | image projection of a region crop | region_features | supported | — | region_crop_embedding |
| `alphaclip.region_embedding` | AlphaCLIP | image projection conditioned on an alpha (mask) channel | region_features | supported | — | mask_conditioned_region_embedding |
| `clip.semantic_scoring` | CLIP | text-image cosine similarity of scene claims and the global feature | semantic_scoring | planned | #527 | claim_support.clip_global |
| `alphaclip.semantic_scoring` | AlphaCLIP | text-image cosine similarity of region claims and the same region | semantic_scoring | planned | #527 | claim_support.alphaclip_region |
| `contextmap2.semantic_view_assembly` | ContextMap2 | content-addressed semantic views of the frame or of one region | semantic_view_assembly | supported | — | — |
| `qwen.scene_interpretation` | Qwen | scene interpretation from an instruction and ordered images | semantic_interpretation | supported | — | scene_interpretation.instruction_following |
| `qwen.region_interpretation` | Qwen | region interpretation from an instruction and ordered images | semantic_interpretation | supported | — | region_interpretation.instruction_following |
| `gemini.scene_interpretation` | Gemini | scene interpretation from an instruction and ordered images | semantic_interpretation | supported | — | scene_interpretation.instruction_following |
| `gemini.region_interpretation` | Gemini | region interpretation from an instruction and ordered images | semantic_interpretation | supported | — | region_interpretation.instruction_following |
| `eagle2_5.scene_interpretation` | Eagle 2.5 | scene interpretation from an instruction and ordered images | semantic_interpretation | supported | — | scene_interpretation.instruction_following |
| `eagle2_5.region_interpretation` | Eagle 2.5 | region interpretation from an instruction and ordered images | semantic_interpretation | supported | — | region_interpretation.instruction_following |
| `eagle2_5.long_context_video` | Eagle 2.5 | long-context video and multi-image reasoning (up to 512 frames) | — | out_of_scope | — | — |
| `locateanything.category_detection` | LocateAnything | detect every instance of an ordered category set (PBD boxes) | region_grounding | supported | — | category_conditioned_detection.box |
| `locateanything.phrase_grounding` | LocateAnything | ground every instance matching a free phrase (PBD boxes) | region_grounding | supported | — | phrase_grounding.box |
| `locateanything.pointing` | LocateAnything | point to what a phrase refers to | region_grounding | supported | — | phrase_pointing.point |
| `locateanything.visual_prompt_grounding` | LocateAnything | detection with an image crop as the query (visual prompt) | region_grounding | blocked | #574 | — |
| `locateanything.text_gui_layout` | LocateAnything | scene-text detection, GUI element grounding, layout grounding | — | out_of_scope | — | — |
| `locateanything3d.open_vocabulary_3d_proposal` | LocateAnything3D | open-vocabulary 3D box proposals from images | proposal_3d | blocked | #575 | — |
| `contextmap2.metric_3d_verification` | ContextMap2 | verify a model-predicted 3D proposal against LiDAR/depth geometry | metric_verification | blocked | #575 | — |
| `contextmap2.mask_membership_association` | ContextMap2 | assign visible projected map points to the frozen region masks | sensor_association | supported | — | — |
| `contextmap2.dense_feature_sampling` | ContextMap2 | map each visible point to the cells of a dense feature map | dense_feature_sampling | planned | #577 | — |
| `contextmap2.semantic_fusion` | ContextMap2 | accumulate evidence over supports grouped by physical observation | semantic_fusion | supported | — | — |
<!-- capabilities:end -->

### Composições

<!-- compositions:start (gerado de CAPABILITY_MATRIX; não editar à mão) -->
| produtor | consumidor | status | issue | grupo de comparação |
|---|---|---|---|---|
| `locateanything.category_detection` | `sam2.box_prompt_refinement` | supported | — | category_conditioned_instance_masks |
| `locateanything.phrase_grounding` | `sam2.box_prompt_refinement` | supported | — | phrase_conditioned_instance_masks |
| `locateanything.pointing` | `sam2.box_prompt_refinement` | supported | — | — |
| `sam2.box_prompt_refinement` | `contextmap2.mask_membership_association` | supported | — | — |
| `locateanything.category_detection` | `contextmap2.mask_membership_association` | incompatible | — | — |
| `florence2.object_detection` | `contextmap2.mask_membership_association` | incompatible | — | — |
| `florence2.open_vocabulary_detection` | `contextmap2.mask_membership_association` | not_scientifically_comparable | — | — |
| `sam2.automatic_mask_generation` | `contextmap2.mask_membership_association` | supported | — | — |
| `sam3.text_concept_segmentation` | `contextmap2.mask_membership_association` | supported | — | — |
| `florence2.referring_expression_segmentation` | `contextmap2.mask_membership_association` | supported | — | — |
| `dinov2.dense_patch_features` | `contextmap2.dense_feature_sampling` | planned | #577 | — |
| `dinov3.dense_patch_features` | `contextmap2.dense_feature_sampling` | planned | #577 | — |
| `siglip2.dense_patch_features` | `contextmap2.dense_feature_sampling` | planned | #577 | — |
| `sam2.automatic_mask_generation` | `alphaclip.region_embedding` | supported | — | — |
| `sam3.text_concept_segmentation` | `alphaclip.region_embedding` | supported | — | — |
| `florence2.object_detection` | `alphaclip.region_embedding` | incompatible | — | — |
| `locateanything.category_detection` | `alphaclip.region_embedding` | incompatible | — | — |
| `sam2.automatic_mask_generation` | `clip.region_embedding` | supported | — | — |
| `florence2.object_detection` | `clip.region_embedding` | supported | — | — |
| `locateanything.category_detection` | `clip.region_embedding` | planned | #577 | — |
| `sam2.automatic_mask_generation` | `contextmap2.semantic_view_assembly` | supported | — | — |
| `florence2.object_detection` | `contextmap2.semantic_view_assembly` | supported | — | — |
| `locateanything.category_detection` | `contextmap2.semantic_view_assembly` | planned | #577 | — |
| `contextmap2.semantic_view_assembly` | `qwen.scene_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `qwen.region_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `gemini.scene_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `gemini.region_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `eagle2_5.scene_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `eagle2_5.region_interpretation` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `florence2.scene_caption` | supported | — | — |
| `contextmap2.semantic_view_assembly` | `florence2.region_category` | supported | — | — |
| `qwen.region_interpretation` | `alphaclip.semantic_scoring` | planned | #527 | — |
| `alphaclip.region_embedding` | `alphaclip.semantic_scoring` | planned | #527 | — |
| `qwen.scene_interpretation` | `clip.semantic_scoring` | planned | #527 | — |
| `clip.region_embedding` | `clip.semantic_scoring` | incompatible | — | — |
| `contextmap2.mask_membership_association` | `contextmap2.semantic_fusion` | supported | — | — |
| `locateanything3d.open_vocabulary_3d_proposal` | `contextmap2.metric_3d_verification` | blocked | #575 | — |
| `contextmap2.metric_3d_verification` | `contextmap2.semantic_fusion` | blocked | #575 | — |
<!-- compositions:end -->
