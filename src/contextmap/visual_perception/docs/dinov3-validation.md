# Validação real do backend DINOv3 (issue #69)

Relatório de uma execução **real** (pesos e frames reais; não é fake nem contrato) do `DinoV3DenseFeatureBackend`. Complementa [`dinov3.md`](dinov3.md) e segue o procedimento já usado para o DINOv2 ([`dinov2.md`](dinov2.md)).

## O que foi executado

| Item | Valor |
| --- | --- |
| Código | `dev` @ `ac1eb59` (adapter DINOv3 inalterado; a correção descrita abaixo está em `feature_diagnostics.py`) |
| Checkpoints | `facebook/dinov3-vitb16-pretrain-lvd1689m` @ `5931719e67bbdb9737e363e781fb0c67687896bc` (safetensors sha256 `9a21ac3d…dc8b`) e `facebook/dinov3-vits16-pretrain-lvd1689m` @ `114c1379950215c8b35dfcd4e90a5c251dde0d32` (sha256 `4610ad75…f91d`); o sha256 local foi conferido contra o `oid` LFS publicado pelo Hub |
| Licença | DINOv3 License (repositório *gated*, termos aceitos pela conta; `LICENSE.md` sha256 `25d122eb…999e`); pesos não versionados |
| Amostra | 20 frames reais de `corridor-02` (`selection.json` v2, identidade `sha256:dc641b345ffc142cbc50452bbaacef2433990478295f4720feb0f165ee4ed1c4`; hashes dos frames em `reports/frames.json`), sequência `e145f73f8d894f18b96ef1f55ca308c2` lida por payload, nunca inteira (issue #374) |
| Regiões | 429 regiões congeladas (com máscara) do run SAM2 da campanha anterior, mesmos 20 frames |
| Ambiente | Python 3.12.14, torch 2.14.0+cu130, transformers 5.17.0, torchvision 0.29, NumPy 2.3.5, Pillow 12.3, RTX 3060 (8 GB, compartilhada, sob `flock`) |
| Configuração | 448×336 (4:3, múltiplos de 16), `float32`, `cuda`, `local_files_only=True`, sem normalização L2 |

## Resultados

Todos os números são do run real; `Δ` é diferença máxima absoluta e `cos` o cosseno por patch.

**Carga.** Primeira chamada incluindo a carga: 1,3–1,6 s (ViT-B) e 1,0 s (ViT-S). GPU residente 335,5 MiB (ViT-B) e 91,4 MiB (ViT-S), pico 359,7 e 104,1 MiB. Saída `(21, 28, 768)` e `(21, 28, 384)`, `float32`, finita; `EmbeddingSpace(family="dinov3", layer="last_hidden_state.patch_tokens_after_registers")`; `stride = support = 16 × 640/448 ≈ 22,857` px, origem `(0, 0)`.

**Layout de tokens.**

- A contagem de registers é 4 pelo `config`, pelo parâmetro `embeddings.register_tokens` e pela aritmética (`593 = 1 + 4 + 21 × 28`).
- Contra um forward cru independente (resize/normalização escritos à mão, sem processor): Δ 7,6e-6 (ViT-B) e 8,0e-6 (ViT-S), cos mínimo 0,99999999998.
- Estímulo localizado: um bloco magenta de 64×64 px, colado em duas posições/frames, é localizado só com a geometria do `sampling` (as células cujo vetor mais muda com o estímulo). ViT-B: IoU 1,0 em três de quatro combinações posição×entrada e 0,88 na outra; erro de centroide 1,8–8,8 px (célula de 16–22,9 px); correlação de Pearson com a cobertura de pixel 0,75–0,99. ViT-S: IoU 1,0 nas quatro.
- Controles negativos sobre o mesmo forward cru: **não remover os registers** (descartar só o CLS) desloca o bloco exatamente 4 células (63–65 px em escala 1) e o IoU cai a 0; ler a sequência em ordem de colunas o desloca 82–424 px (IoU 0).
- Equivariância à translação em resolução nativa: recortes deslocados de 4 colunas e de 3 linhas dão pico de similaridade exatamente em `(0, 4)` e `(3, 0)` (cos médio 0,970 e 0,973 contra 0,841 e 0,781 sem deslocamento, ViT-B).

**Repetibilidade.** Δ 0,0 na mesma instância, depois de processar outro frame e em instância nova; os 20 payloads têm hash idêntico entre dois processos separados; `assert_repeatable_feature_outputs` passa entre instâncias.

**Precisão e dispositivo.** `float16` contra `float32`: finito, cos mínimo 0,999974 (Δ 1,2e-2). CPU `float32` contra GPU `float32`: cos mínimo 0,9999999998 (Δ 3,5e-5). `l2_normalize=True`: erro máximo de norma 1,2e-7 e cosseno preservado.

**Custo (20 frames, ViT-B).** Mediana 42 ms e p95 48 ms por frame (decodificação PNG, resize, inferência e cópia); 1,72 MiB de payload por frame (0,86 MiB no ViT-S); RSS de pico 1,36 GB.

**Pooling real.** 429 regiões SAM2 → 429 vetores, nenhuma falha (`EmptyRegionSupportError` = 0), 1–279 células por região (mediana 8), cobertura 1,0. Contra uma reimplementação independente da política `mask_weighted_mean_preserve_l2_v2` (integral image): Δ 2,4e-7, pesos e contagem de células idênticos nas 429. As células dentro da região ficam mais perto do vetor que as de fora do bbox em 429/429; cos entre vetor por máscara e por bbox: mediana 0,994 (mínimo 0,64); cos entre regiões do mesmo frame: mediana 0,44 (0,18–0,98), ou seja, os vetores não colapsam.

**Persistência.** Um `PerceptionRunArtifact` (`fe-dinov3-vitb16-20260921`, digest `sha256:5a58edf9…e97c`, 34,5 MiB, `verify_integrity` sem problemas) com o DINOv3 selecionado por `PipelinePreset`/`resolve_pipeline` reabre com o payload idêntico nos 20 frames. Com `FeatureDebugLevel.NONE` (nenhum `debug/`), o `DenseFeatureSampling` é reconstruído sem suposições de `metrics/feature-extraction.jsonl` em 20/20 e o pooling refeito só a partir do artefato coincide com o da memória (Δ 0,0).

**Falhas explícitas com o SDK real.** Revisão inexistente com `local_files_only` → `DinoV3ModelLoadError`; tamanho de metadata diferente do PNG, payload ausente e escape de caminho → `DinoV3InferenceError`; revisão móvel → `ValueError`. Nenhuma tem fallback.

## Defeito encontrado e corrigido

A reconstrução a partir do run real mostrou que o mapeamento célula→pixel **não era recuperável dos metadados persistidos**: a geometria densa só era gravada em `debug/` (ausente no nível `none`, o default do writer) e sem `origin_x`/`origin_y`, forçando o leitor a supor `(0, 0)`. Correção: `DenseFeatureDiagnostic` passa a exigir a origem e a geometria completa (com o tamanho da imagem preparada) entra nas métricas obrigatórias. Testes de regressão: `test_dense_sampling_is_rebuildable_from_required_metrics_without_debug`, `test_non_dense_metrics_carry_no_dense_geometry`, `test_dense_diagnostic_rejects_non_finite_origin` e `test_persisted_required_metrics_rebuild_the_dense_map_and_reproduce_pooling`. O estado anterior ficou registrado no relatório `30_run_artifact_vitb16_cpu_before-fix.json` (`persisted_event_has_origin: false`).

## Observações sem alteração

- **Entradas não múltiplas do patch.** `230×170` gera a grade `10×14` e deixa 2,6 % da largura e 5,9 % da altura sem célula. O `sampling` é fiel e o pooling informa `coverage_fraction`, mas nada avisa na configuração. O DINOv2 tem o mesmo comportamento; mudar só o DINOv3 quebraria a simetria dos dois adapters.
- **`coordinate_transform_id` inclui o fingerprint da configuração** (device, precisão, `payload_prefix`…): a mesma geometria em CPU, GPU e `float16` tem ids diferentes, e `assert_repeatable_feature_outputs` só vale sob o mesmo contexto.
- **`EmbeddingSpace` não inclui o dtype:** `float16` e `float32` compartilham `embedding_space_id`; diferem em `VisualFeature.dtype` e no fingerprint do backend.
- **TF32.** `precision="float32"` em GPU Ampere usa TF32 no conv do patch por default do cuDNN (`torch.backends.cudnn.allow_tf32=True`); o adapter não controla a flag, mas a diferença medida contra a CPU é de 3,5e-5 em valor absoluto.
- **Composition root.** `extract()` só devolve o `VisualFeature`; o `sampling` e o `EmbeddingSpace` exigem `extract_dense()`, então a composição real deve chamar este último no estágio (a validação usou um backend irmão, o que dobra a inferência). Também não existe helper público que converta `Region2D.mask` (`InlineMask` da imagem inteira) na máscara local ao bbox que `pool_region_feature` exige; o driver fez esse recorte.

## Limites

Uma cena de corredor, 20 frames, duas configurações; nenhuma avaliação científica da qualidade dos embeddings nem comparação com DINOv2. A repetibilidade vale para este hardware e software (GPU `sm_86`, cuDNN padrão).

## Reprodução

Os drivers e os relatórios JSON ficam fora do repositório, em `workspace/corridor-02/validation-feature-extraction-20260921/visual_perception/{scripts,reports}/` (sha256 dos drivers: `00_prepare_frames` `8aaf5585…`, `10_dinov3_real` `6d664489…`, `20_pooling_real` `9e3abdd8…`, `30_run_artifact` `8e83820b…`). Cada um roda com o Python do ambiente de auditoria, `PYTHONPATH` no `src/` do checkout e, na GPU, sob `flock /tmp/contextmap2-gpu.lock`.
