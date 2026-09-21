# Backend opcional PTv3

Este documento descreve `src/contextmap/point_representation/backends/ptv3.py`.

## O que é e o que não é

O adaptador permite selecionar um backbone **Point Transformer V3** como `PointEncoder` sem nenhuma mudança downstream. É **inferência** com um encoder PTv3 genérico como candidato/baseline de canal 3D aprendido.

**Não é** aprendizado de representação no estilo Sonata ou Vernata. Esses métodos acrescentam auto-destilação e, no Vernata, supervisão cross-modal 2D→3D de alta resolução; nada disso é implementado nem sugerido aqui. Um encoder pré-treinado ou destilado deve ser um backend **identificado separadamente**, sem mudar `PointEncoder` nem `PointRepresentation`. Para impedir o rótulo errado, uma configuração cujo `variant` ou `checkpoint` cite Sonata ou Vernata é rejeitada.

O pipeline canônico não exige PTv3: o backend é opcional e desabilitá-lo não altera nenhum contrato canônico de geometria ou evidência.

## Estado de verificação

O adaptador (`backends/ptv3.py`) é verificado com um runtime falso determinístico: configuração, identidade, encaminhamento das coordenadas, normalização, falhas explícitas, telemetria e isolamento de importação. Isso testa a **fronteira**, não a qualidade nem o custo de um PTv3.

O runtime real (`backends/ptv3_pointcept.py`, ver abaixo) tem testes determinísticos sem torch para tudo que o cerca: voxelização, agrupamento, verificação do hash do checkpoint, compatibilidade com a configuração e falha explícita por dependência ausente. A passagem direta real só roda nos testes opt-in `tests/point_representation/test_ptv3_pointcept_real.py`, que exigem GPU, o clone do Pointcept e o checkpoint (variáveis `CONTEXTMAP_POINTCEPT_ROOT` e `CONTEXTMAP_PTV3_CHECKPOINT`) e são ignorados nas demais máquinas.

**Ainda não registrado:** a execução dessa suíte opt-in, o tempo por suporte e o pico de VRAM medidos pela classe `PointceptPTv3Runtime`. O carregamento estrito do backbone com os argumentos de `NUSCENES_SEMSEG_PTV3M1_BASE` e uma passagem direta na GPU foram verificados em um protótipo descartável (não versionado), não pela classe.

## Configuração (`PTv3Config`)

| Campo | Significado |
| --- | --- |
| `variant` | identidade da arquitetura, por exemplo `ptv3-base` |
| `checkpoint` | identificador legível do checkpoint (não um caminho de arquivo) |
| `checkpoint_hash` | `sha256:<64 hex>` dos pesos; faz parte da identidade do espaço |
| `device` | dispositivo de execução; identidade de configuração, **não** do espaço |
| `precision` | `float32`, `float16` ou `bfloat16` |
| `grid_size_m` | tamanho da grade de serialização/voxelização, nas unidades das coordenadas preparadas |
| `output_dimension` | dimensão do vetor agrupado |
| `pooling` | `center` (feature do elemento central) ou `mean` (média dos elementos) |
| `normalization` | `none` ou `l2` |
| `min_support_points` | suportes menores falham sem chamar o runtime |
| `padding_channels` | canais de entrada além do XYZ que o checkpoint espera e o runtime preenche com **zeros** (por exemplo, a intensidade do LiDAR do modelo treinado no nuScenes); entra na `input_definition` e portanto na identidade do espaço; nenhum valor medido é fornecido a eles |

A configuração é validada antes de qualquer modelo ser construído e não contém segredos.

## Identidade

O `RepresentationSpace` tem `family = "ptv3"`, `model = "<variant>/<pooling>-pooling"`, `checkpoint = "<checkpoint>@<checkpoint_hash>"`, `dtype = "float32"`, `normalization` da configuração e `input_definition = "xyz-local-prepared;grid_size_m=<g>;precision=<p>"`, além da política de suporte. Tudo que muda o vetor (variante, pesos, precisão, grade, dimensão, pooling, normalização, política de suporte) muda o fingerprint, então a saída fica identificada por ele. O `device` altera só o fingerprint de configuração do `EncoderIdentity`, que também traz o `checkpoint_hash`. A entrada é somente XYZ local preparado: não há RGB, DINO, CLIP nem `SemanticClaim`, e nenhuma cabeça de classificação.

## `PTv3Runtime`

O runtime é a fronteira interna que isola torch, CUDA, PTv3 e o carregamento do checkpoint. `infer(coordinates_m=..., center_index=..., config=...)` devolve um `PTv3Inference` (`vector` de floats Python e `peak_memory_bytes` quando mensurável); nenhum tensor de framework cruza a fronteira. Um runtime real deve: verificar que os pesos correspondem a `checkpoint_hash`; serializar/voxelizar as coordenadas com `grid_size_m`; rodar apenas o backbone, sem cabeça de classificação; agrupar as features conforme `pooling`; medir o pico de memória do dispositivo; e levantar `PTv3OutOfMemoryError` para falta de memória de um suporte ou `PTv3RuntimeUnavailableError` para dependência, dispositivo ou checkpoint indisponível ou incompatível. A construção do runtime pertence à composição (`runtime`).

## Runtime real: `PointceptPTv3Runtime`

`backends/ptv3_pointcept.py` implementa `PTv3Runtime` sobre o código do Pointcept. É um runtime **só do backbone**: constrói o encoder-decoder PT-v3m1, carrega apenas os pesos `backbone.*` e **nunca constrói nem aplica a cabeça de classificação** (`seg_head`, descartada). As features por ponto são um descritor 3D aprendido da estrutura local, não um rótulo. O módulo não importa torch, spconv, torch-scatter, timm, Pointcept nem NumPy ao ser importado; tudo isso é importado no primeiro suporte codificado.

**Insumo e saída.** Os pontos do suporte que caem na mesma célula de `grid_size_m` viram um voxel no centróide (a convolução esparsa exige índices únicos); as coordenadas do voxel, mais `padding_channels` zeros, entram no backbone; as features por voxel voltam aos pontos do suporte e são agrupadas: `center` devolve a feature do voxel que contém o elemento central, `mean` a média sobre os **pontos** (um voxel conta uma vez por ponto que contém). O resultado não depende da ordem dos pontos.

**Inferência determinística.** O Pointcept sorteia uma permutação das curvas de serialização a cada passagem (`shuffle_orders`), o que faria o mesmo suporte gerar vetores diferentes; o runtime a desliga, além de `enable_flash` (FlashAttention não está instalado; a atenção por patch tem os mesmos parâmetros) e `drop_path`. Os pesos não dependem de qual curva cada bloco usa.

**Proveniência e identidade do checkpoint.**

| Item | Valor |
| --- | --- |
| Código | clone do repositório oficial `Pointcept/Pointcept`, commit `1342eda30e96cbb5fadf5374ff8eb18f6de15c71` (não é instalável por pip; licença MIT) |
| Pesos | HF `Pointcept/PointTransformerV3` (MIT), revisão `5e25cfaf759289b03db1f90f143b3dc5172929d3`, arquivo `nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth` |
| Tamanho e hash | 554 519 016 bytes; `sha256:e2774d2fa1dd33e640a514afe0bd1e5af94eb08be19d08d1f9402f20dfd6db94` (igual ao OID LFS publicado pelo HF) |
| Origem do treino | segmentação semântica supervisionada (16 classes) de varreduras LiDAR de veículo do nuScenes; é um baseline genérico, **não** adaptado a corredor interno |
| Backbone | 46 158 272 parâmetros, 4 canais de entrada (XYZ + intensidade, então `padding_channels=1`), 64 canais de saída |

O SHA-256 do arquivo é verificado contra `checkpoint_hash` **antes** de qualquer peso ser lido, e o carregamento é `strict`: um checkpoint que não corresponde ao backbone declarado falha em vez de gerar outra rede.

**`weights_only`.** O checkpoint de treino do Pointcept guarda também estado do otimizador e do scheduler e escalares NumPy, e não carrega com `torch.load(weights_only=True)`. Uma inspeção estática do pickle, sem executá-lo, listou apenas estes globais: `getattr`, `_codecs.encode`, `collections.OrderedDict`, `numpy.core.multiarray.scalar`, `numpy.dtype`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2` e `torch.optim.lr_scheduler.OneCycleLR`. O padrão do runtime é `weights_only=True`; `weights_only=False` é uma decisão explícita de quem chama, aceitável só para um arquivo de proveniência verificada e hash fixado, como este.

**Ambiente.** Um venv separado (`~/.cache/contextmap2-ptv3/venv`, fora do repositório): o venv de auditoria é CUDA 13 e o `spconv` só publica rodas até `cu126` (`spconv-cu130` não existe). Versões: Python 3.12, torch 2.8.0+cu126, torchvision 0.23.0+cu126, spconv-cu126 2.3.8, torch-scatter 2.1.2+pt28cu126 (roda oficial do índice PyG), timm 1.0.29 (instalado com `--no-deps`, para não substituir o torch), addict e NumPy 2.3.5. O `pointcept.models` completo exige também `torch_cluster`, `wandb` e módulos compilados que a inferência não usa; por isso o runtime importa só o módulo do backbone, com pacotes-stub apontando para o clone e uma classe vazia no lugar de `HookBase` (interface de treino).

**Medição de memória.** `peak_memory_bytes` é `torch.cuda.max_memory_allocated` da chamada: o pico de memória de tensores do processo, incluindo os pesos residentes e excluindo o contexto CUDA e o cache do alocador que o `nvidia-smi` também conta.

## Falhas explícitas, sem fallback

| Situação | Resultado |
| --- | --- |
| suporte menor que `min_support_points` | `UnencodableSupportError` → `FailedSupport`, sem chamar o runtime |
| falta de memória do dispositivo em um suporte | `UnencodableSupportError` → `FailedSupport`; contada em `telemetry.out_of_memory` |
| vetor de dimensão errada | `ValueError`; a execução para |
| vetor zero com `normalization = "l2"` | `UnencodableSupportError`; nada é inventado |
| valor não finito | `FailedSupport(NON_FINITE_OUTPUT)` pelo serviço |
| dependência ausente ou checkpoint incompatível | `PTv3RuntimeUnavailableError`; a execução para |

Nunca há queda silenciosa de PTv3 para o descritor determinístico ou para outro modelo: uma comparação entre eles é um run separado e explícito.

## Telemetria

`encoder.telemetry` (`PTv3Telemetry`) registra `encoded`, `out_of_memory`, `inference_seconds` (tempo dentro do runtime) e `peak_memory_bytes` (maior pico reportado; `None` quando o runtime nunca mediu, em vez de zero), para a ablação posterior contra `off` e contra o descritor. Batching não é implementado: um suporte por chamada, até que medições reais mostrem a necessidade.
