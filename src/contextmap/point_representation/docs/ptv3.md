# Backend opcional PTv3

Este documento descreve `src/contextmap/point_representation/backends/ptv3.py`.

## O que é e o que não é

O adaptador permite selecionar um backbone **Point Transformer V3** como `PointEncoder` sem nenhuma mudança downstream. É **inferência** com um encoder PTv3 genérico como candidato/baseline de canal 3D aprendido.

**Não é** aprendizado de representação no estilo Sonata ou Vernata. Esses métodos acrescentam auto-destilação e, no Vernata, supervisão cross-modal 2D→3D de alta resolução; nada disso é implementado nem sugerido aqui. Um encoder pré-treinado ou destilado deve ser um backend **identificado separadamente**, sem mudar `PointEncoder` nem `PointRepresentation`. Para impedir o rótulo errado, uma configuração cujo `variant` ou `checkpoint` cite Sonata ou Vernata é rejeitada.

O pipeline canônico não exige PTv3: o backend é opcional e desabilitá-lo não altera nenhum contrato canônico de geometria ou evidência.

## Estado de verificação

**Nenhuma execução real de PTv3 foi feita.** O ambiente de desenvolvimento não tem torch, Pointcept nem pesos de checkpoint. Os testes usam um runtime falso determinístico e verificam a **fronteira do adaptador** (configuração, identidade, encaminhamento das coordenadas, normalização, falhas explícitas, telemetria, isolamento de importação), não a qualidade nem o custo de um PTv3 real. Continuam pendentes: uma implementação de `PTv3Runtime` sobre torch, a escolha de checkpoint, a medição de tempo e de VRAM reais e a avaliação contra `off` e contra o descritor determinístico (`evaluation`).

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

A configuração é validada antes de qualquer modelo ser construído e não contém segredos.

## Identidade

O `RepresentationSpace` tem `family = "ptv3"`, `model = "<variant>/<pooling>-pooling"`, `checkpoint = "<checkpoint>@<checkpoint_hash>"`, `dtype = "float32"`, `normalization` da configuração e `input_definition = "xyz-local-prepared;grid_size_m=<g>;precision=<p>"`, além da política de suporte. Tudo que muda o vetor (variante, pesos, precisão, grade, dimensão, pooling, normalização, política de suporte) muda o fingerprint, então a saída fica identificada por ele. O `device` altera só o fingerprint de configuração do `EncoderIdentity`, que também traz o `checkpoint_hash`. A entrada é somente XYZ local preparado: não há RGB, DINO, CLIP nem `SemanticClaim`, e nenhuma cabeça de classificação.

## `PTv3Runtime`

O runtime é a fronteira interna que isola torch, CUDA, PTv3 e o carregamento do checkpoint. `infer(coordinates_m=..., center_index=..., config=...)` devolve um `PTv3Inference` (`vector` de floats Python e `peak_memory_bytes` quando mensurável); nenhum tensor de framework cruza a fronteira. Um runtime real deve: verificar que os pesos correspondem a `checkpoint_hash`; serializar/voxelizar as coordenadas com `grid_size_m`; rodar apenas o backbone, sem cabeça de classificação; agrupar as features conforme `pooling`; medir o pico de memória do dispositivo; e levantar `PTv3OutOfMemoryError` para falta de memória de um suporte ou `PTv3RuntimeUnavailableError` para dependência, dispositivo ou checkpoint indisponível ou incompatível. A construção do runtime pertence à composição (`runtime`).

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
