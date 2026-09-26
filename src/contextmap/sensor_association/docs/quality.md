# Qualidade da observação

Este documento descreve `src/contextmap/sensor_association/quality.py` (o contrato) e `quality_derivation.py` (a derivação), versão `observation-quality-v2`.

A `v2` (#562): `ReprojectionStatistics` ganhou `unevaluated_count` e passa a definir suas taxas sobre a população **avaliada** (`evaluated_count`, `invalid_rate`), além de recusar existir sem ao menos uma correspondência avaliada que projete. Um residual medido sobre uma população que a política de candidatos estreitou não é a mesma grandeza que um medido sobre o mapa inteiro, então não mantém a mesma identidade.

`ObservationQuality` guarda, como **medidas** separadas e tipadas, as condições que afetam o quão útil é uma observação espacial visual: a que distância está a geometria de apoio, quanto da região estava visível, com que densidade ela é suportada, onde está na imagem, quão bem a pose se alinha no tempo e, quando existe uma referência confiável, o resíduo de reprojeção.

## O que não é

Não é confiança semântica, similaridade CLIP/AlphaCLIP, probabilidade de o rótulo
estar correto nem peso de fusão obrigatório, e **não há um escalar combinado**.
Semantic Fusion já oferece uma política opcional ciente de qualidade que deriva
pesos de um subconjunto declarado destes componentes; esse cálculo pertence a
`contextmap.semantic_fusion`, não é definido aqui e não altera este contrato. A
qualidade se liga a uma `SpatialObservation` por identidade e nunca altera uma
claim semântica. Estruturalmente:

- não há campo `confidence`, `score`, `weight`, `probability` nem `label`, nem método que resuma o conjunto;
- `SemanticClaimRef` guarda só o `claim_id`: confiança de claim e qualidade de observação vivem em objetos diferentes.

## Componentes

| Componente | Definição |
| --- | --- |
| `associated_count`, `footprint_count` | geometria visível dentro da máscara, e geometria projetada de qualquer estado dentro dela |
| `mask_area_px` | pixels foreground da máscara |
| `support_density_per_mask_pixel` | `associated_count / mask_area_px` |
| `support_pixel_coverage` | fração dos pixels da máscara com geometria suportada (ver [`membership.md`](membership.md)) |
| `support_depth_m` | resumo (`count`, `minimum`, `median`, `maximum`) da profundidade da geometria associada, em metros, sob a `depth_metric` registrada |
| `support_off_axis_angle_rad` | resumo do ângulo da geometria associada com o eixo óptico, `arccos(z / alcance)`; definido em qualquer modelo de câmera |
| `border_distance_px` | resumo da distância do pixel preparado de cada ponto associado à borda mais próxima da imagem preparada, em pixels (proximidade de bordas e de regiões de fisheye inválidas) |
| `visible_share` | `associated_count / footprint_count` |
| `occluded_fraction`, `outside_valid_support_fraction` | parcelas da pegada ocluídas ou fora do suporte válido |
| `temporal_offset_ns` | `pose_ref.time_delta_ns`: a distância do timestamp do frame à pose contribuinte mais próxima (`pose_ref` também guarda o desfecho e a fração de interpolação) |
| `reprojection` | resíduo contra correspondências de referência **confiáveis**: `reference_id`, contagens (`invalid_count` são as que o modelo de câmera não projeta; uma que projeta fora da imagem é válida e entra no resíduo), média, mediana, p95 e máximo, em pixels |

## Ausência explícita

Uma medida que não pôde ser feita é `None` e é listada em `unavailable` com a razão; nunca vira zero. Um teste garante a consistência: `unavailable` cobre exatamente os componentes `None`, cada um com uma razão não vazia.

| Situação | Componentes indisponíveis |
| --- | --- |
| nenhuma geometria visível associada | profundidade, ângulo, distância à borda |
| pegada vazia (nada projetado dentro da máscara) | `visible_share`, `occluded_fraction`, `outside_valid_support_fraction` |
| sem correspondências de referência confiáveis | `reprojection` |

Uma densidade `0.0` com zero pontos associados é uma medida legítima, não uma ausência.

A reprojeção **nunca é fabricada**: a derivação só a anexa quando um conjunto confiável a forneceu.

## Proveniência

Cada qualidade guarda o `spatial_observation_id`, o `source_observation_id` (as regiões de um mesmo frame físico o compartilham, então inferências repetidas sobre o mesmo frame não são novas amostras de qualidade de vista), o mapa geométrico, a `CalibrationRef`, a `PoseRef`, a identidade da cadeia de imagem, a política de visibilidade e a versão das definições. O consumidor lê o contrato pela API pública, sem importar internos de câmera ou projeção; a avaliação estratifica pelos mesmos campos canônicos (distância, ângulo, borda, visibilidade, densidade, alinhamento temporal).

## Derivação

`derive_observation_quality(membership, observations, reprojection=None)` calcula cada componente a partir da projeção, da visibilidade e do pertencimento que produziram as observações, então as mesmas entradas dão a mesma qualidade. As observações devem ser as da associação, na ordem das suas regiões. Nada é estimado a partir da evidência semântica.

## Serialização

`encode_observation_quality` e `decode_observation_quality` usam só primitivas JSON, com `null` explícito para o que está indisponível, e a decodificação revalida o contrato (uma fração fora de `[0, 1]` ou uma ausência sem razão falha).

## Como é verificado

Cena sintética de valor conhecido: profundidade, ângulo (contra `atan2` independente), distância à borda, pegada, densidade e cobertura; alinhamento temporal com pose interpolada; região sem geometria; a reprojeção não fabricada; consistência da disponibilidade; separação estrutural da confiança semântica; proveniência; validação do contrato; ida e volta em JSON. Mutações que trocam `arccos` por `arcsin`, tratam centros como bordas, usam o denominador errado ou a métrica de profundidade errada fazem testes falharem.
