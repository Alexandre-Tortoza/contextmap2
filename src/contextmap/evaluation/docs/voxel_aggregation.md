# Avaliação pareada: mapa bruto × agregação voxel inter-scan

Este documento descreve `src/contextmap/evaluation/voxel_aggregation.py`: as métricas que a issue #624 usa para decidir se uma agregação voxel inter-scan ([`inter-scan-aggregation.md`](../../geometric_mapping/docs/inter-scan-aggregation.md)) pode acompanhar ou substituir a geometria bruta. **As definições foram fixadas aqui antes de qualquer execução real ser inspecionada.** O experimento só fornece entradas e os limiares declarados abaixo.

Nada aqui altera um artifact. Escala, fidelidade geométrica e estabilidade de associação ficam em seções separadas, e nenhuma é resumida em um score único.

## Fidelidade geométrica (`evaluate_aggregation_fidelity`)

Medida sobre **todos** os pontos brutos, não sobre uma amostra.

| Campo | Definição |
|---|---|
| `distance_m` | para cada ponto bruto `p` no agregado `a`: `‖p − centroid(a)‖`; mínimo, mediana, p95 por posto mais próximo, máximo |
| `mean_distance_m` | média da mesma distância |
| `by_range` | a mesma distância por faixa de alcance ao sensor. O alcance é `‖p − o‖`, onde `o` é a origem do sensor posta no mapa pela cadeia do próprio scan; a cadeia é rígida, então é o alcance medido |
| `reduction_ratio` | `aggregate_count / source_point_count` |
| `statistics_bounds_preserved` | o envelope dos mínimos e máximos por agregado é **exatamente** o limite do mapa bruto |
| `centroid_bounds`, `max_bounds_shrinkage_m` | o envelope dos centróides, que é o que um consumidor do representante vê, e a maior distância entre uma face dele e a mesma face do limite bruto |

Invariante verificável: um centróide fica dentro do seu voxel, então `distance_m.maximum ≤ √3 · cell_m` (teste).

As métricas de overlap de [`geometric_mapping.md`](geometric_mapping.md) **não se aplicam** ao agregado: elas comparam scans adjacentes entre si, e um agregado inter-scan já não pertence a um scan. Continuam valendo para o mapa bruto, que não muda.

## Estabilidade da associação 2D→3D (`compare_paired_associations`)

Sensor Association roda **sem mudança** duas vezes, com os mesmos frames, regiões, máscaras, calibração, trajetória e políticas: uma sobre o mapa bruto e outra sobre os centróides (o `AggregatedGeometry` implementa `GeometryBlockSource`). Regiões são pareadas por `(source_observation_id, perception_result_id, region_id)`. Uma região que só aparece em um dos runs é contada (`raw_only_region_count`, `aggregate_only_region_count`) e não entra na comparação.

Para cada região pareada, `R` é o suporte bruto (pontos), `A` é o suporte agregado (agregados) e `m(R)` é o conjunto de agregados que contêm os pontos de `R`, pela lineage (`membership`). A comparação é feita **entre conjuntos de voxels**, nunca entre linhas de array.

| Métrica | Definição |
|---|---|
| retenção de região | uma região com `R ≠ ∅` é retida se `A ≠ ∅`. `region_retention_rate = retidas / regiões com suporte bruto` |
| perdida / ganha | `R ≠ ∅, A = ∅` / `R = ∅, A ≠ ∅` |
| `support_retention` | `|m(R) ∩ A| / |m(R)|`: fração dos voxels que sustentavam a região e continuam sustentando |
| `disagreement` | `1 − |m(R) ∩ A| / |m(R) ∪ A|` (um menos o índice de Jaccard) |
| pontos de suporte | distribuições de `|R|` e `|A|` por região |
| regiões finas | regiões com `0 < |R| ≤ thin_support_max_points`, e quantas delas são retidas |

### Efeito downstream adicional: a geometria do suporte

Entidades e relações são construídas a partir do suporte de cada região. Para medir o efeito sobre elas de forma pareada e objetiva, sem rodar a cadeia inteira, cada região pareada registra:

| Métrica | Definição |
|---|---|
| `support_centroid_shift_m` | distância entre a média dos pontos de `R` e a média dos centróides de `A`, que é o que um consumidor calcularia a partir do suporte |
| `support_extent_change_m` | maior mudança, por eixo, da extensão da caixa do suporte |
| `support_bounds_iou` | IoU de volume das duas caixas; `None` quando uma caixa tem volume zero, como um suporte de um único ponto ou um plano perfeito |

A média dos centróides não é ponderada pelas contagens, porque um consumidor do suporte não vê as contagens. Então, quando voxels com muitas revisitas se misturam a voxels com poucas, o deslocamento do centróide do suporte é um efeito real da agregação, não um erro de medição (teste).

Materializar entidades e avaliar predicados de contato sobre o agregado **não** é medido nesta etapa: exigiria rodar Semantic Fusion → Semantic Mapping → Entity Resolution → Spatial Relations sobre um mapa derivado cujos elementos não são `GeometryPoint`. Esses consumidores resolvem o suporte por `GeometrySource.get`, que um agregado deliberadamente não implementa. A geometria do suporte acima é a medida pareada disponível. Estender a comparação a entidades e relações é o follow-up.

## Parâmetros fixados antes dos resultados

No experimento [`experiments/geometric-aggregation-raw-vs-voxel-20260926/`](../../../../experiments/geometric-aggregation-raw-vs-voxel-20260926/README.md):

| Parâmetro | Valor | Onde |
|---|---|---|
| faixas de alcance | `0, 1, 2, 5, 10, 20` m, a última aberta | `scripts/arm.py` |
| região fina | `|R| ≤ 20` pontos | `scripts/compare.py` |
| consultas por caixa | 16 caixas de 2 m centradas em pontos igualmente espaçados da diagonal do limite bruto | `scripts/arm.py` |
| verificação de chunking | blocos de `1_000_000` e de `4_099` pontos, `equivalence_problems` vazio | `scripts/arm.py` |

## Testes

`tests/evaluation/test_voxel_aggregation_evaluation.py` cobre, sobre o corredor simulado em que cada scan vê o mundo inteiro: revisitas colapsam com erro desprezível; o limite `√3 · cell`; faixas de alcance contra o alcance recomputado das coordenadas de origem; o envelope das estatísticas igual ao limite bruto; e retenção, desacordo, regiões perdidas, ganhas, finas e não pareadas, e geometria do suporte. `tests/end_to_end/test_voxel_aggregation_pairing.py` roda o `SensorAssociationService` real sobre os agregados da cadeia sintética de CI e confere que nenhuma região é perdida.
