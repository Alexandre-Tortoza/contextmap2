# Comparação geométrica

Um canal de evidência do par de entidades: compara o **suporte 3D persistente** das duas entidades no frame comum do mapa e devolve `GeometryEvidence` (`compare_geometry`). A geometria é **um** canal, nunca a única autoridade: um score geométrico alto não interpreta as entidades como o mesmo objeto, e nenhuma métrica desta função usa semântica, aparência ou representação 3D.

## Sinais medidos (`GeometryMeasurement`)

Cada sinal tem unidade e semântica próprias e **nenhum é somado a outro**. O par vem em ordem canônica (`a` é a entidade de menor referência), então `compare_geometry(x, y) == compare_geometry(y, x)`.

| Sinal | Definição |
| --- | --- |
| `centroid_distance_m` | Distância euclidiana entre os centroides (m). |
| `bounds_gap_m` | Distância euclidiana entre as caixas alinhadas aos eixos (m); `0` se sobrepõem ou se tocam. |
| `bounds_iou` | Volume da interseção sobre o volume da união das caixas. **Ausente (`None`)** se uma caixa é plana, porque uma caixa plana não tem sobreposição volumétrica: ausência não é zero. |
| `bounds_overlap_fraction_a/b` | Parte do volume da caixa de `a` (ou `b`) que está dentro da outra; `None` se a caixa é plana. |
| `support_count_a/b`, `shared_support_count`, `support_jaccard` | Elementos de geometria de cada suporte e os compartilhados: **os mesmos pontos autoritativos**, não apenas pontos próximos. Jaccard = compartilhados / união. |
| `extent_ratio` | Menor razão, entre os três eixos, entre o menor e o maior lado das caixas, em `(0, 1]`; `1` num eixo em que ambas são planas; `None` se uma é plana num eixo em que a outra não é. |
| `support_distance` | Estatísticas de vizinho mais próximo entre os pontos dos dois suportes (média de `a` para `b`, de `b` para `a`, Hausdorff). Opcional: exige o `GeometrySource` do mapa e a política ligá-lo. |
| `orientation_angle_rad` | Ângulo entre os eixos principais, em `[0, π/2]` (o eixo não tem sentido). `None` se uma entidade não tem orientação bem definida. |
| `caveats` | Diagnósticos dos resumos das entidades que qualificam as medidas, por lado: `a:sparse_support`, `b:disconnected_support`, `a:degenerate_extent`, ... |

`geometric_map_id` e `map_frame` dizem em que mapa e frame toda medida foi feita.

## Frame comum

Os gates duros (`evaluate_comparison_gates`) rodam antes de qualquer medida: uma entidade nunca é comparada com ela mesma, e coordenadas de **mapas geométricos** ou **frames** diferentes nunca são comparadas sem um alinhamento explícito. Um gate reprovado torna o canal `unavailable` com motivo `blocked_by_gate`, e o detalhe nomeia os valores. Não há transformação implícita.

## Regras `entity-geometry-comparison-v1`

`GeometryComparisonPolicy` **não tem valores padrão**: os limiares dependem da escala e da densidade da cena e são escolha que um perfil declara e justifica com dados de validação (issue #140). Os limiares sobre razões (IoU, contenção, Jaccard, razão de extensão) não dependem de escala; os dois em metros (`min_conflict_gap_m`, `max_mean_distance_m`) são declarados pelo perfil, nunca fixados a um dataset.

Cada regra vira um `Finding` com a métrica, o valor observado e o limiar, e o status do canal é derivado deles (conflito domina, depois apoio, senão neutro).

| Regra | Métrica | `supporting` | `conflicting` | Neutro |
| --- | --- | --- | --- | --- |
| `shared-support` | `support_jaccard` | ≥ `min_shared_support_jaccard` | nunca | senão (não compartilhar nada **não** é evidência contra) |
| `bounds-iou` | `bounds_iou` | ≥ `min_bounds_iou` | nunca | senão, ou caixa plana (não avaliável) |
| `bounds-containment` | maior `bounds_overlap_fraction` | contida (≥ `min_bounds_containment`) **e** de tamanho comparável (`extent_ratio` ≥ `min_extent_ratio`) | nunca | senão |
| `bounds-separation` | `bounds_gap_m` | nunca | ≥ `min_conflict_gap_m` | senão |
| `extent-mismatch` | `extent_ratio` | nunca | < `min_extent_ratio` **e** uma caixa não contém a outra | senão (inclusive quando a contenção explica) |
| `support-proximity` (se a política liga) | maior média de vizinho mais próximo | ≤ `max_mean_distance_m` | nunca | senão |
| `orientation-mismatch` (se a política dá um limite) | `orientation_angle_rad` | nunca | > limite | senão, ou sem orientação |

Conflito não é prova de identidade distinta: é evidência que a política de resolução pesa junto com os outros canais.

### Casos que a comparação trata de forma explícita

- **Suporte esparso ou desconexo:** entram como `caveats` da medida, nunca lidos como suporte confiável.
- **Entidade grande e plana (piso, parede):** um objeto pequeno **totalmente contido** nela não é evidência de identidade. A contenção só apoia quando os tamanhos são comparáveis, e uma diferença de tamanho que a contenção explica (visão parcial, fundo) é neutra, não conflito.
- **Visão parcial:** uma parte contida de tamanho comparável (a metade de um objeto, por exemplo) recebe apoio por contenção e por suporte compartilhado, e a diferença de extensão fica neutra.
- **Densidade desigual:** as contagens de cada suporte ficam visíveis; nenhuma regra usa densidade.
- **Caixa degenerada:** sem IoU nem fração (ausentes, não zero) e com o caveat `degenerate_extent`; os demais sinais continuam valendo.
- **Vizinhos disjuntos:** dois objetos próximos mas sem sobreposição têm IoU `0` e, a partir de `min_conflict_gap_m`, um achado de separação conflitante; abaixo dele a separação é neutra.
- **Referência inválida:** com `support_distance` ligado, uma referência que não resolve no `GeometrySource` levanta `GeometryResolutionError`; o canal não pula silenciosamente. Pedir estatísticas sem `GeometrySource` levanta `ValueError`.

## Estatísticas de vizinho mais próximo

Opcionais (`SupportDistancePolicy(max_points_per_side, max_mean_distance_m)`). Suportes maiores que `max_points_per_side` são amostrados em posições igualmente espaçadas das referências ordenadas, de forma determinística, e o tamanho da amostra fica em `SupportDistance.points_a/points_b` (compare com `support_count_a/b` para saber se houve amostragem). O NumPy é importado só nesse caminho: ler e gravar resoluções não precisa dele. O cálculo é força bruta em blocos (memória limitada).

No `MatchEvidenceBuilder`, o canal usa um `GeometryComparator` que guarda a amostra resolvida de cada entidade pela vida do builder (um por run), como os canais de aparência e de representação: uma entidade em k pares candidatos é lida do `GeometrySource` uma vez, não k (#599). `compare_geometry` continua sem estado e resolve as duas amostras a cada chamada.

Custo medido por par (sintético, uma máquina de desenvolvimento): só as métricas de caixa e suporte, 0,15 a 0,67 ms; com as estatísticas, 58 ms com 500 pontos por lado, 207 ms com 2 000 e 1,2 s com 5 000. É quadrático na amostra, por isso a amostra é limitada pela política. Para um mapa com milhares de pares candidatos, mantenha `max_points_per_side` na casa de centenas ou desligue o canal e use só as métricas de caixa.

## O que este módulo não faz

Nenhuma decisão `MATCH`/`DISTINCT`, nenhuma fusão de entidades, nenhuma métrica semântica ou visual e nenhuma comparação entre frames sem alinhamento explícito.
