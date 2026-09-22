# Geometria da entidade

Este documento descreve `src/contextmap/semantic_mapping/geometry.py`: o suporte 3D de uma entidade e os resumos espaciais derivados dele.

## Autoridade

O suporte é o conjunto de `GeometryReference` da entidade, e a geometria persistida a que elas apontam é **a autoridade**. Centroide, limites, extensão, estatísticas e orientação são resumos **derivados** dela, expressos no mesmo frame do mapa. Nada copia XYZ para o registro da entidade, e um centroide nunca é a única representação espacial.

Recalcular um resumo **nunca muda a identidade** da entidade: o `EntityId` é alocado pela política de materialização, não por valores de geometria.

## `EntityGeometry`

| Campo | Significado |
| --- | --- |
| `geometry_refs` | Suporte autoritativo: ordenado por `geometry_id`, único, de um só mapa e **nunca vazio**. |
| `map_frame` | Frame do mapa; todo resumo está expresso nele. |
| `centroid_m` | Centroide, em metros; finito e dentro de `bounds`. |
| `bounds` | `Bounds3D` justo do suporte, no `map_frame`. |
| `extent_m` | Comprimento dos lados de `bounds`. |
| `statistics` | `SupportStatistics`: pontos, volume, densidade, componentes conexos. |
| `summary` | `SpatialSummaryProvenance`: como os resumos foram derivados. |
| `orientation` | `EntityOrientation` ou `None`: só quando a política pede **e** os eixos são bem definidos. |
| `diagnostics` | `GeometryDiagnostic` ordenados por tipo e únicos: ressalvas ditas em vez de escondidas. |

Suporte vazio não pode se passar por geometria válida: `EmptyGeometrySupportError` (uma `ValueError`) na construção e em `summarize_geometry`.

## Proveniência dos resumos

`SpatialSummaryProvenance` identifica o algoritmo (`entity-geometry-summary-v1`), o frame, o **conjunto de entrada** (contagem e digest SHA-256 dos `geometry_id`, ver `geometry_set_digest`), as convenções numéricas, se houve filtragem ou downsampling (a baseline declara explicitamente que **não**) e o fingerprint da política. O contrato recusa um resumo cujo digest não corresponde às referências da entidade.

## Algoritmo baseline

Configurado por `GeometrySummaryPolicy`, **sem defaults científicos**: os limiares (`sparse_point_threshold`, `connectivity_radius_m`, orientação) são escolhas que um perfil declara. A única exceção é o limite de custo da conectividade, `max_connectivity_points`, que tem o padrão nomeado `DEFAULT_MAX_CONNECTIVITY_POINTS` (10 000) porque é uma trava de custo, não um limiar científico.

- **Centroide:** média aritmética somada de forma exata (`math.fsum`). Ela está matematicamente dentro da caixa; um clamp só corrige o erro de arredondamento.
- **Limites e extensão:** a caixa alinhada aos eixos e seus lados.
- **Estatísticas:** número de pontos, volume dos limites e densidade (pontos por m³); a densidade é `None` quando a caixa é plana.
- **Conectividade:** pontos a uma distância de até `connectivity_radius_m` (inclusive) são ligados, e os componentes conexos são contados. Os pontos são agrupados numa grade de célula igual ao raio, então só as 27 células vizinhas são consultadas. É Python puro, sem NumPy, e o custo cresce com o número de pontos por vizinhança; por isso a política limita o tamanho do suporte que é ligado (ver o limite abaixo).
- **Orientação (opcional):** eixos principais da covariância do suporte (Jacobi para matrizes simétricas 3×3), ordenados por variância decrescente, com o sinal fixado para que a maior componente seja positiva e o terceiro eixo dado pelo produto vetorial dos dois primeiros (referencial destro).

### Política de custo da conectividade

A comparação de pares dentro de cada vizinhança de 27 células é **aproximadamente quadrática na densidade local de pontos**, e o custo **não foi validado em suportes reais**. Uma medição sintética (pontos aleatórios numa caixa de 2 × 2 × 1 m, raio de 0,3 m, uma única execução em Python puro) deu cerca de 0,2 s com 2 mil pontos, 4,5 s com 10 mil e 42 s com 30 mil. Um suporte denso e grande pode, portanto, dominar o tempo de materialização de uma entidade.

**A política é explícita e limitada.** `GeometrySummaryPolicy.max_connectivity_points` (padrão `DEFAULT_MAX_CONNECTIVITY_POINTS = 10_000`, que mantém uma entidade na ordem de segundos na medição acima) é o maior suporte cujos componentes são contados. Um suporte com **mais** pontos que o limite **não é ligado**: `statistics.component_count` e `statistics.largest_component_fraction` ficam `None` (nunca `1`: não contar não é o mesmo que estar conectado), o diagnóstico `connectivity_not_computed` diz quantos pontos havia e qual era o limite, e o restante do resumo (centroide, limites, extensão, densidade, orientação) não depende do limite. Um suporte no limite ainda é ligado. O limite entra no `fingerprint()` da política, então mudá-lo muda a identidade da configuração; um perfil que aceita o custo o aumenta de forma explícita. Não é um limiar científico e o valor padrão **não** foi calibrado em suportes reais.

Otimizações que preservam o resultado (por exemplo, pular pares que já estão no mesmo componente ou vetorizar com NumPy) **não foram feitas nem medidas**. Trocar por uma conectividade de ocupação de voxels mudaria o resultado e exigiria uma nova versão do algoritmo (`entity-geometry-summary-v2`); essa decisão fica para quando houver suportes reais para medir.

### Quando a orientação é justificada

`OrientationPolicy(min_points, min_variance_ratio)` só produz eixos quando o suporte tem pontos suficientes **e** as três variâncias são claramente distintas (cada uma ao menos `min_variance_ratio` vezes a seguinte). Um suporte isotrópico ou uma linha não tem eixos bem definidos: a orientação fica `None` e o diagnóstico `orientation_not_justified` diz por quê. Sem `OrientationPolicy`, a orientação nunca é calculada e não há diagnóstico.

## Diagnósticos

| Tipo | Quando |
| --- | --- |
| `sparse_support` | Menos pontos que `sparse_point_threshold`. |
| `disconnected_support` | Mais de um componente conexo; a fração do maior é registrada. |
| `connectivity_not_computed` | O suporte tem mais pontos que `max_connectivity_points`: os componentes não foram contados (`None`), e o suporte pode ou não ser conexo. |
| `degenerate_extent` | Os limites são planos em algum eixo: volume e densidade não são significativos. |
| `orientation_not_justified` | A orientação foi pedida mas os eixos não são bem definidos. |

O contrato exige que `disconnected_support`, `connectivity_not_computed` e `degenerate_extent` estejam presentes **exatamente** quando as estatísticas e a extensão os implicam, então um diagnóstico não pode ser removido nem inventado.

## Resolução e verificação

- `resolve_geometry(references, source=...)` resolve referências pela porta `GeometrySource` e levanta `GeometryResolutionError` quando uma referência é de outro mapa, não existe ou resolve para geometria em outro frame que o do mapa.
- `summarize_geometry(references, source=..., policy=...)` constrói o `EntityGeometry`.
- `verify_geometry_summary(geometry, source=..., policy=...)` recalcula os resumos a partir da geometria persistida e devolve os problemas encontrados (centroide, limites, contagens, orientação, diagnósticos, política). Um suporte que não resolve é reportado, não levantado.

Assim, os resumos de uma entidade são reproduzíveis e checáveis contra a geometria autoritativa, e Spatial Relations consome o suporte por referência, sem ler estruturas privadas de nenhum backend.
