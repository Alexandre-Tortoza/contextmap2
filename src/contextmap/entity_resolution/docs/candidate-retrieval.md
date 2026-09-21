# Recuperação de candidatos

Comparar toda entidade com toda outra não escala, e a evidência de comparação (geometria, aparência, ...) é cara. A recuperação de candidatos é o **primeiro passo**: para cada entidade, devolve de forma permissiva e determinística as entidades **plausivelmente** iguais, usando só filtros espaciais e temporais interpretáveis. Ela **não decide nada**: um candidato devolvido não implica `MATCH`, e uma entidade ausente foi excluída por uma regra explícita da política, que `explain_candidacy` sabe dizer qual.

## Política `entity-candidate-retrieval-v1`

`CandidateRetrievalPolicy` não tem valores padrão para os limiares espaciais: eles dependem da escala da cena e são escolha científica que um perfil declara.

| Parâmetro | Significado |
| --- | --- |
| `centroid_radius_m` | Distância entre centroides abaixo da qual duas entidades são plausíveis (m). |
| `bounds_margin_m` | Distância entre as caixas (`Bounds3D`) abaixo da qual duas entidades são plausíveis (m); `0` mantém só caixas que se sobrepõem ou se tocam. |
| `max_time_gap_ns` | Maior lacuna entre os intervalos de observação que mantém as entidades candidatas (ns); `None` significa que o tempo nunca exclui. |

`fingerprint()` faz o hash da identidade da política e dos três limiares; ele vai em `EntityCandidateSet.policy` (`PolicyRef`).

Um par é candidato quando **qualquer** uma destas vale:

| Motivo (`RetrievalReason`) | Regra |
| --- | --- |
| `centroid_within_radius` | distância entre centroides ≤ `centroid_radius_m` |
| `bounds_overlap` | as caixas se sobrepõem ou se tocam (lacuna = 0) |
| `bounds_within_margin` | as caixas não se tocam, mas a lacuna ≤ `bounds_margin_m` |

As regras de caixa existem porque uma entidade grande e plana (parede, piso) e um objeto encostado nela têm centroides distantes. Os motivos que valem ficam no candidato, em ordem canônica e nunca vazios.

### O que nunca exclui

- **Semântica.** Labels diferentes, iguais ou ausentes não mudam a recuperação: a semântica upstream pode estar errada, então discordância semântica não é exclusão universal.
- **Evidência opcional ausente** (aparência, representação 3D).
- **Tempo entre relógios diferentes.** As lacunas só são comparáveis dentro de um domínio de relógio; entre domínios diferentes `time_gap_ns` é `None` e o par não é excluído.
- O tempo só exclui quando a política declara `max_time_gap_ns`, e nunca infere movimento: a premissa é cena estática ou quase estática, sem re-identificação após grandes deslocamentos.

### O que sempre exclui

Entidades sobre **outro mapa geométrico ou outro frame** nunca são candidatas (`incompatible_map`): coordenadas de mapas ou frames diferentes não se comparam sem um alinhamento explícito. E uma entidade nunca é candidata de si mesma (`same_entity`).

## Diagnóstico de exclusão

`explain_candidacy(entity, other, policy)` é a **única definição** de candidatura: devolve um `CandidacyAssessment` com a exclusão (`ExclusionReason`: `same_entity`, `incompatible_map`, `outside_spatial_range`, `temporal_gap_exceeded`) ou `None`, os motivos espaciais, as distâncias medidas e um texto que nomeia os valores comparados. É simétrica. O índice só reduz quais pares vale perguntar; todo candidato é confirmado por esta função, então a grade pode tornar a recuperação mais rápida, nunca diferente. Um teste compara o índice com a força bruta sobre esta função.

## Índice espacial

Semantic Mapping persiste só `entity-geometry-index.jsonl` (uma caixa e um centroide por entidade), não uma estrutura espacial, e o artifact não a reconstrói. `EntitySpatialIndex` constrói em memória, a cada execução, uma grade uniforme sobre as caixas das entidades (tamanho de célula `max(centroid_radius_m, bounds_margin_m)`), **sem persistir nada novo**.

- Entidades de mapas geométricos ou frames diferentes ficam em grupos separados e nunca se veem.
- A região de consulta de uma entidade é a união do cubo do raio em torno do centroide (um candidato por centroide tem o centroide, logo a caixa, ali) e da própria caixa crescida pela margem. Uma entidade é examinada quando sua caixa toca uma célula dessa região.
- Uma entidade que cobriria mais de 512 células (piso ou parede muito maiores que o raio) fica numa lista curta examinada em toda consulta, e uma consulta que cobre mais células que as ocupadas varre as ocupadas. As duas são decisões de custo que nunca mudam o resultado, e os testes forçam os dois caminhos.
- A região de consulta tem folga de `1e-6` m para o arredondamento nas bordas das células, de modo que um par exatamente no limiar não escapa da grade.

`retrieve_candidate_sets(entities, policy)` devolve um `EntityCandidateSet` por entidade, em ordem canônica de referência; o resultado independe da ordem de `entities`. `candidate_pairs` lista cada par uma vez (a recuperação é simétrica, então o par aparece nos conjuntos das duas entidades), com a entidade de menor referência primeiro.

## Contratos

| Tipo | Campos |
| --- | --- |
| `EntityCandidateSet` | `source_entity_ref`, `candidates`, `policy` (`PolicyRef`), `diagnostics` |
| `EntityCandidate` | `entity_ref`, `reasons`, `centroid_distance_m`, `bounds_gap_m`, `time_gap_ns` |
| `RetrievalDiagnostics` | `compatible_entity_count` (o que o all-pairs examinaria), `examined_entity_count` (onde a regra exata rodou) |

Invariantes: candidatos ordenados por referência e únicos, nunca incluindo a própria origem, cada um com pelo menos um motivo em ordem canônica, e nunca mais candidatos que entidades examinadas. `encode_candidate_set` / `decode_candidate_set` seguem o codec estrito do módulo.

## Linha de base de desempenho

Medição sintética, uma máquina de desenvolvimento, Python puro, entidades reais construídas por `summarize_geometry` (caixas de 0,3 a 1,2 m, densidade de 0,5 entidade por m² no plano, `centroid_radius_m = 1.0`, `bounds_margin_m = 0.2`). O tempo é só o da recuperação; a construção das entidades não entra.

| Entidades | Recuperação | Examinadas por entidade (média / máx.) | Candidatos por entidade (média / máx.) | Pares do all-pairs |
| ---: | ---: | ---: | ---: | ---: |
| 1 000 | 0,10 s | 6,5 / 15 | 1,44 / 6 | 499 500 |
| 5 000 | 0,67 s | 7,0 / 19 | 1,45 / 8 | 12 497 500 |
| 20 000 | 2,88 s | 7,0 / 19 | 1,45 / 10 | 199 990 000 |

O custo cresce de forma aproximadamente linear e o número examinado por entidade não depende de N (depende da densidade local). É uma cena sintética uniforme, sem suportes reais: cenas com aglomerados densos examinam mais por entidade, e a lista de entidades grandes é examinada em toda consulta.

## O que este módulo não faz

Nenhuma decisão `MATCH`/`DISTINCT`, nenhum índice aprendido, nenhum vizinho mais próximo aproximado, nenhuma re-identificação de objetos dinâmicos e nenhuma exclusão silenciosa por falta de evidência semântica ou visual.
