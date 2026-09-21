# Estado temporal da entidade

Este documento descreve `src/contextmap/semantic_mapping/temporal.py`: o estado e o histórico temporal mínimos de uma entidade.

Mesmo num mapa majoritariamente estático, uma entidade precisa de proveniência temporal para que um consumidor saiba **quando** e **quantas vezes** ela foi observada. Rastrear objetos móveis é outro problema e **não** está aqui: não há velocidade, trajetória, re-identificação após movimento nem predição, e o ciclo de vida nunca infere movimento, desaparecimento ou destruição.

## `EntityTemporalState`

| Campo | Significado |
| --- | --- |
| `first_seen`, `last_seen` | Aquisição do primeiro e do último frame físico que contribuiu; mesmo clock, `last_seen` não antecede `first_seen`. |
| `physical_observation_count` | Frames físicos distintos, no mínimo 1. |
| `inference_result_count` | Resultados de inferência sobre esses frames, correlacionados dentro de cada frame; nunca menor que os frames. |
| `observation_refs` | O **índice de histórico**: uma `ObservationRef` por frame físico, em ordem cronológica e sem repetição. |
| `provenance` | `TemporalProvenance`: a regra versionada e se a seleção de evidência chegou em ordem cronológica. |
| `lifecycle` | `EntityLifecycle` ou `None`. |

A propriedade `time_bounds` devolve o `TimeBounds` de `first_seen` a `last_seen`.

### Contagens distintas

Frames físicos e resultados de inferência são contados à parte, como em Semantic Fusion: várias execuções sobre um mesmo frame são inferência **correlacionada**, nunca vistas independentes. Um frame interpretado por três execuções conta **1** frame físico e **3** resultados de inferência.

### Histórico auditável

`ObservationRef` guarda o identificador do frame, o instante de aquisição e o número de resultados de inferência sobre ele: uma entrada pequena por frame, sem duplicar o payload de evidência. O contrato exige que `first_seen`, `last_seen` e as duas contagens **concordem com o histórico**, então são reproduzíveis a partir da evidência exata que contribuiu. `Entity` também exige que o histórico e os vínculos de evidência (`EntityEvidenceLinks.physical_observation_ids`) listem os mesmos frames físicos.


## Ciclo de vida

Conservador e definido explicitamente:

| Valor | Significado |
| --- | --- |
| `observed` | Pelo menos um frame físico da evidência selecionada contribuiu. Nada afirma sobre persistência ou presença atual. |
| `stale` | A última observação é mais antiga que um horizonte declarado por uma política, relativo a um instante de referência que ela declarou. |
| `uncertain` | Uma política julgou as observações inconsistentes demais para afirmar que a entidade existe. |

A baseline atribui **apenas** `observed`; `stale` e `uncertain` são valores de contrato que uma política posterior e documentada pode atribuir. Nada é inferido aqui.

## `summarize_temporal_state`

Versionada: `physical-observation-temporal-summary-v1`. Deriva o estado dos `PhysicalObservationGroup` da evidência fundida.

- `first_seen`/`last_seen` são o primeiro e o último frame; os frames são ordenados por instante e, em empate, por identidade.
- Uma seleção que **não** chegou em ordem cronológica é **registrada** (`input_order_chronological = False`), não reordenada em silêncio.
- Evidência temporal ausente ou inválida é um erro explícito (`TemporalEvidenceError`), nunca um timestamp fabricado: nenhuma observação física, uma observação repetida ou timestamps de mais de um domínio de clock.

## Validação

`tests/semantic_mapping/test_entity_temporal_state.py` cobre uma observação física com várias execuções de inferência, várias observações ao longo do tempo, seleção não cronológica, empate de instante, observação repetida, clocks misturados, ausência de evidência, concordância com `FusedEvidence.temporal_summary` e as contagens da fusão (usando um run de fusão real), as invariantes do contrato e a persistência.
