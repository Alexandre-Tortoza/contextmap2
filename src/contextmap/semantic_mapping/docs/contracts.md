# Contratos de Semantic Mapping

Referência dos contratos públicos de `contextmap.semantic_mapping`. Todos são `dataclass(frozen=True)`, validam suas invariantes na construção e referenciam tudo o que vem de upstream por identidade: nada aqui copia XYZ, embeddings, claims nem payloads de fusão.

## Ideias que moldam os contratos

1. **Entidade não é label.** `Entity` nunca se reduz a `{id, label, confidence, xyz}`: guarda suporte 3D, hipóteses, evidência e tempo.
2. **Identidade tem escopo.** Um `EntityId` só significa algo dentro do semantic map que o possui.
3. **Nada é colapsado.** Todas as hipóteses ficam, cada uma com os sinais tipados de cada claim; um score ausente (`None`) é diferente de um score baixo.
4. **Sem decisão de mesmo objeto.** Construir ou ler uma entidade não exige nenhum algoritmo de Entity Resolution.

## `Entity`

| Campo | Significado |
| --- | --- |
| `entity_id` | Identidade, local a `semantic_map_id`. |
| `semantic_map_id` | O semantic map dono; torna o escopo explícito no próprio registro. |
| `geometry` | `EntityGeometry`: o suporte 3D persistente. |
| `semantic_state` | `EntitySemanticState`: o que a entidade pode ser. |
| `evidence` | `EntityEvidenceLinks`: de qual evidência fundida ela veio. |
| `temporal_state` | `EntityTemporalState`: quando e quantas vezes foi observada. |
| `provenance` | `EntityProvenance`: política de materialização e de identidade, fingerprint da configuração, versão do código. |

Invariante entre componentes: toda hipótese vem de uma evidência fundida à qual a entidade está vinculada.

A propriedade `Entity.reference` devolve o `EntityReference` da entidade.

## `EntityReference`

`(semantic_map_id, entity_id)`. É o handle estável de uma entidade: a igualdade considera os dois campos, então `entity-0001` de dois mapas são referências **diferentes**.

## `EntitySet`

As entidades de **um** semantic map, ordenadas por `entity_id` e únicas, todas pertencentes ao mesmo `semantic_map_id`. `EntitySet.of` aceita qualquer ordem.

`resolve(reference)`:

- devolve a entidade quando a referência é do mapa e o id existe;
- levanta `ForeignEntityReferenceError` quando a referência nomeia **outro** mapa, mesmo que o `entity_id` exista aqui;
- levanta `UnknownEntityError` quando o mapa não tem a entidade.

## `EntityGeometry`

O suporte 3D persistente e seus resumos derivados; detalhes em [`geometry.md`](geometry.md).

| Campo | Significado |
| --- | --- |
| `geometry_refs` | `GeometryReference` ordenadas por `geometry_id`, únicas, de um só mapa e **nunca vazias**: é a autoridade. |
| `map_frame` | Frame do mapa; todo resumo está expresso nele. |
| `centroid_m`, `bounds`, `extent_m` | Centroide, `Bounds3D` justo e lados da caixa, derivados do suporte. |
| `statistics` | `SupportStatistics`: pontos, volume, densidade (`None` se a caixa é plana), componentes conexos. |
| `summary` | `SpatialSummaryProvenance`: algoritmo, frame, conjunto de entrada (contagem e digest), convenções numéricas, filtragem e fingerprint da política. |
| `orientation` | `EntityOrientation` (eixos ortonormais destros e variâncias) ou `None`. |
| `diagnostics` | `GeometryDiagnostic` (`sparse_support`, `disconnected_support`, `degenerate_extent`, `orientation_not_justified`). |

Suporte vazio não pode se passar por geometria válida: a construção é recusada (`EmptyGeometrySupportError`). O contrato também recusa resumos em outro frame, centroide fora dos limites, extensão que não é o tamanho da caixa, estatísticas ou digest que não correspondem às referências e diagnósticos que contradizem as estatísticas.

## `EntitySemanticState` e seus tipos

O estado semântico, sem colapso; detalhes em [`semantic-state.md`](semantic-state.md).

| Campo | Significado |
| --- | --- |
| `hypotheses` | `EntityHypothesis` ordenadas por evidência fundida e hipótese; vazio quando nenhuma vista produziu uma claim. |
| `ambiguity_state` | `AmbiguityState` (`unambiguous`, `ambiguous`, `conflicting`, `insufficient_evidence`), sempre o que `derive_ambiguity_state` calcula: o contrato recusa outro valor. |
| `provenance` | `SemanticStateProvenance`: regra de mapeamento e política do primário. |
| `primary_hypothesis` | `EntityHypothesisRef` ou `None`; só existe quando o estado é `unambiguous`. |
| `attributes` | `EntityAttribute` (`name`, `value`, `origin`, `derivation_id`, `evidence`, `support`). |
| `uncertainty` | `EntityUncertainty`: cada registro de incerteza da fusão com a evidência fundida que o reportou. |

`EntityHypothesis` guarda `fused_evidence_id`, `hypothesis_id`, o `label` verbatim (equivalência entre labels é decisão de política, nunca suposta) e a `evidence` (`HypothesisEvidence` de Semantic Fusion: claim, stance, papel e sinais tipados). Uma hipótese exige ao menos uma evidência que a suporte, e um label não se repete dentro da mesma evidência fundida.

Um atributo observado ou derivado sem evidência é recusado; `EXTERNAL_KNOWLEDGE` é rotulado e exige uma derivação documentada. Uma primária não pode ser exposta enquanto o estado não é `unambiguous`.

## `EntityEvidenceLinks` e seus tipos

Onde está a evidência que suporta a entidade; detalhes em [`evidence.md`](evidence.md).

| Campo | Significado |
| --- | --- |
| `fused_evidence` | `FusedEvidenceRef` (run, versão do schema, digest do artifact, sequência canônica, evidência e suporte); **nunca vazio**. |
| `spatial_observation_ids` | `SpatialObservationId` que contribuíram, ordenados e únicos. |
| `physical_observation_ids` | `SourceObservationId` dos frames físicos que contribuíram, ordenados e únicos. |
| `visual_feature_refs` | `EntityFeatureRef` (`perception_run_id`, `perception_result_id`, `feature_id`, `embedding_space_id`, `scope`, `region_id`), identificada pela tripla run, resultado e feature e ordenada nessa ordem; vazio se o canal está ausente. |
| `point_representation_refs` | `PointRepresentationRef` das representações 3D do suporte; vazio se o canal está ausente. |

Nada aqui copia imagens, máscaras, embeddings nem payloads de fusão. Uma representação 3D ancorada fora do suporte geométrico da entidade é recusada por `Entity`.

## `EntityTemporalState`

Quando e quantas vezes a entidade foi observada; detalhes em [`temporal-state.md`](temporal-state.md).

| Campo | Significado |
| --- | --- |
| `first_seen`, `last_seen` | Aquisição do primeiro e do último frame físico que contribuiu; mesmo clock, `last_seen` não antecede `first_seen`. |
| `physical_observation_count` | Frames físicos distintos, no mínimo 1. |
| `inference_result_count` | Resultados de inferência sobre esses frames, correlacionados dentro de cada frame; nunca menor que os frames. |
| `observation_refs` | `ObservationRef` (frame, instante de aquisição, resultados de inferência), em ordem cronológica e sem repetição: o índice de histórico. |
| `provenance` | `TemporalProvenance`: regra versionada e se a seleção chegou em ordem cronológica. |
| `lifecycle` | `EntityLifecycle` (`observed`, `stale`, `uncertain`) ou `None`. |

O intervalo e as contagens devem concordar com o histórico, e `Entity` exige que o histórico e os vínculos de evidência listem os mesmos frames físicos. Não há velocidade, trajetória nem rastreamento.

## Serialização

`encode_entity` / `decode_entity` (e `encode_entity_reference` / `decode_entity_reference`), exportados pela API pública, produzem e leem registros JSON com primitivas apenas. A decodificação reconstrói os contratos pelos construtores, então todas as invariantes são revalidadas e um registro adulterado é recusado. A geometria é guardada como deltas posicionais (`map--geom-N` é posicional), o que mantém compacta uma entidade de milhares de pontos.
