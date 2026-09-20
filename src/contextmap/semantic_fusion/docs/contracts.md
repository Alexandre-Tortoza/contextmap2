# Contratos de Semantic Fusion

Referência dos contratos públicos de `contextmap.semantic_fusion`. Todos são `dataclass(frozen=True)`, validam suas invariantes na construção e referenciam a evidência de origem por identidade: nada aqui copia XYZ, embeddings, claims nem componentes de qualidade.

## Ideias que moldam os contratos

1. **Suporte não é identidade.** `FusionSupport` só diz *onde* a evidência é acumulada.
2. **Frame físico não é inferência.** Várias execuções sobre o mesmo frame são inferência **correlacionada**, não vistas independentes.
3. **Sinais tipados e sem mistura.** Confiança da claim, suporte de scorer e qualidade de observação são três quantidades diferentes. Nenhuma é convertida em outra nem combinada em um escalar.
4. **Ausência não é zero.** Um score `None` é "não pontuado", diferente de `0.0`.
5. **Estrutura estática pertence ao suporte.** Uma `PointRepresentation` é listada uma vez, nunca por vista.

## `FusionSupport`

Onde a evidência é acumulada, antes de existir identidade persistente.

| Campo | Significado |
| --- | --- |
| `fusion_support_id` | Identidade, local à execução de fusão. |
| `geometric_map_id` | Mapa imutável ao qual a geometria pertence. |
| `geometry_support` | `GeometryReference` ordenadas por `geometry_id`, únicas, todas do mesmo mapa. |
| `spatial_observation_ids` | `SpatialObservation` acumuladas, ordenadas e únicas. |
| `bounds` | `Bounds3D` que declara o frame em que está expressa. |
| `centroid_m` | Centroide em metros, no frame de `bounds`; finito e dentro dos limites (faces incluídas). |
| `time_bounds` | `TimeBounds` da aquisição das observações. |
| `provenance` | `FusionSupportProvenance`: `support_policy_id` versionada, `configuration_fingerprint`, `code_version`. |

Não existe campo de label, classe, confiança nem identidade de objeto. O suporte não é estável entre versões do mapa.

## `EvidenceContribution`

Uma vista: uma região de um frame físico, como interpretada por uma execução.

| Campo | Significado |
| --- | --- |
| `contribution_id` | Identidade, local à execução de fusão. |
| `physical_observation_id` | O `SourceObservationId` do frame físico; **a chave de correlação**. |
| `perception_result_id`, `perception_run_id` | A inferência que produziu as claims. |
| `spatial_observation_id`, `region_id` | A observação espacial e a região de origem. |
| `geometry_support` | Geometria que a região enxerga, ordenada, única, de um só mapa, nunca vazia. |
| `claim_refs` | `SemanticClaimRef` ordenadas por claim; vazio quando a inferência não produziu claim. |
| `score_refs` | `ScoreReference` (claim + scorer) ordenadas; cada uma nomeia uma claim da própria contribuição. |
| `visual_feature_refs` | `VisualFeatureRef` ordenadas; features de região pertencem à região da contribuição. |
| `observation_quality` | `ObservationQualityRef` ou `None`: a qualidade mensurável, só por referência. |

Uma claim aparece **uma vez** por contribuição, qualquer que seja o número de elementos de geometria que a região enxerga: um `Region2D` associado a mil pontos é uma evidência semântica com suporte espacial, não mil votos.

Não há campo de `PointRepresentation` aqui: estrutura estática não é evidência por vista.

## `PhysicalObservationGroup`

Separa "quantos frames" de "quantas inferências".

| Campo | Significado |
| --- | --- |
| `physical_observation_id` | O frame físico. |
| `acquisition_timestamp` | `SourceTimestamp` da aquisição. |
| `spatial_observation_ids` | Observações espaciais deste frame, ordenadas e únicas. |
| `perception_result_ids`, `perception_run_ids` | Os resultados e execuções correlacionados sobre o frame. |

Três execuções sobre `frame-0120` formam **um** grupo com **três** resultados: `physical_observation_count == 1`, `inference_result_count == 3`. O grupo não depende de suporte: o agrupamento global é calculado por `group_by_physical_observation` (ver [`grouping.md`](grouping.md)) e o grupo de um `FusedEvidence` lista só as observações espaciais do seu suporte.

## `SupportSignal`

Um score tipado de uma claim.

| Campo | Significado |
| --- | --- |
| `kind` | `CLAIM_CONFIDENCE` (a confiança que o interpretador reportou) ou `SCORER_SUPPORT` (o suporte que um scorer atribuiu). |
| `producer` | `BackendProvenance` do modelo cuja semântica o valor tem. |
| `value` | Score em `[0, 1]` sob a semântica do produtor, ou `None` (não pontuado). |

Qualidade de observação **não** é um `SupportSignalKind`. Ela é medida a montante, por Sensor Association, em componentes separados, e só é referenciada. Nenhuma política aqui pode chamá-la de confiança semântica.

## `FusedHypothesis` e `HypothesisEvidence`

Um candidato semântico e a evidência exata por trás dele.

- `label`: o texto proposto, verbatim (vocabulário aberto). Equivalência entre labels é decisão de política, nunca suposta.
- `evidence`: `HypothesisEvidence` ordenadas por contribuição e claim, cada uma com `stance`, `role` e `signals`.
- `EvidenceStance`: `SUPPORTING` (a claim propõe a hipótese), `CONFLICTING` (propõe uma hipótese incompatível para o mesmo suporte), `AMBIGUOUS` (mantida, com relação indecidida).
- `role`: se o interpretador propôs a claim como `PRIMARY` ou `ALTERNATIVE`.
- Toda hipótese tem ao menos uma evidência `SUPPORTING`. Não há probabilidade nem confiança única.

## `UncertaintyRecord`

Conflito, ambiguidade ou falta de evidência, com a evidência exata que o produziu.

| `UncertaintyKind` | Significado | Exige |
| --- | --- | --- |
| `CONTRADICTION` | Observações físicas distintas propõem hipóteses incompatíveis. | ≥ 2 hipóteses e evidência de ≥ 2 observações físicas distintas. |
| `AMBIGUITY` | Hipóteses concorrentes que nenhuma contradição explica (por exemplo, alternativas de uma só interpretação). | ≥ 2 hipóteses. |
| `NEAR_TIE` | Hipóteses indistinguíveis sob a regra que declarou o empate. | ≥ 2 hipóteses. |
| `INSUFFICIENT_EVIDENCE` | Evidência semântica insuficiente para sustentar hipótese alguma. | — |

`evidence` é uma lista de `EvidenceReference` (`contribution_id` e, quando a evidência é uma claim, `claim_id`; `None` para uma vista sem claim). `rule_id` identifica a regra versionada que o declarou. `unknown` nunca é convertido em "não-X", e "score ausente" nunca em `0`.

## `FusedEvidence`

| Campo | Significado |
| --- | --- |
| `fused_evidence_id`, `fusion_support_id` | Identidade e o suporte acumulado. |
| `physical_observation_groups` | Um grupo por frame físico, ordenado. |
| `contributions` | Toda contribuição, ordenada por identidade. |
| `hypotheses` | Candidatos ordenados por identidade; vazio quando nenhuma vista produziu claim. |
| `point_representation_refs` | `PointRepresentationRef` do suporte, cada uma listada uma vez. |
| `uncertainty` | Conflitos, ambiguidades e falta de evidência. |
| `temporal_summary` | `TimeBounds` que cobre exatamente a aquisição dos grupos. |
| `provenance` | `FusedEvidenceProvenance`: `grouping_policy_id`, `fusion_policy_id`, `configuration_fingerprint`, `code_version`. |

Não há vencedor, hipótese primária nem confiança combinada. Validações de consistência entre as partes:

- toda observação espacial contribui no máximo uma vez;
- todo frame com contribuições tem exatamente um grupo, e o grupo lista exatamente as observações espaciais, resultados e execuções desse frame;
- um resultado de inferência não pode pertencer a duas observações físicas;
- todo frame compartilha um único domínio de relógio e `temporal_summary` é exatamente o intervalo dos grupos;
- toda evidência de hipótese e de incerteza referencia uma contribuição e uma claim que existem;
- um sinal `SCORER_SUPPORT` exige a `ScoreReference` correspondente na contribuição;
- rótulos de hipótese não se repetem.

`FusedEvidence.supporting_physical_observations(hypothesis_id)` devolve os frames distintos cujas claims sustentam a hipótese: inferência repetida sobre um frame e uma claim sobre muita geometria contam uma vez, então o resultado é seguro como contagem de evidência independente.

## Ordem canônica

Coleções cuja ordem não tem significado são exigidas ordenadas e únicas (por identidade), para que a mesma evidência produza sempre o mesmo registro. Quem produz a evidência ordena; o contrato rejeita o resto.

## O que ainda não existe

Serialização, o artifact de run e os serviços de construção de suporte e acumulação ainda não foram implementados. Nenhum contrato aqui exige compatibilidade com formatos históricos.
