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

| Campo | Significado |
| --- | --- |
| `geometry_refs` | `GeometryReference` ordenadas por `geometry_id`, únicas, de um só mapa e **nunca vazias**. |
| `map_frame` | Frame do mapa ao qual o suporte pertence. |

Suporte vazio não pode se passar por geometria válida: a construção é recusada.

## `EntitySemanticState` e `EntityHypothesis`

`EntitySemanticState.hypotheses` são todos os candidatos, ordenados por evidência fundida e hipótese; vazio quando nenhuma vista produziu uma claim.

| Campo de `EntityHypothesis` | Significado |
| --- | --- |
| `fused_evidence_id`, `hypothesis_id` | A evidência fundida de origem e a hipótese, local a ela. |
| `label` | O texto da hipótese, verbatim; equivalência entre labels é decisão de política, nunca suposta. |
| `evidence` | `HypothesisEvidence` de Semantic Fusion: claim, stance (suporta, conflita, ambígua, abstém), papel e sinais tipados; ordenadas e únicas. |

Uma hipótese exige ao menos uma evidência que a suporte. Um label não se repete dentro da mesma evidência fundida.

## `EntityEvidenceLinks` e `FusedEvidenceRef`

`FusedEvidenceRef(fusion_run_id, fused_evidence_id, fusion_support_id)` aponta para a evidência fundida dentro de um `SemanticFusionRunArtifact`. `EntityEvidenceLinks.fused_evidence` nunca é vazio: uma entidade sem evidência por trás não pode ser auditada.

## `EntityTemporalState`

| Campo | Significado |
| --- | --- |
| `first_seen`, `last_seen` | Aquisição da primeira e da última observação física; mesmo clock, `last_seen` não antecede `first_seen`. |
| `physical_observation_count` | Frames físicos distintos, no mínimo 1. |
| `inference_result_count` | Resultados de inferência sobre esses frames, correlacionados dentro de cada frame; nunca menor que os frames. |

## Serialização

`serialization.encode_entity` / `decode_entity` produzem e leem registros JSON com primitivas apenas. A decodificação reconstrói os contratos pelos construtores, então todas as invariantes são revalidadas e um registro adulterado é recusado. A geometria é guardada como deltas posicionais (`map--geom-N` é posicional), o que mantém compacta uma entidade de milhares de pontos.
