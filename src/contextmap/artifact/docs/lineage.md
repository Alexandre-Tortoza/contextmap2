# Linhagem, referências de evidência e fechamento de proveniência

Este documento descreve `src/contextmap/artifact/provenance.py` e `_lineage_checks.py`. O mapa final precisa distinguir fatos e evidências **pela forma como foram produzidos**: observação de sensor, interpretação de modelo, derivação geométrica determinística, acúmulo multi-vista, anotação humana e, no futuro, raciocínio sobre conhecimento prévio não têm o mesmo status epistêmico e nunca são achatados em um único campo `confidence`.

```text
ContextMap
├── lineage[]          UpstreamArtifact  ── artifact_id, kind, content_identity,
│                                            configuration_fingerprint, code_version, model_identities[]
├── entities[]
│   ├── origin         EvidenceOrigin    ── kind, derived_from[] (UpstreamRecordRef), policy?
│   ├── member_entities[]      EntityReference: entidades de origem resolvidas em uma
│   ├── resolution_decisions[] ResolutionDecisionId: decisões que as agruparam
│   ├── unresolved_neighbors[] EntityReference: deixadas UNRESOLVED contra um membro
│   └── semantic_state.hypotheses[].origin
└── relations[].origin
```

## `DerivationKind` não é confiança

| Categoria | Significado | Só pode citar | Política |
| --- | --- | --- | --- |
| `SENSOR_OBSERVED` | ancorada diretamente em observação de sensor ou em geometria medida a partir dela | `SEQUENCE`, `GEOMETRIC_MAP` (e mais nada) | opcional |
| `MODEL_INFERRED` | produzida por um modelo de percepção, linguagem ou scorer | pelo menos um `PERCEPTION_RUN` que **nomeia o modelo** | opcional |
| `GEOMETRY_DERIVED` | derivada de forma determinística da geometria por uma regra versionada | pelo menos um `GEOMETRIC_MAP` ou `SPATIAL_RELATIONS_RUN` | **obrigatória** |
| `MULTIVIEW_FUSED` | acumulada a partir de várias contribuições de evidência | pelo menos um `SEMANTIC_FUSION_RUN` **ou** `ENTITY_RESOLUTION_RUN` | **obrigatória** |
| `HUMAN_ANNOTATED` | anotação humana explícita, fornecida como entrada do pipeline | pelo menos um `HUMAN_ANNOTATION_SET` | opcional |
| `PRIOR_KNOWLEDGE` | reservada para raciocínio sobre conhecimento prévio explícito | qualquer artifact da linhagem | **obrigatória** |

A categoria é **metadado de proveniência**, não uma pontuação: não ordena as categorias por confiabilidade e nenhum campo deste módulo é um número. Os valores de suporte que justificam uma hipótese continuam nos artifacts a montante, que a origem aponta com exatidão.

### O que a categoria nunca pode esconder

- **Saída de VLM não é `SENSOR_OBSERVED`** só por ter consumido uma imagem: uma origem `SENSOR_OBSERVED` que cita qualquer artifact fora de `SEQUENCE`/`GEOMETRIC_MAP` é rejeitada (`ProvenanceError`);
- **uma relação derivada geometricamente nunca é observação direta**: `ContextRelation.origin` não pode ser `SENSOR_OBSERVED`;
- **anotação usada só para avaliação não entra silenciosamente**: um conjunto de anotações só aparece em uma origem `HUMAN_ANNOTATED`; citá-lo sob qualquer outra categoria é rejeitado, e **não existe** `ArtifactKind` para conjunto de referência de avaliação nem para saída de debug, então nenhum dos dois pode ser citado;
- **a categoria sozinha não explica um resultado**: `derived_from` lista **toda** a evidência de origem, ordenada e única, com pelo menos um registro;
- `PRIOR_KNOWLEDGE` é representável (com política e evidência), sem mudar a semântica das demais origens, e nenhum estágio da v0.1.0 a emite: o schema apenas preserva a distinção.
- `ENTITY_RESOLUTION_RUN` foi acrescentado a `MULTIVIEW_FUSED` para a montagem real do `ContextMap` (`serialization/assembly.py`, [#537](https://github.com/Alexandre-Tortoza/contextmap2/issues/537)): a materialização de Entity Resolution também é uma regra versionada que acumula várias contribuições de evidência — a de cada membro fundido — e é o único artifact que a montagem (que nunca abre um run de Semantic Fusion) tem de fato aberto para citar essa derivação com honestidade. `SEMANTIC_FUSION_RUN` continua a citação correta para um resultado fundido a partir de uma leitura real desse run (a fixture representativa em `validation.md` continua usando-o).

## Artifacts a montante e o fechamento

`lineage` lista, ordenado por `artifact_id` e sem repetição, **todo artifact citado** pelo mapa, com as identidades exatas necessárias para auditá-lo:

| Campo | Significado |
| --- | --- |
| `artifact_id` | identidade exata do artifact ou run; nunca um caminho |
| `kind` | o que o artifact é (`ArtifactKind`), o que fixa o significado de seus registros |
| `content_identity` | `sha256:<hex>` do conteúdo: a referência nomeia exatamente **este** artifact e uma cópia obsoleta ou alterada é detectável |
| `configuration_fingerprint`, `code_version` | configuração efetiva e revisão de código que o produziram; `None` explícito quando desconhecidos |
| `model_identities` | modelos e checkpoints por trás das inferências, ordenados e únicos; vazio quando nenhum modelo rodou |

Todo registro citado pelo mapa resolve para essa tabela com o **tipo certo** (`ReferenceIntegrityError` caso contrário):

- `geometry_ref` → `GEOMETRIC_MAP`; cada `source_sequences` → `SEQUENCE`;
- `ContextEntity.source.resolution_run_id` → `ENTITY_RESOLUTION_RUN` (as `resolution_decisions` são locais a esse run e não precisam de outra entrada); `member_entities` e `unresolved_neighbors` (`semantic_map_id`) → `SEMANTIC_MAP`;
- `ContextRelation.source_run_id` → `SPATIAL_RELATIONS_RUN`;
- todo artifact em `derived_from` de qualquer origem.

`ContextMap.upstream_artifact(artifact_id)` resolve um artifact citado às suas identidades exatas; é a travessia que um consumidor usa para ir de um resultado até a evidência sem carregar debug.

### Observação física versus inferência repetida

`ArtifactKind.is_physical_observation` é verdadeiro só para `SEQUENCE`: os registros de uma sequência são frames físicos; os de qualquer outro artifact de evidência são inferências ou derivações. Um consumidor que percorre `derived_from` separa os dois sem carregar nada além do próprio mapa, e reprocessar a inferência sobre o mesmo frame não parece observação física adicional.

### Dependência estrutural versus evidência opcional

`ArtifactKind.is_structural` distingue o que é **necessário para resolver** o mapa (`GEOMETRIC_MAP`, `ENTITY_RESOLUTION_RUN`, `SPATIAL_RELATIONS_RUN`) do que é **evidência opcional** para inspeção profunda (`SEQUENCE`, `PERCEPTION_RUN`, `SEMANTIC_FUSION_RUN` e os demais). Uma dependência de debug nunca é contratual e por isso não tem tipo.

## Capacidades respaldadas pela linhagem

Uma capacidade é declarada **exatamente quando** o artifact que a produz está na linhagem:

| Capacidade | Artifact |
| --- | --- |
| `ENTITIES` | `ENTITY_RESOLUTION_RUN` |
| `RELATIONS` | `SPATIAL_RELATIONS_RUN` |
| `POINT_REPRESENTATION_EVIDENCE` | `POINT_REPRESENTATION_RUN` |

Declarar sem o artifact, ou listar o artifact sem declarar, é `ValueError`. Uma capacidade declarada com conteúdo vazio continua válida (o estágio rodou e nada encontrou).

## Linhagem de entidade e de relação

- **Resolução de entidade**: `member_entities` guarda todas as entidades de origem que foram resolvidas em uma (`EntityReference` de Semantic Mapping) e `resolution_decisions` os `ResolutionDecisionId` das decisões `MATCH` que as agruparam, como `ResolvedEntity` os define; pelo menos uma entidade de origem é obrigatória. `unresolved_neighbors` preserva as entidades que uma decisão deixou `UNRESOLVED`, para o mapa nunca afirmar mais certeza de identidade do que o run de resolução;
- **Relação**: `origin.derived_from` cita a evidência da relação (os `RelationEvidenceId` de Spatial Relations e a geometria de suporte, como `UpstreamRecordRef`) e `origin.policy` registra a política e a versão da taxonomia que a derivaram.

## Round-trip

Toda a proveniência é composta de enums, textos, referências e tuplas: sobrevive a `context_map_to_record()` / `context_map_from_record()` sem perda, e um registro adulterado (por exemplo, uma origem `sensor_observed` que cita inferência) é **rejeitado na decodificação**, nunca reparado.
