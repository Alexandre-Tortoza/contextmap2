# Composição: geometria, entidades e relações

Este documento descreve `src/contextmap/artifact/composition.py`, `references.py` e `_invariants.py`. O `ContextMap` conecta geometria autoritativa, entidades resolvidas e relações espaciais **sem duplicar os artifacts a montante e sem criar propriedade ambígua**.

```text
ContextMap
├── geometry_ref   -> GeometricMapArtifact (por identidade e tamanho)
├── entities[]     -> ContextEntity   ── geometry_refs[] -> GeometryReference
└── relations[]    -> ContextRelation ── subject/object   -> ContextEntityReference
```

## Decisões

| Pergunta | Decisão |
| --- | --- |
| A geometria é embutida, referenciada ou empacotada? | **Referenciada**. Nenhum ponto é copiado; um bundle é decisão de transporte do serializador e não muda o schema. |
| Escopo da identidade de entidade | `ContextEntityId` é único **dentro de um** `ContextMap`. A referência estável é `(context_map_id, entity_id)`. |
| Identidades de origem | Preservadas em `source` (`UpstreamRecordRef`: artifact + id local). O id no mapa **pode** diferir do id de origem, mas o mapeamento é sempre explícito; nada é reescrito em silêncio. |
| Sujeito e objeto de uma relação | `ContextEntityReference` para entidades **deste** mapa; a relação é direcionada (`sujeito predicado objeto`). |
| Índices | Derivados e reconstruíveis a partir das listas; nunca autoritativos (ver abaixo). |
| Estado não resolvido ou opcional | Preservado, nunca descartado nem promovido (ver abaixo). |

## Referências e escopos

- `ContextEntityReference` de outro mapa é recusada mesmo quando o id existe (`ForeignContextEntityReferenceError`), separando esse caso de "entidade inexistente" (`UnknownContextEntityError`). Equivalência entre mapas nunca é deduzida de ids com o mesmo texto;
- `UpstreamRecordRef(artifact_id, record_id)` nomeia o registro exato no artifact exato; o id do artifact não pode ser um caminho;
- dois registros do mapa nunca mapeiam o mesmo registro a montante (a identidade seria ambígua): `ReferenceIntegrityError`.

## Entidade

`ContextEntity` guarda `entity_id`, `source`, `geometry_refs` e `semantic_state`.

- **Suporte de geometria autoritativo**: pelo menos um `GeometryReference`, ordenado e único, **nunca coordenadas**. O mapa valida que cada referência pertence ao mapa geométrico referenciado, usa a identidade canônica (`geometry_index_of`) e está dentro do intervalo `[0, point_count)`. Assim toda entidade resolve a suporte existente sem abrir a geometria; a verificação contra o `GeometricMap` real é do validador do artifact;
- várias entidades podem compartilhar elementos de geometria; o mapa não decide se isso é ou não um problema de resolução;
- resumos derivados (centroide, bounds) **não** fazem parte do schema: não têm consumidor hoje e seriam um segundo dado a manter consistente com a geometria. Podem ser adicionados depois como campo opcional (mudança aditiva, ver [`versioning.md`](versioning.md)).

### Estado semântico sem colapso de incerteza

`ContextSemanticState(status, hypotheses)`; `LabelHypothesis` guarda só o `label` verbatim (vocabulário aberto, sem score: os números que sustentam uma hipótese continuam no artifact a montante).

| `AmbiguityStatus` | Hipóteses exigidas |
| --- | --- |
| `UNAMBIGUOUS` | exatamente uma |
| `AMBIGUOUS` | pelo menos duas, sem observações contraditórias |
| `CONFLICTING` | pelo menos duas, com observações que sustentam hipóteses incompatíveis |
| `INSUFFICIENT_EVIDENCE` | nenhuma; é abstenção, não evidência negativa |

O status precisa concordar com as hipóteses: um estado competitivo **não pode** ser reduzido a um rótulo pelo mapa. A ordem das hipóteses (por label) não é ranking.

## Relação

`ContextRelation` guarda `relation_id`, `source`, `subject`, `predicate`, `object` e `state`.

- sujeito e objeto resolvem a entidades **do mesmo mapa** (`ReferenceIntegrityError` caso contrário) e são distintos: uma relação de uma entidade consigo mesma não existe no schema;
- `RelationState`: `SUPPORTED`, `UNRESOLVED`, `CONFLICTING`. Uma relação candidata nunca é uma relação confirmada; relações não resolvidas ou conflitantes **permanecem no mapa com seu estado**, e um consumidor que quer só as confirmadas filtra por `SUPPORTED`;
- `predicate` vem da taxonomia da política que produziu a relação; direção, simetria e versão da taxonomia pertencem a Spatial Relations. Enquanto o contrato de Spatial Relations não existe na `dev`, o schema guarda o predicado como texto canônico.

## Travessia sem consulta

`ContextMap.entity(reference)` e `ContextMap.relations_for(reference)` cobrem entidade → geometria (`geometry_refs`), relação → entidades (`subject`/`object`) e entidade → relações (como sujeito ou objeto, em ordem de `relation_id`). São mapas de busca em memória, derivados e reconstruíveis; **não há motor de consulta, grafo nem dependência de backend**, e um leitor leve implementa a mesma travessia sobre payloads preguiçosos.

## Capacidades versus conteúdo

`ContextMap` valida a declaração de `DeclaredCapabilities` contra o conteúdo real:

- entidades existentes exigem `ENTITIES` declarada; relações existentes exigem `RELATIONS`;
- com `RELATIONS` declarada, `relation_predicates` é **exatamente** o conjunto de predicados presentes;
- uma capacidade declarada com conteúdo vazio é válida (o estágio rodou e nada encontrou), e é diferente de uma capacidade ausente.

## Ordenação canônica

Entidades e relações são ordenadas por id e únicas; referências de geometria, por `(map_id, geometry_id)`. Dois mapas equivalentes codificam para o mesmo registro.
