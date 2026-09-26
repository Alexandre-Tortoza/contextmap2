# Materialização de entidades

Este documento descreve `src/contextmap/semantic_mapping/materialization.py`: o serviço que converte a evidência fundida selecionada em `Entity` **sem nenhuma resolução entre suportes**.

## Regra baseline (`one-support-one-entity-v1`)

```text
um FusionSupport / FusedEvidence selecionado
→ uma Entity materializada
```

Explícita, versionada e determinística. Para cada suporte selecionado o serviço:

1. aloca um `EntityId` estável local ao artifact;
2. anexa o suporte geométrico exato e deriva os resumos espaciais ([`geometry.md`](geometry.md));
3. materializa o estado semântico preservando alternativas e conflitos ([`semantic-state.md`](semantic-state.md));
4. anexa os vínculos de evidência e a proveniência ([`evidence.md`](evidence.md));
5. deriva o estado temporal mínimo e o histórico ([`temporal-state.md`](temporal-state.md));
6. valida as invariantes do contrato de `Entity`;
7. emite um registro explícito de rejeição para todo candidato inválido.

## A fronteira com Entity Resolution

O serviço **nunca** pergunta:

```text
o suporte A é o mesmo objeto físico que o suporte B?
```

Ele não funde dois candidatos porque labels ou features se parecem, não divide um candidato em vários objetos, não re-identifica entidades entre tempo ou runs e não atualiza uma entidade existente por similaridade. Dois suportes independentes permanecem duas entidades, **mesmo com o mesmo label e geometria adjacente**, até que o milestone de Entity Resolution os resolva explicitamente. O resultado é uma função de **cada candidato isoladamente**.

Também não há extração de relações, o `FusedEvidence` upstream nunca é reescrito e nenhuma percepção ou fusão é executada.

## Identidade (`support-derived-entity-id-v1`)

`entity_id_for(fusion_support_id=...)` devolve `entity--<fusion_support_id>`, uma função pura do suporte. Por isso o id:

- é reproduzível dado o mesmo artifact de fusão selecionado;
- **não depende da ordem** em que os candidatos chegam;
- **não muda** porque um candidato irmão foi rejeitado;
- é uma identidade **local ao artifact**: não diz nada sobre outro semantic map (ver escopo de identidade em [`contracts.md`](contracts.md)).

## Entrada e seleção

```python
materialize_entities(
    outcomes,  # FusionOutcome selecionados, em qualquer ordem
    fusion_manifest=...,  # manifest do run de fusão que os possui
    geometry=...,  # GeometrySource do mapa geométrico
    semantic_map_id=...,  # identidade do semantic map em construção
    policy=EntityMaterializationPolicy(geometry=GeometrySummaryPolicy(...)),
    code_version=...,
)
```

A seleção é **explícita**: só o que é passado é lido; nenhum run de fusão é carregado automaticamente. `MaterializationInputError` recusa a seleção inteira quando um suporte é selecionado duas vezes ou quando o mapa do suporte, o da linhagem do run e o da fonte de geometria não coincidem.

`EntityMaterializationPolicy.fingerprint()` cobre a política de materialização e de identidade, as regras do estado semântico e do temporal e a política geométrica, e é gravado em `EntityProvenance.configuration_fingerprint`.

## Rejeição de candidatos

Um candidato que não vira uma entidade válida **não é descartado**: vira um `CandidateRejection` (`fusion_support_id`, `fused_evidence_id`, `reason`, `detail`).

| `RejectionReason` | Quando |
| --- | --- |
| `empty_geometry_support` | O candidato não tem suporte 3D. |
| `unresolvable_geometry` | As referências de geometria não resolvem no mapa. |
| `invalid_temporal_evidence` | A evidência temporal está ausente ou inconsistente. |
| `invalid_entity` | As partes, válidas uma a uma, não satisfazem juntas o contrato de `Entity` (`EntityContractError`). |

Só essas quatro causas viram rejeição. Qualquer outro erro durante a materialização, inclusive um `ValueError` cru levantado ao construir uma parte, é defeito e **propaga**: engoli-lo rejeitaria todos os candidatos em silêncio e o run terminaria "bem-sucedido" com um mapa vazio (#604).

`EntityMaterialization` guarda as entidades (ordenadas por id) e as rejeições (ordenadas por suporte); cada candidato está em exatamente um dos dois.

Um suporte cujo `FusedEvidence` não tem nenhuma hipótese (por exemplo, só abstenções) **não é rejeitado**: vira uma entidade com estado semântico `insufficient_evidence`, porque o mapa deve registrar "há algo aqui, sem classe conhecida".

## Determinismo

Dados o mesmo artifact de fusão selecionado, a mesma política e a mesma configuração, os ids, a ordem e o estado das entidades são reproduzíveis, independentemente da ordem da seleção.

## Validação

`tests/semantic_mapping/test_entity_materialization.py`, com um run de fusão real escrito em disco: entidade completa a partir de evidência real, dois suportes independentes permanecendo duas entidades, uma vista de dois runs permanecendo uma, ausência de qualquer API de merge/split/match, identidade pura e independente de irmãos rejeitados, ordem da seleção, rejeições e erros de seleção.
