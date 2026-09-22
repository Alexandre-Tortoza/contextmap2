# Montagem do `ContextMap` a partir de runs reais

Este documento descreve `src/contextmap/artifact/serialization/assembly.py` (issue [#537](https://github.com/Alexandre-Tortoza/contextmap2/issues/537)). O schema está em [`composition.md`](composition.md) e [`lineage.md`](lineage.md); a persistência em disco, em [`writer.md`](writer.md).

## Por que existe

Antes desta issue, nada transformava um run real de Entity Resolution e um de Spatial Relations num `ContextMap` real: o único código que fazia essa ligação era um builder **só de teste** (`tests/artifact/context_map_serialization_builders.py::assemble_from_runs`), cujo próprio docstring admitia escolher o `semantic_state` e a `origin` de cada registro num rodízio de três valores fixos, em vez de os derivar da evidência. `assemble_context_map` é a implementação real: uma função determinística, sem estado, chamável fora da CLI, que faz exatamente essa tradução.

## Por que mora em `serialization/`, não ao lado de `composition.py`

`tests/artifact/test_context_map_invariants.py::test_the_schema_package_performs_no_file_io_and_owns_no_format` proíbe qualquer módulo de topo de `contextmap/artifact/*.py` (`composition.py`, `models.py`, `metadata.py`, …) de importar `pathlib`, `os`, `json` e afins: o schema não sabe o que é um arquivo. `assemble_context_map` abre diretórios de run reais (`EntityResolutionRunReader`, `SpatialRelationsRunReader`, `GeometricMapArtifactReader`) e lê manifests do disco para calcular `content_identity` — exatamente a mesma categoria de trabalho que `structural_dependencies.py` já faz do mesmo lugar. Por isso o módulo mora em `serialization/`, ao lado dele, e é reexportado na raiz da capability (`contextmap.artifact.assemble_context_map`) como qualquer outro contrato público de serialização (`ContextMapArtifactWriter`, `artifact_digest`).

## Uso

```python
context_map = assemble_context_map(
    context_map_id=ContextMapId("context-map--corridor-02--0007"),
    metadata=metadata,  # ContextMapMetadata já pronto: criação (com a política de montagem
    # explícita e versionada), sequências de origem, frame, extensão, janela
    # temporal e capacidades declaradas
    geometric_map_location=Path("…/geometric_mapping"),
    entity_resolution_location=Path("…/entity_resolution"),  # None para um mapa só de geometria
    spatial_relations_location=Path("…/spatial_relations"),  # None sem entity_resolution_location
    additional_lineage=(sequence_upstream_artifact,),  # ver "O que esta função não pinca"
)
```

O retorno é um `ContextMap` validado (referências, fechamento de proveniência, capacidades declaradas versus conteúdo — tudo isso já roda em `ContextMap.__post_init__`); persisti-lo continua sendo trabalho do `ContextMapArtifactWriter`.

## O que é traduzido, e o que é copiado tal e qual

A montagem nunca resolve uma entidade nem decide uma relação: Entity Resolution e Spatial Relations já decidiram tudo. Dois tratamentos diferentes, de propósito:

- **copiado verbatim**: `member_entities`, `resolution_decisions`, `unresolved_neighbors` e `geometry_refs` de cada `ResolvedEntity`, e o sujeito, o objeto, o `predicate`, o `state` e os `uncertainty_kinds` de cada `Relation`. `composition.py` já exige que a cópia local da entidade/relação seja **igual** ao registro a montante (a mesma disciplina que `structural_dependencies.py` verifica depois, "provar conteúdo, não só existência"), então copiar é a única tradução honesta;
- **traduzido**: o estado semântico. Ver a seção seguinte.

### `ResolvedSemanticState` → `ContextSemanticState`, campo a campo

| `ResolvedSemanticState` (Entity Resolution) | `ContextSemanticState` (`contextmap.artifact`) | Decisão |
| --- | --- | --- |
| `ambiguity_state: AmbiguityState` | `status: AmbiguityStatus` | Tradução direta pelo valor do enum: os dois têm exatamente os mesmos quatro valores (`unambiguous`, `ambiguous`, `conflicting`, `insufficient_evidence`). |
| `hypotheses: tuple[EntityHypothesis, ...]`, chaveadas por `(fused_evidence_id, hypothesis_id)` | `hypotheses: tuple[LabelHypothesis, ...]`, chaveadas só pelo texto do `label` | **Descasamento genuíno de granularidade**, não um atalho de montagem: duas hipóteses de evidências fundidas diferentes podem propor o mesmo rótulo, e `ContextSemanticState` recusa duas `LabelHypothesis` com o mesmo `label` (`require_canonical`). A montagem agrupa por texto do rótulo: dois itens que concordam no rótulo são evidência que se reforça, não duas hipóteses concorrentes, então isso não é achatar ambiguidade — ambiguidade é exatamente a presença de mais de um rótulo **distinto**, e todo rótulo distinto proposto por qualquer membro sobrevive como sua própria hipótese. Nenhum score nem contagem é inventado para a fusão: o schema não tem campo para nenhum dos dois. |
| `attributes: tuple[EntityAttribute, ...]` | *(sem campo correspondente)* | Não copiado. `composition.py` já documenta a política: "Support values are deliberately not copied; the evidence behind a result stays reachable through the upstream artifact." Um consumidor que precisa de atributos abre o run de Entity Resolution que `source`/`origin` da entidade já citam. |
| `uncertainty: tuple[EntityUncertainty, ...]` (registros detalhados) | *(sem campo correspondente; só `status` resume)* | Mesma política: o resumo (`status`) sobrevive, o registro detalhado fica a montante. |
| `member_ambiguity: tuple[MemberAmbiguity, ...]` | *(sem campo correspondente)* | Idem; é lineage interna de Entity Resolution, não parte do schema do mapa. |

`LabelHypothesis.origin` e o `ContextEntity.origin` da entidade citam o mesmo registro (o `resolved_entity_id` no run de Entity Resolution, o único endereço que o leitor público expõe nesse grão) com `DerivationKind.MULTIVIEW_FUSED` e a política de materialização do run (`materialization().policy`, traduzida — ver abaixo). `ContextRelation.origin` cita o `relation_id` no run de Spatial Relations com `DerivationKind.GEOMETRY_DERIVED` e `PolicyRef(decision_policy_id, taxonomy_version)` do próprio `RelationProvenance` da relação — exatamente como `composition.md` já documentava essa origem, sem mudança.

### Extensão deliberada de `MULTIVIEW_FUSED`

`_lineage_checks.py` exigia, antes desta issue, que uma origem `MULTIVIEW_FUSED` citasse pelo menos um `SEMANTIC_FUSION_RUN` na linhagem — mas a montagem, por desenho (issue #537), nunca abre um run de Semantic Fusion. A materialização de Entity Resolution **também** é uma regra versionada (`connected-components-materialization-v1`) que acumula várias contribuições de evidência — as dos membros que fundiu — o que é exatamente o que `MULTIVIEW_FUSED` descreve. `_GROUNDING[DerivationKind.MULTIVIEW_FUSED]` foi por isso ampliado para aceitar também `ArtifactKind.ENTITY_RESOLUTION_RUN`: uma ampliação mínima e só permissiva (nenhum caso antes aceito passa a ser rejeitado) do único módulo que já possui essa política. Ver [`lineage.md`](lineage.md).

### `PolicyRef` de Entity Resolution não tem `version`

`contextmap.entity_resolution.PolicyRef` é `(policy_id, configuration_fingerprint)`: a versão da regra é, por convenção daquela capability, um sufixo de `policy_id` (`…-v1`), e `configuration_fingerprint` é a configuração efetiva de uma execução. `contextmap.artifact.PolicyRef` exige `version` explícito. A montagem carrega `configuration_fingerprint` como `version`: é o único sinal que o contrato de Entity Resolution de fato registra e que distingue uma execução da regra de outra; nada é inventado para preencher o campo.

## Linhagem: o que a montagem deriva de graça, e o que ela nunca pode pinçar sozinha

A montagem pinça (`artifact_digest`, o mesmo digest que os irmãos já usam) o mapa geométrico, o run de Entity Resolution e o de Spatial Relations que de fato abriu. Um `SEMANTIC_MAP` por `semantic_map_id` referenciado por `member_entities`/`unresolved_neighbors` também é derivado — de graça, sem abrir nenhum leitor a mais — porque o manifest de Entity Resolution que a montagem já leu nomeia o run de Semantic Mapping de origem e o digest que aquele run já publicou (`ResolutionRunLineage.semantic_mapping_run_id`/`semantic_mapping_artifact_digest`).

Um `SEQUENCE` **não pode** ser derivado do mesmo jeito: toda `ContextMapMetadata` exige pelo menos uma `source_sequences`, e a validação de linhagem do mapa exige o `SEQUENCE` correspondente — mas pinçar seu digest exigiria abrir um artifact de `contextmap.ingestion`, uma dependência que `contextmap.artifact` não tem e não deveria ganhar só para isto (`tests/architecture/test_boundaries.py`). Por isso `assemble_context_map` recebe `additional_lineage`: um `tuple[UpstreamArtifact, ...]` que quem chama (a runtime, que já tem a identidade da sequência) fornece já pinçado. Um `artifact_id` duplicado entre `additional_lineage` e o que a montagem deriva é recusado pelo próprio `ContextMap`, nunca sobrescrito em silêncio.

## Rejeições explícitas

| Situação | Erro |
| --- | --- |
| `spatial_relations_location` sem `entity_resolution_location` | `ValueError` |
| o run de Entity Resolution ou o de Spatial Relations foi construído sobre um mapa geométrico diferente do referenciado | `UpstreamArtifactError` |
| o run de Spatial Relations foi construído sobre um run de Entity Resolution diferente do aberto (identidade, versão de schema, digest, ou uma entidade resolvida que a relação cita e o run não tem) — o que o próprio `SpatialRelationsRunReader.validate_resolution` já relata | `UpstreamArtifactError` |
| uma relação cita uma entidade resolvida ausente do conjunto montado (defensivo: `validate_resolution` já teria relatado isso antes) | `UnresolvedReferenceError` |

## Ordenação canônica, independente do reader

`resolved_entities().entities` já vem ordenado e único por `resolved_entity_id` — um hash dos membros, o `ResolvedEntitySet` exige isso na própria construção — então numerar `entity-0001..N` nessa ordem já é determinístico, qualquer que seja a ordem física do arquivo. `iter_relations()` **não** vem ordenado por `relation_id` (a ordem própria do run é por sujeito/predicado/objeto, que não tem relação com o hash do `relation_id`): a montagem ordena explicitamente por `str(relation_id)` antes de numerar `relation-0001..M`, o que é o que de fato torna a numeração do mapa independente da ordem do reader.

## Fora de escopo (por desenho desta issue)

Conceitos de `ContextBuild`/`ContextRun`/contexto incremental (v0.1.1, issue #499); execução de um estágio de runtime `context_map` que chame esta função; e qualquer mudança no layout em disco do `ContextMapArtifact`.
