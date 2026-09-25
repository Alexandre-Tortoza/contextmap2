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

`ContextEntity.origin` cita o `resolved_entity_id` no run de Entity Resolution (o único endereço que o leitor público expõe nesse grão) com `DerivationKind.MULTIVIEW_FUSED` e a política de materialização do run (`materialization().policy`, traduzida — ver abaixo). Isso é honesto: a materialização (agrupar membros e acumular a crença resultante) é de fato a regra que produziu *este* resultado, o `ResolvedEntity`.

Cada `LabelHypothesis.origin`, ao contrário, **não** reutiliza esse mesmo registro (issue [#541](https://github.com/Alexandre-Tortoza/contextmap2/issues/541), corrigido nesta revisão — ver "Proveniência de `LabelHypothesis`" abaixo). `ContextRelation.origin` cita a evidência e a geometria de suporte da relação com `DerivationKind.GEOMETRY_DERIVED` — ver "Proveniência de `ContextRelation`" abaixo, também revisado por #541.

### Proveniência de `LabelHypothesis`: por que não é o mesmo registro da entidade

Antes de #541, `LabelHypothesis.origin` reutilizava o mesmo `EvidenceOrigin` de `ContextEntity.origin` — citando o `resolved_entity_id` do run de Entity Resolution como se Entity Resolution tivesse *produzido* o rótulo. Isso está errado por construção: Entity Resolution resolve identidade e agrega crença já existente; quem propôs o rótulo ("chair", "pallet", …) foi Semantic Fusion, a partir de uma `SemanticClaim`, e Entity Resolution nunca abre uma claim, uma region ou uma saída de modelo. Colapsar essas duas camadas no mesmo registro é exatamente o achatamento de nível semântico que o §4 do `AGENTS.md` proíbe (evidência → crença → conhecimento).

A correção usa a informação que já está em `ResolvedEntity`, sem abrir o artifact de Semantic Fusion: `ResolvedEntity.evidence: EntityEvidenceLinks` guarda `FusedEvidenceRef`s (com `fusion_run_id`, `fusion_artifact_digest` e `fused_evidence_id`), e cada `EntityHypothesis` de `ResolvedSemanticState.hypotheses` carrega seu próprio `fused_evidence_id`. `_label_hypothesis_origin` casa cada hipótese do rótulo mesclado com o `FusedEvidenceRef` correspondente (por `fused_evidence_id`) e cita `UpstreamRecordRef(fusion_run_id, fused_evidence_id)` — nunca o `resolved_entity_id`. O `SEMANTIC_FUSION_RUN` correspondente é derivado de graça (ver "Linhagem" abaixo), sem abrir o run de Semantic Fusion: `FusedEvidenceRef` já carrega a identidade e o digest pinçados por Semantic Mapping quando materializou a entidade de origem.

`kind` continua `MULTIVIEW_FUSED` (o rótulo genuinamente foi acumulado de uma ou mais contribuições de evidência por uma regra versionada — é exatamente essa a categoria), mas a `policy` citada **não é** a política de materialização de Entity Resolution: mesclar as (possivelmente várias) evidências fundidas que propuseram o mesmo texto de rótulo em uma única `LabelHypothesis` é uma regra da própria montagem, `LABEL_HYPOTHESIS_MERGE_POLICY_ID = "label-hypothesis-merge-by-text-v1"`. Reaproveitar a política de materialização aqui só moveria o mesmo achatamento de nível epistêmico um campo adiante. `ContextEntity.origin` continua legitimamente citando a materialização de Entity Resolution — essa distinção é esperada, não um bug.

Isto foi resolvido sem ampliar ainda mais o significado de `MULTIVIEW_FUSED`: `_GROUNDING[DerivationKind.MULTIVIEW_FUSED]` já aceitava `SEMANTIC_FUSION_RUN` desde antes de #537 (ver "Extensão deliberada de `MULTIVIEW_FUSED`" abaixo, que continua descrevendo por que `ENTITY_RESOLUTION_RUN` também é aceito, para `ContextEntity.origin`).

### Proveniência de `ContextRelation`: evidência e geometria de suporte, não o próprio registro

Antes de #541, `ContextRelation.origin.derived_from` citava só `UpstreamRecordRef(source_run_id, relation.relation_id)` — o próprio registro da relação. Isso é redundante com `source_run_id`/`source_relation_id`, que já nomeiam esse registro estruturalmente (`docs/lineage.md`), e degrada a profundidade da proveniência que o contrato de Spatial Relations já expõe: `Relation.relation_evidence_refs` (as evidências que realmente justificaram a decisão) e, quando o run tem, a geometria de suporte de cada evidência (`RelationEvidence.geometry[].geometry_refs`).

A montagem agora cita, em `derived_from`: cada `RelationEvidenceId` de `relation.relation_evidence_refs` (no `SPATIAL_RELATIONS_RUN`); a geometria de suporte de cada evidência, lida com `reader.evidence_of(relation)` — o mesmo run já aberto, nenhum leitor a mais — quando a evidência lista referências explícitas (no `GEOMETRIC_MAP` já citado na linhagem); e, quando a relação é derivada (inversa ou simétrica), a relação de origem (`Relation.derived_from`, também no `SPATIAL_RELATIONS_RUN`). O próprio registro da relação só é citado como último recurso, quando não sobra mais nada: uma relação `UNRESOLVED` por falta de evidência pode não ter nenhuma evidência nem `derived_from`, e `EvidenceOrigin` exige ao menos uma citação.

### Extensão deliberada de `MULTIVIEW_FUSED`

`_lineage_checks.py` exigia, antes da issue #537, que uma origem `MULTIVIEW_FUSED` citasse pelo menos um `SEMANTIC_FUSION_RUN` na linhagem — mas a montagem, por desenho, nunca abre um run de Semantic Fusion. A materialização de Entity Resolution **também** é uma regra versionada (`connected-components-materialization-v1`) que acumula várias contribuições de evidência — as dos membros que fundiu — o que é exatamente o que `MULTIVIEW_FUSED` descreve. `_GROUNDING[DerivationKind.MULTIVIEW_FUSED]` foi por isso ampliado para aceitar também `ArtifactKind.ENTITY_RESOLUTION_RUN`: uma ampliação mínima e só permissiva (nenhum caso antes aceito passa a ser rejeitado) do único módulo que já possui essa política, usada por `ContextEntity.origin`. Ver [`lineage.md`](lineage.md).

### `PolicyRef` de Entity Resolution e de Spatial Relations não têm `version` explícito

`contextmap.entity_resolution.PolicyRef` é `(policy_id, configuration_fingerprint)`, e o próprio docstring da capability é explícito que os dois "nunca são confundidos": a versão da regra é, por convenção daquela capability, um sufixo numerado de `policy_id` (`…-v1`), e `configuration_fingerprint` é a configuração efetiva de *uma execução* da regra — duas execuções da mesma versão de regra com parâmetros diferentes são duas execuções, não duas regras. `contextmap.spatial_relations.RelationProvenance` tem o problema análogo: `decision_policy_id` (`conservative-relation-decision-v1`) segue a mesma convenção, e `taxonomy_version` versiona o *vocabulário de predicados* (`RelationPredicate`), não a regra de decisão — dois runs sob exatamente a mesma versão da política de decisão podem usar versões de taxonomia diferentes, e vice-versa.

Até a issue [#541](https://github.com/Alexandre-Tortoza/contextmap2/issues/541), a montagem carregava `configuration_fingerprint`/`taxonomy_version` como `version` de `contextmap.artifact.PolicyRef` — que documenta "uma versão diferente é uma regra diferente". Isso é falso sob essa tradução: duas configurações diferentes da *mesma* versão de regra apareceriam no mapa como duas regras diferentes, e a versão de taxonomia nunca foi a versão da regra de decisão. A correção usa o que o próprio `policy_id`/`decision_policy_id` já registra: `_split_policy_id` recupera o sufixo convencional `-v<N>` (por exemplo `connected-components-materialization-v1` → `policy_id="connected-components-materialization"`, `version="v1"`), sem inventar nada e sem reaproveitar uma identidade de outro conceito.

`taxonomy_version` não tem um campo correspondente em `contextmap.artifact.PolicyRef`: como os demais campos que `docs/assembly.md` já documenta como não copiados (`attributes`, `uncertainty` detalhado), continua alcançável, sem alteração, no próprio registro que `ContextRelation.source_run_id`/`source_relation_id` cita.

**Segunda rodada de revisão do #541**: a primeira correção citava `materialization().policy.configuration_fingerprint` em `_resolution_upstream_artifact` e a `configuration_fingerprint` da primeira relação em `_relations_upstream_artifact`, como se cada um representasse a configuração efetiva do run inteiro. Isso também estava errado: um run de Entity Resolution tem vários papéis configurados independentemente (materialização, retrieval de candidatos, portões de comparação, ...), e um de Spatial Relations tem várias políticas (`frame_conventions`, `candidate`, `geometry_summary`, `geometric`, `contact`) — a montagem só abre/lê um papel de cada um, e citar o fingerprint desse único papel como se fosse o do run inteiro é exatamente o mesmo tipo de conflação parcial que motivou a correção original. `UpstreamArtifact.configuration_fingerprint` de `ENTITY_RESOLUTION_RUN`/`SPATIAL_RELATIONS_RUN` fica `None` até que a capability correspondente exponha um fingerprint que cubra o run inteiro; o fingerprint de materialização/decisão continua alcançável, sem alteração, pelos mesmos leitores públicos (nunca perdido, só não citado onde não pertence).

## Linhagem: o que a montagem deriva de graça, e o que ela nunca pode pinçar sozinha

A montagem pinça (`artifact_digest`, o mesmo digest que os irmãos já usam) o mapa geométrico, o run de Entity Resolution e o de Spatial Relations que de fato abriu. Um `SEMANTIC_MAP` por `semantic_map_id` referenciado por `member_entities`/`unresolved_neighbors` também é derivado — de graça, sem abrir nenhum leitor a mais — porque o manifest de Entity Resolution que a montagem já leu nomeia o run de Semantic Mapping de origem e o digest que aquele run já publicou (`ResolutionRunLineage.semantic_mapping_run_id`/`semantic_mapping_artifact_digest`).

Um `SEMANTIC_FUSION_RUN` por fusion run citado pelas `LabelHypothesis` de qualquer entidade é derivado da mesma forma gratuita, desde a issue #541: `ResolvedEntity.evidence.fused_evidence` já carrega `fusion_run_id` e `fusion_artifact_digest` pinçados desde que Semantic Mapping materializou a entidade de origem — nenhum leitor de Semantic Fusion é aberto. Só os fusion runs que alguma `LabelHypothesis` de fato cita entram na linhagem (deduplicados entre entidades); um `fusion_run_id` derivado com dois `content_identity` diferentes por entidades distintas é um `ProvenanceError` — nunca resolvido em silêncio.

Um `SEQUENCE` **não pode** ser derivado do mesmo jeito: toda `ContextMapMetadata` exige pelo menos uma `source_sequences`, e a validação de linhagem do mapa exige o `SEQUENCE` correspondente — mas pinçar seu digest exigiria abrir um artifact de `contextmap.ingestion`, uma dependência que `contextmap.artifact` não tem e não deveria ganhar só para isto (`tests/architecture/test_boundaries.py`). Por isso `assemble_context_map` recebe `additional_lineage`: um `tuple[UpstreamArtifact, ...]` que quem chama (a runtime, que já tem a identidade da sequência) fornece já pinçado. Um `artifact_id` duplicado entre `additional_lineage` e o que a montagem deriva é recusado pelo próprio `ContextMap`, nunca sobrescrito em silêncio.

### `additional_lineage` não aceita dependência estrutural nem `SEMANTIC_MAP` (issue #541, blocker 1)

Antes de #541, `additional_lineage` aceitava **qualquer** `UpstreamArtifact`, sem restrição. Isso permitia um uso indevido sério: um chamador podia passar uma entrada `ENTITY_RESOLUTION_RUN` por `additional_lineage`, nunca fornecer um `entity_resolution_location` real, declarar `ENTITIES` em `metadata.capabilities`, e receber de volta um `ContextMap` com `entities=()` — um mapa que *afirma* estar fundamentado em um run real de Entity Resolution (citando-o na linhagem, declarando a capacidade) sem conter nenhuma de suas entidades. As checagens existentes não pegavam isso: `_lineage_checks.py` só verifica a *presença* de algum artifact do tipo certo, e `structural_dependencies.py` só valida entidades que já estão no mapa — com zero entidades não há nada para cruzar.

A montagem agora rejeita, em `additional_lineage`, qualquer `UpstreamArtifact` cujo `kind.is_structural` seja verdadeiro (`GEOMETRIC_MAP`, `ENTITY_RESOLUTION_RUN`, `SPATIAL_RELATIONS_RUN`) ou cujo `kind` seja `SEMANTIC_MAP` — os únicos tipos que a montagem sempre deriva sozinha, de um run que de fato abriu (ou de graça, de um manifest já aberto). `SEQUENCE` e evidência opcional (`PERCEPTION_RUN`, `POINT_REPRESENTATION_RUN`, `HUMAN_ANNOTATION_SET`, …) continuam aceitos: esse é o uso legítimo de "pinçar" que `additional_lineage` sempre existiu para cobrir.

### A metadata é cruzada com o mapa geométrico de fato aberto (issue #541, should-strengthen 5)

`metadata` é sempre fornecida pelo chamador e nunca derivada dos runs que a montagem abre. Antes de #541, isso deixava uma lacuna: era possível montar um `ContextMap` que *declarava* uma sequência/seleção, frame ou janela temporal, enquanto a geometria por trás na verdade veio de outra — desde que `additional_lineage` (já restrito pelo item anterior, mas ainda fornecido pelo chamador) citasse a identidade alegada. `_require_metadata_matches_geometry` fecha essa lacuna para tudo que o `GeometricMapArtifactManifest` já aberto consegue provar sozinho: `metadata.source_sequences` precisa conter o `(sequence_artifact_id, selection_id)` real do manifest; `metadata.frame.frame_id` precisa ser o `map_frame` real; e, quando os dois compartilham o mesmo domínio de clock, `metadata.time_bounds` precisa ser exatamente `[start_time_ns, end_time_ns]`. Uma divergência é `UpstreamArtifactError`, no mesmo padrão das rejeições ER↔geometria e SR↔ER já existentes.

## Rejeições explícitas

| Situação | Erro |
| --- | --- |
| `spatial_relations_location` sem `entity_resolution_location` | `ValueError` |
| `additional_lineage` contém um artifact estrutural (`GEOMETRIC_MAP`, `ENTITY_RESOLUTION_RUN`, `SPATIAL_RELATIONS_RUN`) ou `SEMANTIC_MAP` | `ValueError` |
| `metadata` diverge da sequência/seleção, do frame ou da janela temporal que o mapa geométrico realmente aberto reporta | `UpstreamArtifactError` |
| o run de Entity Resolution ou o de Spatial Relations foi construído sobre um mapa geométrico diferente do referenciado | `UpstreamArtifactError` |
| o run de Spatial Relations foi construído sobre um run de Entity Resolution diferente do aberto (identidade, versão de schema, digest, ou uma entidade resolvida que a relação cita e o run não tem) — o que o próprio `SpatialRelationsRunReader.validate_resolution` já relata | `UpstreamArtifactError` |
| uma relação cita uma entidade resolvida ausente do conjunto montado (defensivo: `validate_resolution` já teria relatado isso antes) | `UnresolvedReferenceError` |

## Métricas de execução (`assemble_context_map_with_metrics`)

A issue [#537](https://github.com/Alexandre-Tortoza/contextmap2/issues/537) pede que a montagem meça e relate o tempo de execução, o pico de memória e as contagens de entidades/relações/linhagem. `assemble_context_map_with_metrics` envolve exatamente a mesma tradução de `assemble_context_map` (sem duplicar lógica: as duas chamam o mesmo `_assemble_fields` interno) e devolve um `AssemblyResult(context_map, metrics)`, sem alterar a assinatura nem o comportamento de `assemble_context_map` — todo chamador existente continua intacto.

`AssemblyMetrics` separa `translation_duration_seconds` (abrir os runs a montante e traduzir seus registros, até — sem incluir — construir o `ContextMap`) de `validation_duration_seconds` (a própria construção do `ContextMap`, que é exatamente quando `composition.py` e `_lineage_checks.py` rodam, dentro de `__post_init__`): essa é a "validação" real da issue #537, não um passo extra inventado. Esta função nunca escreve em disco, então **não relata uma duração de escrita** — para isso existe `write_context_map_with_metrics`, uma função irmã que envolve `ContextMapArtifactWriter.write` (sem alterá-lo) e devolve `WriteMetrics(write_duration_seconds)`, mantendo as duas medições separadas exatamente como a issue #537 pede, em vez de uma função só fazer montagem e escrita juntas. `peak_memory_bytes` usa `tracemalloc` (biblioteca padrão), a única medição de pico de memória de CPU comparável já usada no repositório sendo específica de backends de modelo (`torch`/GPU), não aplicável a uma composição determinística como esta. `entity_count`, `relation_count` e `lineage_count` são lidos do próprio `ContextMap` já validado.

## Ordenação canônica, independente do reader

`resolved_entities().entities` já vem ordenado e único por `resolved_entity_id` — um hash dos membros, o `ResolvedEntitySet` exige isso na própria construção — então numerar `entity-0001..N` nessa ordem já é determinístico, qualquer que seja a ordem física do arquivo. `iter_relations()` **não** vem ordenado por `relation_id` (a ordem própria do run é por sujeito/predicado/objeto, que não tem relação com o hash do `relation_id`): a montagem ordena explicitamente por `str(relation_id)` antes de numerar `relation-0001..M`, o que é o que de fato torna a numeração do mapa independente da ordem do reader.

## Fora de escopo (por desenho desta issue)

Conceitos de `ContextBuild`/`ContextRun`/contexto incremental (v0.1.1, issue #499); execução de um estágio de runtime `context_map` que chame esta função; e qualquer mudança no layout em disco do `ContextMapArtifact`.

A duração de escrita (issue #541, achado 6) deixou de ser uma lacuna: `write_context_map_with_metrics` (acima) mede exatamente isso, como uma função irmã de `assemble_context_map_with_metrics`, não como uma alteração de `ContextMapArtifactWriter` em si.
