# Entidades resolvidas

Uma `ResolvedEntity` é a **crença** de que várias entidades de origem são um mesmo objeto físico, mais a linhagem exata dessa crença. Ela é derivada só dos membros (`Entity`) e das decisões `MATCH` (`ResolutionDecision`) que os agruparam, e **nunca substitui os membros**: as entidades de origem continuam imutáveis e endereçáveis pela própria `EntityReference`. `materialize_resolved_entities` é determinística e produz o conjunto, sem nenhuma decisão implícita.

## Toda entidade de origem pertence a exatamente uma resolvida

Uma entidade que nada casou vira uma entidade resolvida de um membro. Assim Spatial Relations consome **só** entidades resolvidas e não depende de mutação de entidade de origem. `ResolvedEntitySet` recusa uma entidade de origem em duas entidades resolvidas e oferece `resolved_of(entity_ref)` (o índice origem para resolvida) e `resolve(ResolvedEntityReference)`, que espelha `EntitySet.resolve`: referência de outro artifact levanta `ForeignResolvedEntityReferenceError` e id inexistente levanta `UnknownResolvedEntityError`.

## Identidade

`resolved_entity_id_for(membros)` deriva o id do hash das referências ordenadas dos membros: as mesmas entradas imutáveis dão sempre o mesmo id, seja qual for a ordem das entidades ou das decisões. O id é **local ao artifact**: o handle estável é `ResolvedEntityReference(resolution_run_id, resolved_entity_id)`, e o mesmo id em dois artifacts são duas referências diferentes, então nada implica permanência entre execuções. `EntityId` de origem nunca é sobrescrito, e o id resolvido nunca coincide com o de um membro.

## Agrupamento (`connected-components-materialization-v1`)

Entidades ligadas por decisões `MATCH` formam uma entidade resolvida (componentes conexos do grafo de matches). `DISTINCT` e `UNRESOLVED` nunca agrupam; um `UNRESOLVED` é registrado como `unresolved_neighbor_refs` das duas entidades (fora do grupo). Um par tem **uma** decisão por execução: decidir o mesmo par duas vezes, nomear uma entidade que não foi dada ou repetir uma entidade são erros explícitos (`MaterializationError`).

### Transitividade e contradições

`A MATCH B` e `B MATCH C` agrupam `A` e `C`, mesmo que alguma decisão diga `A DISTINCT C`. Fundir o componente esconderia essa contradição, então a regra é conservadora e explícita:

- um componente com um par `DISTINCT` **não é fundido**: cada membro continua sua própria entidade resolvida;
- **cada** par `DISTINCT` dentro do componente é uma `TransitivityContradiction` própria, com o id derivado da decisão `DISTINCT`: ela nomeia essa decisão, o par que separa, a **cadeia mais curta de decisões `MATCH`** que o liga e o componente retido;
- cada entidade retida guarda os `contradiction_ids` de **todas** as contradições do componente (um componente pode ter várias, inclusive pares que compartilham uma entidade), então o consumidor vê por que um match provável não foi honrado.

Um componente com várias contradições continua não fundido e cada uma é reportada; o resultado não depende da ordem das decisões. Um `DISTINCT` entre entidades de componentes diferentes não é contradição. Isso perde um match que pode ser verdadeiro (a duplicata permanece), e é a escolha conservadora: um falso merge é pior que uma duplicata visível e rastreada.

## Agregação exata, sem inventar confiança

| Parte | Regra |
| --- | --- |
| **Geometria** (`ResolvedGeometry`) | **União das referências** de geometria dos membros (o suporte autoritativo, sem copiar coordenadas), ordenada e única, de um só mapa. As caixas são a união das caixas dos membros, que é **exata**. Referências em mais de um membro são contadas (`duplicate_reference_count`), não duplicadas. O resumo carrega o digest do conjunto e a regra (`union-of-member-geometry-v1`). Não há centroide: `bounds_center_m` é o centro da caixa, uma posição representativa, porque o centroide exato do suporte exigiria as coordenadas. |
| **Semântica** (`ResolvedSemanticState`) | Toda hipótese, atributo e incerteza dos membros, **verbatim**, sem hipótese primária e sem confiança combinada. A ambiguidade do todo nunca é menor que a de um membro e vira `ambiguous` quando os membros propõem labels diferentes; um membro sem hipótese não põe dúvida nos outros. Hipótese e incerteza com a mesma identidade e conteúdo diferente são erro. Um **atributo** com a mesma identidade (nome, valor, origem e derivação) em membros independentes não é conflito: é o caso comum de duas observações do mesmo objeto, e a evidência e o suporte de cada membro são **unidos**, não comparados por igualdade; só uma referência de evidência ou um sinal de suporte que realmente discordem (mesma identidade interna, conteúdo diferente) são erro. Um atributo de conhecimento externo (`AttributeOrigin.EXTERNAL_KNOWLEDGE`) não tem evidência para unir: o `external_source` (`source_id`, `source_version`, `entry_id`) de todos os membros da mesma identidade precisa **concordar**; fontes diferentes são erro, porque o schema não representa mais de uma fonte externa num só atributo e escolher uma descartaria a proveniência da outra. |
| **Tempo** (`EntityTemporalState`) | Recalculado da **união exata** das observações físicas dos membros, cada frame contado uma vez, com `first_seen`/`last_seen` da união. Um frame compartilhado guarda o **maior** número de inferências dos membros (um limite inferior, já que os membros podem contar as mesmas inferências); frames iguais com instantes diferentes, ou domínios de relógio diferentes, são erro. |
| **Evidência** (`EntityEvidenceLinks`) | União dos vínculos dos membros. |

Membros de mapas geométricos ou frames diferentes não podem ser agrupados: seus suportes e caixas não se unem sem alinhamento (`MaterializationError`).

## Linhagem de fusão

Por membro (`ResolvedMember`): a `EntityReference` original (com o `semantic_map_id`) e as decisões `MATCH` que o envolvem (`matched_by`). Por entidade: `resolution_decision_refs` (a união), `unresolved_neighbor_refs`, `contradiction_ids` e a `provenance` (política de materialização e versão do código). O resultado (`ResolvedEntityMaterialization`) traz o conjunto, as contradições e a política, com o fingerprint das regras versionadas.

## Custo

Os `MATCH` de um membro são lidos da adjacência do grafo (custo do grau do membro), e `ResolvedEntitySet.resolve`/`resolved_of` consultam índices por id e por membro construídos no primeiro uso (#599). Medido numa máquina de desenvolvimento, com 10 000 entidades em cadeias de quatro (7 500 `MATCH`, 2 500 `DISTINCT`): materializar levava 14,4 s e passou a 0,9 s; materializar e consultar o dono de cada entidade levava ~66 s no total. `tests/entity_resolution/test_resolution_scaling.py` mantém esse cenário no CI com contadores e um envelope largo.

## O que este módulo não faz

Nenhuma relação espacial, nenhum rastreamento de objetos dinâmicos, nenhuma remoção ou alteração de entidade de origem e nenhuma fusão fora de uma decisão `MATCH` explícita. A detecção de candidatos a divisão (over-merge) é só diagnóstico, opcional e desligada por padrão ([`split-detection.md`](split-detection.md)).
