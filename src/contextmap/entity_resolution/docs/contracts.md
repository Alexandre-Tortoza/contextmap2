# Contratos de Entity Resolution

Referência de campos e invariantes dos contratos públicos de `contextmap.entity_resolution`. Um contrato só entra aqui quando existe no código.

## Princípios

- **Sem score único.** Nenhum contrato oferece um número que resuma os canais. Distância em metros, cosseno entre embeddings visuais e relação entre labels não se somam, não se fazem média e não recebem um limiar conjunto sem uma regra documentada e testável.
- **Evidência, decisão e conhecimento separados.** `EntityMatchEvidence` é evidência de uma comparação; `ResolutionDecision` é o veredito de uma política versionada sobre essa evidência; a entidade resolvida ([`resolved-entities.md`](resolved-entities.md)) é a estrutura derivada das decisões `MATCH`.
- **Ausente não é zero.** Um canal que não pôde comparar o par é `unavailable`, com o motivo, e nunca vira zero nem voto por `DISTINCT`.
- **`UNRESOLVED` é resultado válido.** Nunca é forçado a match nem a não-match.
- **Nada muta uma entidade de origem.** Contratos só guardam `EntityReference`.

## Identidade

| Tipo | Escopo | Significado |
| --- | --- | --- |
| `EntityResolutionRunId` | global | Identidade de um artifact de resolução imutável; é o escopo dos ids resolvidos. |
| `ResolvedEntityId` | um artifact de resolução | Identidade de uma entidade resolvida, local ao artifact que a alocou. |
| `ComparisonId` | derivado do par | `comparison--<sha256[:16]>` do par `(entity_a_ref, entity_b_ref)`; a mesma comparação tem sempre o mesmo id. |
| `ResolutionDecisionId` | derivado | `decision--<sha256[:16]>` da comparação e da política (id e fingerprint): outra política ou configuração dá outro id. |

`PolicyRef(policy_id, configuration_fingerprint)` identifica, em qualquer canal, na recuperação de candidatos, na política de resolução e na materialização, a regra versionada (`policy_id`) e os parâmetros efetivos de uma execução (`configuration_fingerprint`), que nunca se confundem.

`ResolvedEntityId` é um `NewType` sobre `str`: o tipo distingue o id de uma entidade resolvida de um `EntityId` de origem em tempo de checagem, mas a identidade completa é sempre a referência.

## `ResolvedEntityReference`

| Campo | Tipo | Descrição |
| --- | --- | --- |
| `resolution_run_id` | `EntityResolutionRunId` | O artifact de resolução que possui a entidade resolvida. |
| `resolved_entity_id` | `ResolvedEntityId` | A entidade resolvida, local a esse artifact. |

Invariantes:

- as duas identidades são obrigatórias e não podem ser vazias nem só espaços;
- a referência é imutável e hashable, então serve de chave;
- o mesmo `resolved_entity_id` em dois artifacts é **duas** referências diferentes: uma nova execução nunca herda identidade;
- a referência **não** contém `semantic_map_id`: ela não nomeia uma entidade de origem, e as entidades de origem continuam endereçáveis por `EntityReference`.

## O par comparado

Toda comparação tem `entity_a_ref` e `entity_b_ref` em **ordem canônica**: `entity_a_ref` ordena estritamente antes por `(semantic_map_id, entity_id)`. Assim a comparação tem uma identidade só, independente de quem a pediu primeiro, e dois registros equivalentes codificam para os mesmos bytes. Uma entidade nunca é comparada com ela mesma. Todas as medições `_a`/`_b` se referem a esse par ordenado.

## `EntityMatchEvidence`

| Campo | Descrição |
| --- | --- |
| `comparison_id` | Derivado do par; um id adulterado é recusado. |
| `entity_a_ref`, `entity_b_ref` | O par, em ordem canônica. |
| `gates` | Resultados dos gates de validade, ordenados por id e únicos. |
| `provenance` | `MatchEvidenceProvenance`: identidade versionada dos gates e versão do código. |
| `geometry`, `semantic`, `appearance`, `temporal`, `point_representation` | A evidência de cada canal. |

Um canal `None` **não foi avaliado** (por exemplo, a política o desabilitou). Um canal avaliado que não pôde comparar o par está presente e `UNAVAILABLE`. As duas situações nunca se confundem, e nenhuma delas é evidência contra.

A evidência responde, sem reexecutar nada:

| Pergunta | Onde |
| --- | --- |
| Por que foram fundidas? | `signals()` e a decisão `MATCH` que aponta para esta evidência. |
| Que evidência apoiou? | `supporting_channels()`, `signals()` com status `supporting`. |
| Que evidência argumentou contra? | `conflicting_channels()`, `signals()` com status `conflicting`. |
| Que gate impediu a comparação? | `blocked`, `failed_gates()`. |

Invariantes: um gate reprovado **bloqueia** a comparação, e uma comparação bloqueada só pode conter canais `UNAVAILABLE` com motivo `blocked_by_gate`. Cada slot só aceita a evidência do próprio canal.

### Gates

`evaluate_comparison_gates` (`entity-comparison-gates-v1`) avalia três gates duros: `distinct-entities` (uma entidade nunca é comparada com ela mesma), `same-geometric-map` e `same-map-frame`. Coordenadas de mapas ou frames diferentes nunca são comparadas sem um alinhamento explícito, então o gate reprovado nomeia os valores comparados.

## `EntityCandidateSet`

Os alvos plausíveis de comparação de uma entidade, produzidos **antes** de qualquer evidência cara. É recuperação, não decisão: não tem `decision`, `match` nem `score`. Ver [`candidate-retrieval.md`](candidate-retrieval.md) para os campos, a política e o índice.

## Evidência de canal

Todo canal (`ChannelEvidence`) tem:

| Campo | Descrição |
| --- | --- |
| `policy` | `PolicyRef`: política versionada e fingerprint da configuração sob a qual foi medido. |
| `measurement` | A medição própria do canal; `None` quando indisponível. |
| `findings` | O resultado de cada regra explícita aplicada à medição; vazio quando indisponível. |
| `unavailable` | `Unavailability(reason, detail)`; `None` quando medido. |

Um canal é **medido ou indisponível, nunca os dois e nunca nenhum**. Um canal indisponível não tem medição nem achados.

`status` deriva dos achados: um conflito domina, depois o apoio, senão é neutro; `UNAVAILABLE` só com `unavailable`. Um conflito dentro do canal **não é diluído** pelos achados que apoiam, então um canal materialmente conflitante chega à política como tal. `Finding(rule_id, status, detail, metric, observed, threshold)` registra a regra, o valor observado e o limiar aplicado; um achado nunca é `unavailable`.

| Motivo (`UnavailableReason`) | Significado |
| --- | --- |
| `missing_evidence` | Uma das entidades não tem o que o canal usaria. |
| `incompatible_domain` | Espaços de embedding, de representação ou domínios de relógio diferentes: comparar seria sem sentido. |
| `blocked_by_gate` | Um gate duro reprovou; a comparação nem foi tentada. |
| `insufficient_evidence` | Há evidência, mas pouca demais para a política medir. |

### Geometria (`GeometryEvidence`)

Calculada por `compare_geometry` ([`geometry-comparison.md`](geometry-comparison.md)): as definições exatas das métricas, as regras versionadas e os casos tratados.

`GeometryMeasurement` guarda, no frame comum do mapa: distância entre centroides, distância entre as caixas (`0` se sobrepõem), IoU e fração de sobreposição das caixas (`None` quando a caixa é plana, o que **não** é sobreposição zero), contagens de suporte, suporte compartilhado e Jaccard (coerentes entre si), razão de extensão, estatísticas de distância entre suportes (opcionais, quando a geometria foi resolvida), ângulo entre eixos principais (opcional) e as ressalvas dos resumos das entidades (`a:sparse_support`, ...).

### Semântica (`SemanticEvidence`)

`SemanticMeasurement` mantém **todas** as hipóteses em vez de um label: o estado de ambiguidade e o número de hipóteses de cada lado, a relação de cada par de labels (`same`, `refinement`, `related`, `different`, com a regra versionada) e as comparações de atributos (nome, os dois valores, a origem `observed` ou `derived` de cada um e se são o mesmo; conhecimento externo nunca é comparado). `different` não é prova de objetos diferentes. Calculada por `compare_semantics` ([detalhes](semantic-temporal-comparison.md)).

### Aparência (`AppearanceEvidence`)

Calculada por `AppearanceComparator` ([detalhes](appearance-comparison.md)).

`AppearanceMeasurement` guarda o espaço de embedding (fingerprint) em que todas as features vivem, a métrica, a política de agregação, a similaridade e a faixa das similaridades entre pares de observações, e as `FeatureContribution` de cada lado: as features agrupadas **por observação física**. Várias features do mesmo frame (inferência repetida) ficam juntas e contam como uma observação. Uma feature de outro espaço é recusada.

### Temporal (`TemporalEvidence`)

Calculada por `compare_temporal` ([detalhes](semantic-temporal-comparison.md)).

`TemporalMeasurement` guarda o domínio de relógio, a sobreposição e a lacuna entre os intervalos (nunca as duas), as observações físicas e os resultados de inferência de cada lado **contados à parte**, e as observações físicas compartilhadas e a união, que precisam fechar (`união = a + b - compartilhadas`).

### Representação 3D (`PointRepresentationEvidence`)

Calculada por `RepresentationComparator` ([detalhes](representation-comparison.md)).

Canal opcional. `RepresentationMeasurement` é o análogo de aparência para um espaço de representação: o fingerprint do espaço, a métrica, a política de agregação, a similaridade e as `RepresentationRef` de cada lado (com o `representation_space_id` e a geometria em que estão ancoradas), a `dimension` do espaço e os `compared_components` sobre os quais a similaridade foi calculada (componentes indefinidos ficam fora, nunca são lidos como zero). Uma representação de outro espaço é recusada.

## `ResolutionDecision`

| Campo | Descrição |
| --- | --- |
| `decision_id` | Derivado da comparação e da política. |
| `entity_a_ref`, `entity_b_ref` | O par, em ordem canônica. |
| `decision` | `ResolutionOutcome`: `match`, `distinct` ou `unresolved`. |
| `evidence_ref` | O `ComparisonId` da evidência de onde a decisão saiu; precisa ser o do par. |
| `policy` | `PolicyRef(policy_id, configuration_fingerprint)`. |
| `triggered_rules` | As regras que dispararam, na ordem de avaliação: `TriggeredRule(stage, rule_id, outcome, detail, channels)`. |
| `channels_used`, `channels_ignored` | Canais em que a decisão se apoia e canais disponíveis mas ignorados, em ordem canônica; nunca em ambos. |
| `unresolved_reason` | `comparison_blocked`, `insufficient_evidence` ou `conflicting_evidence`; presente **exatamente** quando `unresolved`. |
| `provenance` | `DecisionProvenance(code_version)`. |

Invariantes: toda decisão precisa de uma regra do estágio `decision` que conclua o mesmo resultado (é o que a torna reconstruível), e uma decisão `match` ou `distinct` precisa se apoiar em pelo menos um canal. Os estágios de uma política são `gate`, `eligibility`, `evidence`, `rule` e `decision`.

Uma decisão vale para referências de entidades sob um contexto de resolução explícito: não implica identidade permanente entre mapas reconstruídos de forma independente, não muta nenhuma entidade e não é um merge.

## Codec

`encode_*` produz um `dict` só com primitivos JSON; `decode_*` reconstrói pelo construtor, então **toda invariante é revalidada**. O decode é estrito: campo ausente, campo desconhecido, valor do tipo errado e membro de enum desconhecido são recusados com o caminho do valor (`ResolutionDecision.policy`, `...measurement.centroid_distance_m`). A implementação é reflexiva sobre as anotações dos dataclasses (`_codec.py`), então um contrato novo não pode divergir do próprio codec.

- `encode_resolved_entity_reference`, `decode_resolved_entity_reference`
- `encode_match_evidence`, `decode_match_evidence`
- `encode_resolution_decision`, `decode_resolution_decision`
- `encode_candidate_set`, `decode_candidate_set`
