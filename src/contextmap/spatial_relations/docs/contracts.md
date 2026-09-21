# Contratos de Spatial Relations

Referência dos contratos públicos de `contextmap.spatial_relations`. Todos são `dataclass(frozen=True)`, validam suas invariantes na construção e referenciam entidades e geometria por identidade: nada aqui copia uma entidade resolvida nem coordenadas.

## Ideias que moldam os contratos

1. **Relação não é medição.** `Relation` aponta para `RelationEvidence`; as medições, os limiares e a geometria vivem na evidência e nunca são reescritos pela decisão.
2. **Entre entidades resolvidas.** Sujeito e objeto são `ResolvedEntityReference`, nunca pixels, regiões, pontos ou candidatos não resolvidos. As entidades resolvidas permanecem imutáveis.
3. **Desconhecido não é negativo.** `UNRESOLVED` (não sei) é diferente de `REJECTED` (a evidência contradiz), e `AMBIGUOUS`/`UNAVAILABLE` (na evidência) são diferentes de `CONFLICTS`.
4. **Canais separados.** A evidência geométrica por limites e a de contato por pontos são registros distintos, cada um com o seu status; nenhum score único os funde.
5. **Identidade com escopo.** `RelationId` e `RelationEvidenceId` são locais ao artifact de relações; sujeito e objeto de uma relação vêm sempre do **mesmo** artifact de resolução.

## `Relation`

| Campo | Significado |
| --- | --- |
| `relation_id` | Identidade, local ao artifact; derivada de `(sujeito, predicado, objeto)` por `relation_id_for()`. |
| `subject_entity_ref` | `ResolvedEntityReference` da entidade de que a sentença trata. |
| `predicate` | `RelationPredicate` canônico; a direção importa (`sujeito PREDICADO objeto`). |
| `object_entity_ref` | `ResolvedEntityReference` da entidade de referência. |
| `state` | `RelationState`: o que a evidência decidiu. |
| `relation_evidence_refs` | Evidência em que a decisão se apoiou, ordenada e única. |
| `uncertainty` | Por que a relação está `UNRESOLVED`; vazio nos demais estados. |
| `derived_from` | A relação de que esta foi gerada como inversa ou gêmea simétrica; `None` quando foi avaliada diretamente. |
| `provenance` | `RelationProvenance`: versão da taxonomia, política de decisão, fingerprint da configuração e versão do código. |

### Estados

| Estado | Significado | Invariantes |
| --- | --- | --- |
| `SUPPORTED` | A evidência sustenta o predicado e nada o contradiz. | Tem evidência; sem incerteza. |
| `REJECTED` | A evidência contradiz o predicado e nada o sustenta. | Tem evidência; sem incerteza. |
| `UNRESOLVED` | A evidência falta, é ambígua ou contraditória. | Tem ao menos um registro de incerteza; pode não ter evidência. |

Além disso, o contrato recusa uma relação de uma entidade consigo mesma, sujeito e objeto de artifacts de resolução diferentes, referências de evidência fora de ordem canônica, incerteza que cita evidência que a relação não usa e uma relação derivada dela mesma.

### `RelationUncertainty`

`kind` (`INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE` ou `INCONSISTENT_STRUCTURE`), `detail` (explicação determinística) e `evidence_refs` (a evidência que a produziu, ordenada e única). Nenhum tipo de incerteza vira número: a incerteza é uma razão declarada, não um score.

## `RelationEvidence`

O que **um canal** mediu sobre **um candidato dirigido**.

| Campo | Significado |
| --- | --- |
| `evidence_id` | Identidade, local ao artifact; derivada de `(canal, sujeito, predicado, objeto)` por `evidence_id_for()`. |
| `channel` | `RelationEvidenceChannel`: `GEOMETRY` (limites), `CONTACT` (pontos) ou `OBSERVATION` (afirmações upstream, corroborante; ver [observation-evidence.md](observation-evidence.md)). |
| `subject_entity_ref`, `predicate`, `object_entity_ref` | O candidato. O predicado é sempre o sentido **avaliado**: o inverso de um predicado derivado (`BELOW`, `BEHIND`, `CONTAINS`) é gerado, nunca medido. |
| `status` | `RelationEvidenceStatus`. |
| `measurements` | `Quantity` (nome, valor finito, unidade) medidas, ordenadas por nome e únicas. |
| `thresholds` | Os limiares e tolerâncias com que foram comparadas, ordenados por nome e únicos. |
| `geometry` | `MeasuredGeometry`: sobre qual geometria se mediu, ordenada por papel. |
| `caveats` | `EvidenceCaveat`: por que o registro não é decisivo. |
| `statements` | Só no canal `OBSERVATION`: as `ObservationRelationStatement` em que o registro se apoia, ordenadas; os canais medidos as recusam, e o canal de observação recusa `geometry`. |
| `provenance` | `RelationEvidenceProvenance`: regra versionada, fingerprint da configuração, versão da taxonomia, frame e mapa geométrico das coordenadas, fingerprint dos eixos declarados e versão do código. |

### Status

| Status | Significado |
| --- | --- |
| `SUPPORTS` | As medições cumprem as condições do predicado. |
| `CONFLICTS` | As medições falham claramente uma condição do predicado. |
| `AMBIGUOUS` | As medições caem numa faixa de tolerância, ou a geometria não é confiável o bastante para decidir. |
| `UNAVAILABLE` | A evidência não pôde ser calculada. |

Invariantes: `AMBIGUOUS` e `UNAVAILABLE` **precisam** de uma ressalva que diga por quê; `SUPPORTS` e `CONFLICTS` precisam das medições que os sustentam; todo registro traz a geometria medida e o frame e o mapa geométrico das coordenadas. A ausência de evidência **não** é um registro: um canal que não avaliou o candidato simplesmente não aparece.

### `EvidenceCaveat`

`kind` (`WITHIN_TOLERANCE`, `UNRELIABLE_GEOMETRY`, `DEGENERATE_GEOMETRY`, `MISSING_INPUT` ou `CONFLICTING_STATEMENTS`) e `detail` (explicação determinística).

### `MeasuredGeometry`

`role` (por exemplo `subject`, `object` ou `subject_contact_band`), `point_count`, `geometry_digest` (digest da identidade exata dos elementos de geometria) e, opcionalmente, `geometry_refs` (as referências, quando o conjunto é pequeno o bastante para listá-las). A evidência nunca copia coordenadas: com o digest e o mapa, o conjunto exato é recuperável.

## Identidade

`relation_id_for()` e `evidence_id_for()` são funções puras de `(sujeito, predicado, objeto)` (mais o canal, no caso da evidência): o mesmo candidato dá sempre o mesmo id e o sentido oposto dá outro. O digest cobre a **referência completa** das duas entidades, então os mesmos ids locais em dois artifacts de resolução nunca colidem.

## Serialização

`encode_relation`/`decode_relation` e `encode_relation_evidence`/`decode_relation_evidence` produzem apenas primitivos JSON, então um registro persistido abre sem ROS, NumPy ou runtime de modelo. A decodificação reconstrói os contratos pelos construtores: toda invariante é revalidada e um registro adulterado (por exemplo, `SUPPORTED` sem evidência) é recusado em vez de aceito. Um campo ausente é nomeado no erro.
