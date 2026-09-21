# Contratos de Entity Resolution

Referência de campos e invariantes dos contratos públicos de `contextmap.entity_resolution`. Um contrato só entra aqui quando existe no código.

## Identidade

| Tipo | Escopo | Significado |
| --- | --- | --- |
| `EntityResolutionRunId` | global | Identidade de um artifact de resolução imutável; é o escopo dos ids resolvidos. |
| `ResolvedEntityId` | um artifact de resolução | Identidade de uma entidade resolvida, local ao artifact que a alocou. |

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

## Codec

`encode_resolved_entity_reference` produz um `dict` só com primitivos JSON (`resolution_run_id`, `resolved_entity_id`). `decode_resolved_entity_reference` reconstrói a referência pelo construtor, então a invariante de identidades obrigatórias é revalidada; um campo ausente ou vazio é recusado com `ValueError` que nomeia o campo.
