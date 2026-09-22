# Validação do `ContextMapArtifact`

Este documento descreve `src/contextmap/artifact/serialization/validation.py` (issue #158). Ele é independente da runtime de mapeamento e de qualquer runtime de modelo ou ROS (há um teste de importação em subprocesso), lê só os arquivos que recebe e **nunca repara nada**.

```python
report = validate_context_map_artifact(path, level=ValidationLevel.FULL, dependency_paths=None)
report.status  # invalid | structurally_valid | verified
report.to_json()  # relatório determinístico legível por máquina
```

O validador **não levanta exceção** para um artifact danificado: todo dano vira um achado (`Finding`) no relatório.

## Níveis e o que cada resposta significa

| Nível | O que lê | Melhor resposta |
| --- | --- | --- |
| `STRUCTURAL` | manifest e versões, um `stat` por arquivo, descritores de payload, estrutura dos índices, documentos pequenos e onde estão as dependências | `structurally_valid` |
| `FULL` | tudo acima, mais o hash de cada arquivo, cada registro, as referências entre registros, os índices reconstruídos e os arquivos das dependências | `verified` |

Um artifact **nunca é chamado de válido sem ter sido verificado**: com o nível estrutural a resposta é `structurally_valid`, que diz explicitamente que conteúdo, referências e arquivos a montante não foram lidos, e o relatório lista cada verificação pulada com o motivo. `verified` significa que toda verificação que este validador implementa rodou e passou; as que ele não pôde executar aparecem como `skipped`, com o motivo (no nível completo, nenhuma é pulada num artifact íntegro).

## Verificações

Em ordem fixa; cada uma termina `passed`, `failed` ou `skipped` (com `detail`).

| Verificação | O que confere | Códigos de erro |
| --- | --- | --- |
| `manifest` | legível, formato e schema suportados, identidade de conteúdo recomputada | `artifact.incomplete`, `manifest.malformed`, `manifest.unsupported_format_version`, `manifest.unsupported_schema_version`, `manifest.identity_mismatch` |
| `inventory` | todo arquivo obrigatório inventariado; cada arquivo presente com o tamanho registrado | `inventory.required_file_missing`, `file.missing`, `file.size_mismatch` |
| `file_hashes` (FULL) | SHA-256 de cada arquivo | `file.hash_mismatch` |
| `payload_descriptors` | os cinco payloads descritos, com papel, origem e contagem esperados | `payload.descriptor_missing`, `payload.descriptor_mismatch`, `payload.count_mismatch` |
| `index_structure` | os índices abrem: ordenados, contíguos, contagem certa, cobrindo exatamente o payload | `index.broken` |
| `documents` | os metadados e a referência de geometria são válidos pelo schema | `map.invalid` |
| `capabilities` | contagens de entidades e relações só com a capability declarada (uma declarada pode estar vazia) | `capabilities.undeclared_entities`, `capabilities.undeclared_relations` |
| `lineage` | a linhagem do schema em `lineage.json` concorda com as dependências do manifest (tipo, id, identidade de conteúdo e exigência pela `ArtifactKind.is_structural`) | `lineage.malformed`, `lineage.mismatch` |
| `dependencies` | cada dependência é achada e é a registrada | `dependency.required_missing`, `dependency.mismatch` (avisos: `dependency.optional_missing`) |
| `geometry_consistency` | o mapa geométrico é o que o mapa nomeia: identidade, número de pontos e frame | `geometry.reference_mismatch`, `geometry.map_id_mismatch`, `geometry.point_count_mismatch`, `geometry.frame_mismatch` |
| `reference_integrity` (FULL) | lê cada entidade e relação; a chave de cada linha é o id do seu registro; toda relação aponta para entidades existentes | `reference.relation_endpoint_missing`, `reference.key_mismatch`, `index.broken` |
| `index_rebuild` (FULL) | cada índice é byte a byte o que os registros produzem (identidade do índice reconstruível, chaves únicas e ordenadas) | `index.mismatch`, `records.invalid` |
| `schema_invariants` (FULL) | os registros guardados formam um `ContextMap` válido pelo schema: referências de geometria dentro do mapa, fechamento de proveniência, capabilities e linhagem coerentes | `map.invalid` |
| `structural_references` (FULL) | abre o run de Entity Resolution e o de Spatial Relations achados (identidade da dependência já confirmada por `dependencies`) com seus próprios leitores e confere o que só eles sabem: a identidade (`run_id`) de cada um contra o que a linhagem do mapa declara, o mapa geométrico sobre o qual cada um foi construído contra o `geometry_ref` do próprio mapa, que o run de Spatial Relations foi construído sobre o **mesmo** run de Entity Resolution que o mapa também cita, que todo `ContextEntity.source` e todo `ContextRelation.source_relation_id` do mapa realmente existe no run correspondente, e que os campos locais de cada entidade e relação (`member_entities`, `resolution_decisions`, `unresolved_neighbors`, `geometry_refs`; os dois extremos, `predicate`, `state` e `uncertainty_kinds`) ainda descrevem o `ResolvedEntity`/`Relation` achado — `semantic_state` e `origin` ficam de fora | `dependency.structural_reference_invalid` |
| `dependency_integrity` (FULL) | os arquivos de cada dependência batem com o próprio inventário | `dependency.upstream_damaged` |
| `unlisted_files` | arquivos fora do inventário | avisos `file.unlisted`, `debug.present` |

Se o manifest não pode ser lido, as demais verificações ficam `skipped` ("o manifest não pôde ser lido"). Uma verificação que só falha porque outra já reportou o dano (por exemplo, tabelas de registro quando o arquivo sumiu) fica `skipped`, sem erro em cascata.

### Referência estrutural além do digest

`dependencies` prova que o diretório achado é exatamente o artifact do digest gravado (`artifact_digest` bate com `content_identity`). Isso não prova que ele é o artifact que os **registros do próprio mapa** precisam: `artifact_id` e `content_identity` são campos independentes, então uma entrada de linhagem pode declarar um `artifact_id` que não é o do artifact que o digest realmente aponta, e nada no digest sozinho percebe que um run de Spatial Relations foi produzido sobre um run de Entity Resolution diferente daquele de onde as entidades do mapa realmente vêm, ou que um `resolved_entity_id`/`source_relation_id` citado não existe a montante. `structural_references` fecha essa lacuna abrindo os dois runs estruturais achados com seus leitores públicos (`EntityResolutionRunReader`, `SpatialRelationsRunReader`, inclusive `validate_resolution`) — a mesma verificação tipada que o writer roda antes de publicar (`contextmap.artifact.serialization.structural_dependencies`, módulo interno sem consumidor fora de `serialization`), então uma dependência estrutural pendurada ou trocada é recusada da mesma forma nos dois lugares.

Existir não é o mesmo que **ainda descrever**: provar que `resolved_entity_id`/`source_relation_id` existe a montante não prova que a cópia do mapa continua sendo o que aquele registro diz. `contextmap.artifact.composition` é explícito que o mapa não possui o que uma entidade ou relação é — os campos locais são só a parte do resultado a montante que um consumidor lê sem abrir o artifact que o produziu. `structural_references` por isso também compara, contra o próprio `ResolvedEntity`/`Relation` que os leitores já devolveram para provar a existência: `member_entities`, `resolution_decisions`, `unresolved_neighbors` e `geometry_refs` de uma entidade; os dois extremos de uma relação (resolvidos pelo `source` das entidades locais), `predicate`, `state` e `uncertainty_kinds`. `semantic_state` e `origin` ficam de fora de propósito, porque a montagem ainda os deriva à parte.

### Obrigatório versus opcional

Uma dependência **obrigatória** ausente é erro (`dependency.required_missing`); uma **opcional** ausente é aviso (`dependency.optional_missing`) e não invalida o artifact, porque o núcleo do mapa continua legível sem ela. Uma dependência **presente mas que não é a registrada** (digest diferente: velha ou de outro mapa) é sempre erro, obrigatória ou não: algo que finge ser a evidência é pior que a ausência.

## O relatório

`ValidationReport` (`to_record()` / `to_json()`) traz: `validator_version`, `level`, `status`, as identidades (`artifact_type`, `format_version`, `schema_version`, `context_map_id`, `content_identity`), `checks` (todas, na ordem fixa), `files` (o inventário conferido, por caminho, com `status`: `verified`, `size_ok`, `missing`, `size_mismatch` ou `hash_mismatch`), `dependencies` (cada uma com `requirement` e `status` `found`/`missing`/`mismatch`) e `findings`.

O relatório é **determinístico e inspecionável**: sem horário, sem caminho absoluto e sem dado da máquina (o assunto de um achado é um caminho relativo, uma chave ou um id de artifact), com chaves ordenadas, achados em ordem (erros antes de avisos, depois por código e assunto) e sem achados repetidos. Validar duas vezes, ou validar uma cópia do artifact, dá os mesmos bytes.

## Sem reparo e sem fallback

O validador só lê. Um teste compara o conteúdo e o `mtime` de todo arquivo antes e depois de validar. Um arquivo fora do inventário nunca é lido (`debug/` inclusive): vira aviso. Nada é reconstruído a partir de `debug/`.

## Validação

`tests/artifact/test_context_map_serialization_validation.py` valida artifacts reais (com o `GeometricMapArtifact` real): artifact intacto, cada nível, alteração que mantém o tamanho (só o nível completo vê), arquivo ausente e truncado, índice quebrado, índice que aponta para outro registro, relação com entidade inexistente, índice de travessia velho, linhagem incompatível, geometria diferente da nomeada, conteúdo não declarado, dependências obrigatória e opcional ausentes, dependência que não é a registrada, artifact movido, arquivos a montante danificados, versões não suportadas, manifest adulterado, diretório incompleto, arquivos fora do contrato, determinismo do relatório e ausência de mutação. Também cobre a referência estrutural além do digest: `resolved_entity_id` pendurado e `source_relation_id` pendurado, cada um provando que `STRUCTURAL` não vê o problema (`STRUCTURALLY_VALID`) e só `FULL` reporta `dependency.structural_reference_invalid` (`test_a_dangling_resolved_entity_id_is_found_only_by_full_verification`, `test_a_dangling_source_relation_id_is_found_only_by_full_verification`); `artifact_id` do run de Entity Resolution trocado mantendo o mesmo digest, provando que a dependência ainda é `found` e mesmo assim o mapa é `INVALID` (`test_a_swapped_entity_resolution_artifact_id_with_a_matching_digest_is_detected`); e um run de Spatial Relations ligado a um run de Entity Resolution diferente do que o mapa cita (`test_a_spatial_relations_run_linked_to_a_different_entity_resolution_run_is_detected`). Mais duas regressões cobrem o conteúdo além da existência, reescrevendo a tabela envolvida diretamente com o próprio codificador do writer para manter o artifact autoconsistente: `ContextRelation.state` divergente do que o `Relation` a montante decidiu, com `source_relation_id` existente (`test_a_relation_whose_state_no_longer_matches_its_upstream_relation_is_detected`), e uma entidade com `geometry_refs` divergente do `ResolvedEntity` a montante, com `source` existente (`test_an_entity_whose_geometry_refs_no_longer_match_its_resolved_entity_is_detected`).
