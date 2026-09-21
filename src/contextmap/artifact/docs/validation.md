# Validação do `ContextMapArtifact`

Este documento descreve `src/contextmap/artifact/validation.py` (issue #158). Ele é independente da runtime de mapeamento e de qualquer runtime de modelo ou ROS (há um teste de importação em subprocesso), lê só os arquivos que recebe e **nunca repara nada**.

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

Um artifact **nunca é chamado de válido sem ter sido verificado**: com o nível estrutural a resposta é `structurally_valid`, que diz explicitamente que conteúdo, referências e arquivos a montante não foram lidos, e o relatório lista cada verificação pulada com o motivo. `verified` significa que toda verificação que este validador implementa rodou e passou; as que ele ainda não sabe fazer aparecem como `skipped` (hoje só `entity_geometry_support`: o suporte geométrico de cada entidade só pode ser conferido quando o schema tipar as entidades, #150).

## Verificações

Em ordem fixa; cada uma termina `passed`, `failed` ou `skipped` (com `detail`).

| Verificação | O que confere | Códigos de erro |
| --- | --- | --- |
| `manifest` | legível, formato e schema suportados, identidade de conteúdo recomputada | `artifact.incomplete`, `manifest.malformed`, `manifest.unsupported_format_version`, `manifest.unsupported_schema_version`, `manifest.identity_mismatch` |
| `inventory` | todo arquivo obrigatório inventariado; cada arquivo presente com o tamanho registrado | `inventory.required_file_missing`, `file.missing`, `file.size_mismatch` |
| `file_hashes` (FULL) | SHA-256 de cada arquivo | `file.hash_mismatch` |
| `payload_descriptors` | os cinco payloads descritos, com papel, origem e contagem esperados | `payload.descriptor_missing`, `payload.descriptor_mismatch`, `payload.count_mismatch` |
| `index_structure` | os índices abrem: ordenados, contíguos, contagem certa, cobrindo exatamente o payload | `index.broken` |
| `map_record` | metadados e referência de geometria formam um mapa válido pelo schema | `map.invalid` |
| `capabilities` | conteúdo presente só se declarado (uma capability declarada pode estar vazia) | `capabilities.undeclared_entities`, `capabilities.undeclared_relations` |
| `lineage` | `lineage.json` lista exatamente as dependências do manifest | `lineage.malformed`, `lineage.mismatch` |
| `dependencies` | cada dependência é achada e é a registrada | `dependency.required_missing`, `dependency.mismatch` (avisos: `dependency.optional_missing`) |
| `geometry_consistency` | o mapa geométrico é o que o mapa nomeia: identidade, número de pontos e frame | `geometry.reference_mismatch`, `geometry.map_id_mismatch`, `geometry.point_count_mismatch`, `geometry.frame_mismatch` |
| `reference_integrity` (FULL) | lê cada entidade e relação; toda relação aponta para entidades existentes | `reference.relation_endpoint_missing`, `index.broken` |
| `index_rebuild` (FULL) | cada índice é byte a byte o que os registros produzem (identidade do índice reconstruível, chaves únicas e ordenadas) | `index.mismatch`, `records.invalid` |
| `dependency_integrity` (FULL) | os arquivos de cada dependência batem com o próprio inventário | `dependency.upstream_damaged` |
| `entity_geometry_support` | (pulada; ver acima) | |
| `unlisted_files` | arquivos fora do inventário | avisos `file.unlisted`, `debug.present` |

Se o manifest não pode ser lido, as demais verificações ficam `skipped` ("o manifest não pôde ser lido"). Uma verificação que só falha porque outra já reportou o dano (por exemplo, tabelas de registro quando o arquivo sumiu) fica `skipped`, sem erro em cascata.

### Obrigatório versus opcional

Uma dependência **obrigatória** ausente é erro (`dependency.required_missing`); uma **opcional** ausente é aviso (`dependency.optional_missing`) e não invalida o artifact, porque o núcleo do mapa continua legível sem ela. Uma dependência **presente mas que não é a registrada** (digest diferente: velha ou de outro mapa) é sempre erro, obrigatória ou não: algo que finge ser a evidência é pior que a ausência.

## O relatório

`ValidationReport` (`to_record()` / `to_json()`) traz: `validator_version`, `level`, `status`, as identidades (`artifact_type`, `format_version`, `schema_version`, `context_map_id`, `content_identity`), `checks` (todas, na ordem fixa), `files` (o inventário conferido, por caminho, com `status`: `verified`, `size_ok`, `missing`, `size_mismatch` ou `hash_mismatch`), `dependencies` (cada uma com `requirement` e `status` `found`/`missing`/`mismatch`) e `findings`.

O relatório é **determinístico e inspecionável**: sem horário, sem caminho absoluto e sem dado da máquina (o assunto de um achado é um caminho relativo, uma chave ou um id de artifact), com chaves ordenadas, achados em ordem (erros antes de avisos, depois por código e assunto) e sem achados repetidos. Validar duas vezes, ou validar uma cópia do artifact, dá os mesmos bytes.

## Sem reparo e sem fallback

O validador só lê. Um teste compara o conteúdo e o `mtime` de todo arquivo antes e depois de validar. Um arquivo fora do inventário nunca é lido (`debug/` inclusive): vira aviso. Nada é reconstruído a partir de `debug/`.

## Validação

`tests/artifact/test_context_map_serialization_validation.py` valida artifacts reais (com o `GeometricMapArtifact` real): artifact intacto, cada nível, alteração que mantém o tamanho (só o nível completo vê), arquivo ausente e truncado, índice quebrado, índice que aponta para outro registro, relação com entidade inexistente, índice de travessia velho, linhagem incompatível, geometria diferente da nomeada, conteúdo não declarado, dependências obrigatória e opcional ausentes, dependência que não é a registrada, artifact movido, arquivos a montante danificados, versões não suportadas, manifest adulterado, diretório incompleto, arquivos fora do contrato, determinismo do relatório e ausência de mutação.
