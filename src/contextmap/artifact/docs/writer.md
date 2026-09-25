# Writer do `ContextMapArtifact`

Este documento descreve `src/contextmap/artifact/serialization/writer.py` (issue #156). O layout e os formatos estão em [`storage-layout.md`](storage-layout.md); as regras gerais de artifact, em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md).

## Uso

```python
manifest = ContextMapArtifactWriter(output_dir=Path("…/context_map")).write(
    context_map,  # ContextMap do schema, com entidades, relações e linhagem
    upstream_locations={  # onde estão agora os artifacts que o mapa cita, por artifact_id
        "corridor-02--map-run-0001": Path("…/geometric_mapping"),
        "entity-resolution--run-0001": Path("…/entity_resolution"),
    },
)
```

`output_dir` é o diretório **final** do artifact e é escolhido por quem chama: o writer não calcula caminho, não aloca índice de run e não mantém `runs.json`. A runtime decide onde o artifact fica (`workspace/<dataset>/<run>/context_map/`).

## Entradas

- `ContextMap`: identidade, `schema_version`, `metadata`, `geometry_ref`, `entities`, `relations` e `lineage`, já validados pelo schema na construção (referências, fechamento de proveniência, capabilities declaradas versus conteúdo). O writer **não repete** essas invariantes: só acrescenta o que o schema não pode saber. `code_version` e `configuration_fingerprint` do manifest vêm de `metadata.creation`: há uma só fonte para a identidade de criação.
- `upstream_locations`: onde estão os artifacts a montante que o mapa cita. A localização de toda dependência **estrutural** é obrigatória; a de evidência opcional não, mas, quando informada, é verificada. A localização serve para ler e verificar o artifact e para calcular uma dica relativa; nunca é gravada como identidade.

## Dependências: exigência e identidade vêm da linhagem

Cada entrada de `context_map.lineage` vira uma dependência no manifest, com o `artifact_type = kind.value`:

- **exigência**: `required` se `ArtifactKind.is_structural` (o mapa geométrico, o run de Entity Resolution e o de Spatial Relations), `optional` nos demais tipos. A classificação é a do schema, sem reinterpretação;
- **identidade**: `content_identity` é a da linhagem. Para uma dependência localizada, o `artifact_digest` do seu diretório (o digest de artifact que os irmãos usam: SHA-256 de `{run_id, schema_version, arquivos [caminho, hash]}`, o mesmo que Spatial Relations grava para o run de Entity Resolution) **tem de ser igual** a ela. Quem monta a linhagem (a runtime) pode reutilizar o valor que cada capability já calcula para o seu artifact. Sem localização (evidência opcional), a identidade da linhagem é gravada sem verificação, e o validador a marca como ausente.

O digest sozinho prova que o diretório achado é o artifact daquele conteúdo; não prova que ele é o artifact que os **registros do mapa** precisam, porque `artifact_id` e `content_identity` são campos independentes de uma entrada de linhagem. Para o run de Entity Resolution e o de Spatial Relations (dependências estruturais, sempre localizadas), o writer também abre cada um com seu leitor público (`EntityResolutionRunReader`, `SpatialRelationsRunReader`) e confere, através de `contextmap.artifact.serialization.structural_dependencies.check_structural_dependencies` — um módulo interno, sem consumidor fora de `serialization`, reaproveitado tal e qual pelo validador `FULL` —: a identidade (`run_id`) de cada run contra o `artifact_id` que a linhagem declara para ele; o mapa geométrico sobre o qual cada run foi construído contra o `geometry_ref` do mapa; que o run de Spatial Relations foi construído sobre o **mesmo** run de Entity Resolution que o mapa também cita (via `validate_resolution` do próprio leitor de Spatial Relations); e que todo `ContextEntity.source` e todo `ContextRelation.source_relation_id` do mapa realmente existe no run correspondente. Um run que não pode nem ser lido para responder uma dessas perguntas (tabela corrompida, por exemplo) é tratado como o mesmo tipo de problema.

Provar que o registro a montante **existe** não prova que a cópia do mapa ainda o **descreve**: `contextmap.artifact.composition` é explícito que o mapa não possui o que uma entidade ou relação é — `source`/`source_relation_id` nomeiam o registro a montante, e os campos locais são só a parte desse resultado que um consumidor lê sem abrir o artifact a montante. Por isso a mesma checagem também compara, contra o próprio `ResolvedEntity`/`Relation` que os leitores já devolveram para a pergunta de existência: para uma entidade, `member_entities`, `resolution_decisions`, `unresolved_neighbors` e `geometry_refs`; para uma relação, os dois extremos (resolvidos pelo `source` das próprias entidades locais), `predicate`, `state` e `uncertainty_kinds`. `semantic_state` e `origin` ficam de fora de propósito: a montagem ainda os deriva à parte, e este módulo não tem como recalculá-los só a partir do registro a montante.

## O que é validado antes de publicar

Nada é escrito no disco se algo abaixo falhar, e nada inválido é descartado em silêncio:

| Verificação | Erro |
| --- | --- |
| `output_dir` já existe (um artifact finalizado é imutável) | `ArtifactExistsError` |
| valor de um registro que JSON não representa (por exemplo `NaN`) | `RecordTableError` |
| localização de um artifact que o mapa não cita | `InvalidContentError` |
| dependência estrutural sem localização | `InvalidContentError` |
| o artifact achado não é o que a linhagem nomeia (`artifact_digest` diferente da `content_identity`) | `InvalidContentError` |
| o mapa geométrico não é o que o mapa declara: identidade, número de pontos ou frame | `InvalidContentError` |
| o run de Entity Resolution ou o de Spatial Relations achado não tem o `run_id` que a linhagem declara, não foi construído sobre o mapa geométrico do mapa, o de Spatial Relations não foi construído sobre o mesmo run de Entity Resolution que o mapa cita, ou um `ContextEntity.source`/`ContextRelation.source_relation_id` do mapa não existe no run correspondente | `InvalidContentError` |
| a cópia local de uma entidade ou relação não descreve mais o registro a montante que ela cita (`member_entities`, `resolution_decisions`, `unresolved_neighbors`, `geometry_refs`, os extremos, `predicate`, `state` ou `uncertainty_kinds` divergem) | `InvalidContentError` |
| artifact a montante ausente, ou diferente do próprio inventário (arquivo ausente, tamanho ou hash) | `UpstreamArtifactError` |

A verificação de hash da geometria lê o payload uma vez; ela é o custo de nunca registrar a identidade de um arquivo que já estava corrompido.

## O que é escrito

Os arquivos de [`storage-layout.md`](storage-layout.md), todos inventariados com tamanho e SHA-256 no `manifest.json`:

- `map-metadata.json` e `geometry/geometry-reference.json`: o `metadata` e o `geometry_ref` do registro canônico do schema (`context_map_to_record`). O `context_map_id` e o `schema_version` ficam no manifest.
- `entities/entities.jsonl`: uma linha por entidade, `{"key", "record"}`, ordenadas pelo id; `record` é o registro canônico da entidade no schema. `relations/relations.jsonl`: `{"key", "subject", "object", "record"}`, com o sujeito e o objeto como ids de entidade.
- `lineage/lineage.json`: `{"upstream_artifacts": […]}`, a linhagem do schema (tipo, identidade de conteúdo, configuração, código e modelos de cada artifact citado).
- `indexes/`: `entity-index.jsonl` e `relation-index.jsonl` (chave → deslocamento e tamanho) e `entity-relation-index.jsonl` (entidade → relações como sujeito e como objeto). São derivados e declarados como tais em `payloads`, com `derived_from`.

Juntos, os quatro documentos e as duas tabelas reconstroem exatamente o registro do schema: `context_map_from_record` os aceita e devolve um mapa igual ao original. Não há `debug/`. O `README.md` é um resumo determinístico, sem horário nem caminho, fora do inventário.

## Determinismo e identidade

O conteúdo depende só do mapa: registros ordenados pelo id, linhas canônicas (ASCII, chaves ordenadas, separadores fixos), listas do manifest em ordem canônica, nenhum UUID, hostname, usuário ou caminho absoluto. O único campo que varia entre duas escritas equivalentes é `written_at` (o horário, informado ou lido do relógio), que **não entra** em `content_identity`; os `locator` das dependências também não. Portanto escrever o mesmo mapa duas vezes, em diretórios diferentes, dá arquivos idênticos (exceto o manifest) e a mesma identidade de conteúdo.

## Publicação atômica

O writer usa `contextmap.shared.AtomicRunDirectory`: os arquivos são escritos num diretório temporário oculto, irmão de `output_dir`; o manifest só é escrito na publicação, depois de conferir o inventário contra o disco; e o diretório é publicado por um único `rename`. Uma falha remove o temporário. Se o processo morrer, sobra um diretório `.tmp-…` **sem `manifest.json`**, que nenhum leitor confunde com um artifact. Depois de publicar, o writer relê o manifest e confere que ele é idêntico ao calculado.

## Limites conhecidos

- O v0 não escreve colunas binárias: nenhum payload denso é necessário enquanto a geometria fica no `GeometricMapArtifact` e o suporte geométrico de cada entidade cabe no próprio registro. O formato de coluna está definido e descrito no manifest para quando um índice denso (por exemplo, suporte geométrico por entidade) for justificado por uma medição.
- Não há política de rejeição declarada: um mapa inconsistente é recusado inteiro.
- Os testes de escrita usam artifacts **reais** de Geometric Mapping, Entity Resolution e Spatial Relations (escritos pelos writers das capabilities numa cena mínima) e um run de evidência simulado. A geometria da cena é sintética (caixas numa fonte em memória, não os pontos do mapa geométrico real), e o estado semântico e a origem de cada registro do mapa vêm dos builders do schema, porque estes testes exercitam o **formato de armazenamento**, não a tradução: a montagem real a partir dos mesmos runs (`contextmap.artifact.assemble_context_map`, [`assembly.md`](assembly.md)) é testada separadamente em `tests/artifact/test_context_map_assembly.py`.
- `tests/artifact/test_context_map_serialization_writer.py` cobre a referência estrutural além do digest com cinco regressões: um `resolved_entity_id` pendurado (`test_an_entity_with_a_dangling_resolved_entity_id_is_refused`), um `source_relation_id` pendurado (`test_a_relation_with_a_dangling_source_relation_id_is_refused`), o `artifact_id` do run de Entity Resolution trocado mantendo o mesmo digest (`test_a_swapped_entity_resolution_artifact_id_with_the_same_digest_is_refused`), o `artifact_id` do run de Spatial Relations trocado do mesmo jeito (`test_a_swapped_spatial_relations_artifact_id_with_the_same_digest_is_refused`) e um run de Spatial Relations ligado a um run de Entity Resolution diferente do que o mapa cita (`test_a_spatial_relations_run_linked_to_a_different_entity_resolution_run_is_refused`); em todas, nada é publicado (`output_dir` não chega a existir). Mais duas regressões cobrem o conteúdo além da existência: `ContextRelation.state` divergente do que o `Relation` a montante decidiu, com `source_relation_id` existente (`test_a_relation_whose_state_no_longer_matches_its_upstream_relation_is_refused`), e uma entidade com `geometry_refs` divergente do `ResolvedEntity` a montante, com `source` existente (`test_an_entity_whose_geometry_refs_no_longer_match_its_resolved_entity_is_refused`).
