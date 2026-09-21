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
- Os testes usam artifacts **reais** de Geometric Mapping, Entity Resolution e Spatial Relations (escritos pelos writers das capabilities numa cena mínima) e um run de evidência simulado. A geometria da cena é sintética (caixas numa fonte em memória, não os pontos do mapa geométrico real), e o estado semântico e a origem de cada registro do mapa, que a montagem ainda inexistente derivaria, vêm dos builders do schema. Nenhuma validação com dados reais.
