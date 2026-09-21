# Writer do `ContextMapArtifact`

Este documento descreve `src/contextmap/artifact/serialization/writer.py` (issue #156). O layout e os formatos estão em [`storage-layout.md`](storage-layout.md); as regras gerais de artifact, em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md).

## Uso

```python
manifest = ContextMapArtifactWriter(output_dir=Path("…/context_map")).write(
    context_map,  # ContextMap do schema
    geometry_dir=Path("…"),  # GeometricMapArtifact que o mapa referencia
    entities=[EntityEntry(...)],
    relations=[RelationEntry(...)],
    evidence=[UpstreamArtifact(...)],  # dependências opcionais (ou obrigatórias) de evidência
)
```

`output_dir` é o diretório **final** do artifact e é escolhido por quem chama: o writer não calcula caminho, não aloca índice de run e não mantém `runs.json`. A runtime decide onde o artifact fica (`workspace/<dataset>/<run>/context_map/`).

## Entradas

- `ContextMap`: identidade, `schema_version`, `metadata` e `geometry_ref`, já validados pelo schema. `code_version` e `configuration_fingerprint` do manifest vêm de `metadata.creation`: há uma só fonte para a identidade de criação.
- `EntityEntry(key, record)` e `RelationEntry(key, subject_key, object_key, record)`: o `record` é o registro canônico (compatível com JSON) da entidade ou da relação, guardado **como veio**; o artifact não o interpreta. `key` é a identidade dentro do mapa, e o writer nunca a reescreve. Enquanto o schema ainda não carrega entidades e relações (issue #150), elas entram por estes registros.
- `UpstreamArtifact(artifact_type, artifact_id, location, requirement)`: artifact a montante referenciado, nunca copiado.

## O que é validado antes de publicar

Nada é escrito no disco se algo abaixo falhar, e nada inválido é descartado em silêncio:

| Verificação | Erro |
| --- | --- |
| `output_dir` já existe (um artifact finalizado é imutável) | `ArtifactExistsError` |
| chave duplicada, chave vazia, valor que JSON não representa (por exemplo `NaN`) | `RecordTableError` |
| relação cujo sujeito ou objeto não é uma entidade do mapa | `InvalidContentError` |
| conteúdo que o mapa não declara (entidades sem a capability `entities`, relações sem `relations`): uma capability nunca é implícita | `InvalidContentError` |
| o `geometry_dir` não é o mapa geométrico que o `geometry_ref` nomeia (`map_id`), tem outro número de pontos ou outro frame que o do mapa | `InvalidContentError` |
| geometria ou evidência ausente, ou diferente do próprio inventário (arquivo ausente, tamanho ou hash) | `UpstreamArtifactError` |
| evidência repetida, ou um mapa geométrico passado como evidência | `InvalidContentError` |

Uma capability **declarada** pode estar vazia: "nenhuma relação foi produzida" é diferente de "relações nunca foram calculadas", e o artifact preserva a distinção. A verificação de hash da geometria lê o payload uma vez; ela é o custo de nunca registrar a identidade de um arquivo que já estava corrompido.

## O que é escrito

Os arquivos de [`storage-layout.md`](storage-layout.md), todos inventariados com tamanho e SHA-256 no `manifest.json`:

- `map-metadata.json` e `geometry/geometry-reference.json`: o `metadata` e o `geometry_ref` do registro canônico do schema (`context_map_to_record`). O `context_map_id` e o `schema_version` ficam no manifest; juntos reconstroem o registro completo.
- `entities/entities.jsonl`: uma linha por entidade, `{"key", "record"}`, ordenadas por chave. `relations/relations.jsonl`: `{"key", "subject", "object", "record"}`.
- `indexes/`: `entity-index.jsonl` e `relation-index.jsonl` (chave → deslocamento e tamanho) e `entity-relation-index.jsonl` (entidade → relações como sujeito e como objeto). São derivados e declarados como tais em `payloads`, com `derived_from`.
- `lineage/lineage.json`: as identidades exatas dos artifacts a montante (tipo, id, digest do inventário, obrigatório ou opcional), sem caminhos. A proveniência semântica de entidades e relações (`EvidenceOrigin`, `DerivationKind`) pertence ao schema (issue #152) e entra aqui quando existir.

Não há `debug/`. O `README.md` é um resumo determinístico, sem horário nem caminho, fora do inventário.

## Determinismo e identidade

O conteúdo depende só das entradas: registros ordenados por chave, linhas canônicas (ASCII, chaves ordenadas, separadores fixos), listas do manifest em ordem canônica, nenhum UUID, hostname, usuário ou caminho absoluto. O único campo que varia entre duas escritas equivalentes é `written_at` (o horário, informado ou lido do relógio), que **não entra** em `content_identity`; os `locator` das dependências também não. Portanto escrever o mesmo mapa duas vezes, em diretórios diferentes, dá arquivos idênticos (exceto o manifest) e a mesma identidade de conteúdo. A ordem em que entidades e relações chegam não importa.

## Publicação atômica

O writer usa `contextmap.shared.AtomicRunDirectory`: os arquivos são escritos num diretório temporário oculto, irmão de `output_dir`; o manifest só é escrito na publicação, depois de conferir o inventário contra o disco; e o diretório é publicado por um único `rename`. Uma falha remove o temporário. Se o processo morrer, sobra um diretório `.tmp-…` **sem `manifest.json`**, que nenhum leitor confunde com um artifact. Depois de publicar, o writer relê o manifest e confere que ele é idêntico ao calculado.

## Limites conhecidos

- Entidades e relações entram como registros opacos com chave e extremos; quando o schema (#150) as incorporar ao `ContextMap`, o writer as extrai dele.
- O v0 não escreve colunas binárias: nenhum payload denso é necessário enquanto a geometria fica no `GeometricMapArtifact`. O formato de coluna está definido e descrito no manifest para quando um índice denso (por exemplo, suporte geométrico por entidade) for justificado.
- Não há política de rejeição declarada: um mapa inconsistente é recusado inteiro.
