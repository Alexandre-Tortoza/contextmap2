# Leitor do `ContextMapArtifact`

Este documento descreve `src/contextmap/artifact/serialization/reader.py`, `directory.py` e `dependencies.py` (issue #157). O layout está em [`storage-layout.md`](storage-layout.md) e a escrita em [`writer.md`](writer.md).

O leitor abre um artifact **só pelo próprio diretório** e responde o que um consumidor precisa para entender o mapa. Ele é um leitor de dados, não um motor de consulta: não faz busca, linguagem natural, planejamento, navegação, inferência de relações nem resolução de entidades. Só instalação base (Python e NumPy): importar `contextmap.artifact.reader` não importa `torch`, `transformers`, `rclpy`, `rosbags`, `cv2` nem `PIL`, e há um teste que garante isso.

## Uso

```python
with ContextMapArtifactReader.open(path) as reader:
    reader.metadata()  # ContextMapMetadata
    reader.map_bounds()  # Bounds3D no frame do mapa
    reader.entity("entity-a")  # EntityEntry, sem ler as outras
    reader.relations_for("entity-a")  # relações em que participa, por chave
    reader.geometry(geometry_reference)  # GeometryPoint autoritativo
    reader.validate_reference(geometry_reference)
```

| Operação | O que faz |
| --- | --- |
| `open(path, dependency_paths=None, verify_hashes=False)` | Lê o manifest e confere versões, identidade e o inventário (presença e tamanho). Não lê registro nem geometria. |
| `manifest`, `context_map()`, `metadata()`, `map_bounds()` | O schema reconstruído dos arquivos, com a decodificação estrita do próprio schema. |
| `entity_keys()`, `entity(key)`, `entities()` | Uma entidade por chave, pelo índice de deslocamentos; ou todas, em fluxo, uma linha por vez. |
| `relation_keys()`, `relation(key)`, `relations()` | Idem para relações. |
| `relations_for(entity_key)` | Travessia pelo índice derivado: relações em que a entidade é sujeito ou objeto. Não interpreta a relação. |
| `geometry_source()`, `geometry(reference)` | A geometria pelo `GeometricMapArtifactReader` (mapeada em memória, não lida). |
| `validate_reference(reference)` | Confere que a referência é deste mapa, é a identidade canônica de um elemento e está dentro do número de pontos. Não abre a geometria. |
| `dependency_location(tipo, id)` | Localiza e verifica uma dependência registrada (por exemplo, evidência opcional). |
| `close()` | Libera o mapeamento da geometria; depois, o leitor recusa ler. |

## Carregamento preguiçoso

Abrir um artifact custa um `stat` por arquivo. As tabelas só são abertas no primeiro acesso, e abrir uma tabela lê apenas o **índice** (proporcional ao número de registros) e o confere contra o tamanho do payload; um registro é lido, e só então interpretado, quando é pedido. A geometria só é aberta na primeira chamada de `geometry_source()` ou `geometry()`, como `mmap`; nada é carregado por inteiro.

## Referências e dependências

A geometria não está no artifact: o leitor a resolve pelo `GeometricMapArtifactReader`, e o `GeometrySource` que ele devolve continua sendo o dono das coordenadas.

Para achar uma dependência, o leitor usa **só** dois lugares, nesta ordem: o caminho informado em `dependency_paths[artifact_id]` (e, se houver, apenas ele: nunca cai para a dica) ou a dica relativa `locator` do manifest, resolvida contra o diretório do artifact. Não há busca. O que for achado só conta se o digest do seu inventário for igual ao `content_identity` registrado:

| Resultado | Erro |
| --- | --- |
| não há onde procurar, ou não há diretório com manifest no lugar | `MissingDependencyError` |
| há um diretório, mas não é o artifact registrado (digest diferente: velho ou de outro mapa) | `DependencyMismatchError` |

Com `verify_hashes=True`, o leitor também confere o hash de todos os arquivos do mapa geométrico ao abri-lo (uma leitura sequencial do payload). Sem isso, ele confia no digest do inventário e na conferência de tamanho da própria geometria. Um artifact movido **sem** as dependências continua abrindo, e tudo que não precisa de geometria funciona; resolver geometria pede o caminho da dependência.

## Erros explícitos

Nada é lido parcialmente:

| Situação | Erro |
| --- | --- |
| não é um diretório, ou não tem `manifest.json` (uma escrita interrompida nunca tem) | `IncompleteContextMapArtifactError` |
| `manifest.json` ilegível, com campo ausente ou desconhecido | `ManifestError` |
| versão de formato não suportada (a mensagem lista as suportadas) | `UnsupportedFormatVersionError` |
| versão de schema ilegível (verificada antes de qualquer outro campo) | `UnsupportedArtifactSchemaError`, que também é a `UnsupportedSchemaVersionError` do schema |
| manifest editado (identidade de conteúdo não confere) | `ArtifactIntegrityError` |
| arquivo do inventário ausente, ou um arquivo obrigatório que o inventário não lista | `MissingPayloadError` / `IncompleteContextMapArtifactError` |
| tamanho diferente (truncado ou alterado) ou, com `verify_hashes`, hash diferente | `ArtifactIntegrityError` |
| índice desordenado, com lacuna, com contagem errada ou que aponta para outro registro | `BrokenIndexError` |
| entidade, relação ou referência que não resolve | `RecordNotFoundError` / `UnresolvedReferenceError` |

## Sem mutação e sem fallback

O leitor só abre arquivos para leitura: não escreve índice, cache nem lock, e funciona sobre um diretório somente leitura. Um teste compara o conteúdo e o `mtime` de cada arquivo (do artifact e do mapa geométrico) antes e depois de ler tudo. Não existe `debug/` para o leitor: se `entities/entities.jsonl` some, o artifact é inválido mesmo que haja uma cópia em `debug/`. Um arquivo que o manifest não lista nunca é confiado.

## Limites

Entidades e relações são devolvidas como `EntityEntry`/`RelationEntry` (chave, extremos e o registro canônico opaco); quando o schema (#150) as tipar, o leitor as decodifica com ele. A conferência de hash de um artifact e de referências entre registros pertence ao validador ([`integrity-validation.md`](integrity-validation.md)).
