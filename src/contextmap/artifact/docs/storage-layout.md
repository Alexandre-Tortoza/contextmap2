# Layout e formatos de armazenamento do `ContextMapArtifact`

Este documento decide **como** o `ContextMap` é gravado em disco (issue #155) e por quê. Ele descreve `src/contextmap/artifact/serialization/layout.py` e `manifest.py`. O que o mapa **significa** (frame, unidades, entidades, relações, linhagem) pertence ao schema e não depende dos nomes de arquivo abaixo: um artifact declara o que contém no `manifest.json` e no `map-metadata.json`, nunca pela presença de um arquivo.

As regras gerais de artifacts (imutabilidade, escrita atômica, inventário com hash) estão em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md) e são implementadas uma vez em `contextmap.shared.run_directory`. Aqui ficam só as decisões do `ContextMapArtifact`.

## Diretório é o artifact

O **diretório** é a forma canônica e a única com schema. Um arquivo compactado (`tar`, `zip`) é somente um *wrapper de transporte* do mesmo diretório: quem recebe o arquivo o extrai e o valida como qualquer outro diretório. Não existe um segundo formato "empacotado" com regras próprias, e o v0 não implementa o empacotamento (nenhum consumidor o exige; o modo de exportação (#159) produz um diretório).

O nome do diretório é escolhido por quem chama (a runtime o coloca em `workspace/<dataset>/<run>/context_map/`) e **nunca é identidade**: a identidade está no manifest, e o mesmo artifact copiado para outro caminho continua sendo o mesmo artifact.

## Layout v0

```text
<diretório do artifact>/
├── manifest.json                    # identidade, versões, payloads, dependências e inventário SHA-256
├── README.md                        # resumo humano; não contratual e fora do inventário
├── map-metadata.json                # ContextMapMetadata: frame, unidades, âncora, limites, capabilities
├── geometry/
│   └── geometry-reference.json      # GeometricMapLink: identidade e tamanho do GeometricMapArtifact; nunca uma cópia
├── entities/
│   └── entities.jsonl               # uma ContextEntity por linha, ordenadas por id
├── relations/
│   └── relations.jsonl              # uma ContextRelation por linha, ordenadas por id
├── indexes/                         # tudo aqui é derivado e reconstruível
│   ├── entity-index.jsonl           # identidade → deslocamento e tamanho em entities.jsonl
│   ├── relation-index.jsonl         # identidade → deslocamento e tamanho em relations.jsonl
│   ├── entity-relation-index.jsonl  # entidade → relações em que é sujeito e objeto
│   └── <coluna>.u32|.u64|.f32|.f64  # colunas numéricas cruas, descritas em manifest.payloads
└── lineage/
    └── lineage.json                 # fechamento de proveniência: artifacts e políticas de origem
```

- Os oito arquivos JSON/JSONL listados em `layout.CONTRACTUAL_FILES` existem **sempre**, mesmo vazios. Uma tabela de relações vazia significa "nenhuma relação foi produzida"; um arquivo ausente significa dano. Os dois nunca se confundem.
- `map-metadata.json` e `lineage/lineage.json` são a serialização dos tipos do schema (`ContextMapMetadata` e a linhagem/proveniência); `geometry/geometry-reference.json` é a do `geometry_ref` do `ContextMap`. Este documento fixa **onde** ficam, não o que significam.
- Não há `debug/`. Evidência de debug é para humanos, não faz parte do contrato do mapa e **nunca** é um lugar onde o leitor procura dados autoritativos: o manifest recusa qualquer caminho sob `debug/` (`require_contractual_path`), o writer nunca o cria e o leitor nunca o abre.
- Não há diretório de evidência opcional. Uma evidência opcional é uma **dependência** no manifest (`requirement = optional`): ela é referenciada por identidade e conteúdo, nunca copiada para dentro do mapa.
- O `README.md` é escrito por `AtomicRunDirectory.publish`, é determinístico (sem carimbo de tempo) e não entra no inventário, então apagá-lo não invalida o artifact.

## Formatos escolhidos

| Conteúdo | Formato | Por quê |
| --- | --- | --- |
| Manifest, metadados, referência de geometria, linhagem | JSON UTF-8, chaves ordenadas, recuo de 2 espaços, LF | Inspecionável com `cat`/`jq`; biblioteca padrão; determinístico e diffável. |
| Entidades e relações | JSON Lines, uma por linha, ordenadas por identidade | Escrita e hash em fluxo; leitura de **um** registro por deslocamento; sem parser binário; igual aos artifacts irmãos (`entities.jsonl` de Semantic Mapping). |
| Índices de registro | JSON Lines (`id`, `offset`, `length`) | Tamanho proporcional ao número de registros, não à geometria; inspecionável; reconstruível dos registros. |
| Arrays numéricos densos | Binário cru little-endian, ordem C, sem cabeçalho, descrito no manifest | `numpy.memmap`, `memoryview` ou `struct` leem sem parser; acesso aleatório por aritmética; hash em fluxo; não há dtype `object`, então não há pickle. |
| Geometria | **Não é copiada**: referência ao `GeometricMapArtifact` | O `geometry.bin` de um mapa tem 64 bytes por ponto (10 milhões de pontos são 640 MB); duplicá-lo em cada consumidor violaria a política de não duplicação de `ARTIFACTS.md`. |

Nenhum formato exige mais que a instalação base (Python e NumPy); o leitor e o validador funcionam sem ROS, Torch, SAM, DINO, VLM, FAST-LIO ou qualquer runtime de modelo.

### Alternativas rejeitadas

- **Parquet, Arrow, Feather.** Exigem `pyarrow`, que não está na instalação base, e são binários opacos para `cat`. As tabelas de entidades e relações têm de dezenas a milhares de linhas: o ganho colunar não compensa a dependência.
- **HDF5, Zarr, NetCDF.** Dependências pesadas (`h5py`, `zarr`), semântica de bloqueio e de metadados internos que impede garantir arquivos idênticos byte a byte.
- **SQLite.** Está na biblioteca padrão, mas é um arquivo binário opaco, mistura dados e consulta (convida busca e consulta semântica, fora da fronteira do leitor) e seus arquivos de journal/WAL quebram a cópia atômica e o determinismo.
- **`.npy` e `.npz`.** Fazem do NumPy o *formato* público. O cabeçalho aceita o dtype `object`, que é pickle (execução de código); `.npz` é um zip e não permite acesso aleatório sem descompactar.
- **MessagePack, CBOR, Protobuf.** Dependência extra e não inspecionáveis com ferramentas comuns.
- **`pickle`, `torch.save`.** Proibidos como formato contratual: executam código na leitura e amarram o arquivo à versão da biblioteca.
- **Compressão por arquivo (gzip, zstd).** Destrói deslocamentos, `mmap` e o hash em fluxo simples. Compressão é assunto do transporte (um `tar.gz` do diretório), não do schema.

## Descrição dos payloads

Um payload binário ou tabular sem descrição não é legível. Por isso `manifest.json` traz `payloads`, uma entrada por payload, com o que uma pessoa precisa para lê-lo sem o código:

| Campo | JSON Lines (`"encoding": "jsonl"`) | Coluna crua (`"encoding": "raw-le"`) |
| --- | --- | --- |
| `path`, `role`, `semantics` | sim | sim |
| `record_count` | sim | não |
| `dtype`, `byte_order`, `shape` | não | sim |
| `unit`, `frame_id` | não | sim (`null` quando os valores não têm unidade nem frame, como índices) |
| `derived_from` | caminhos dos payloads de origem, se for índice | idem |

- `role` é `authoritative` (o dado em si) ou `derived_index` (reconstruível de outros payloads). Um índice corrompido nunca redefine o dado para o qual aponta: ele é detectado reconstruindo-o a partir de `derived_from`.
- `dtype` pertence a um vocabulário fechado: `uint8`, `int32`, `uint32`, `int64`, `uint64`, `float32`, `float64`. Toda coluna é little-endian e em ordem C. O tamanho esperado do arquivo é `prod(shape) * largura(dtype)`, então um payload truncado é detectado sem lê-lo.
- Uma dimensão zero é uma tabela vazia explícita.

## Identidade e versões

Estes campos respondem a perguntas diferentes e nunca se confundem:

| Campo | O que identifica | Entra em `content_identity` |
| --- | --- | --- |
| `context_map_id` | o mapa, como o schema o define | sim |
| `content_identity` | os **bytes contratuais** do artifact | (é o resultado) |
| `format_version` | o layout e as codificações em disco (esta página) | sim |
| `schema_version` | a **semântica** dos dados (o schema) | sim |
| `code_version`, `configuration_fingerprint` | quem escreveu e com que configuração | sim |
| `written_at` | quando foi escrito | **não** |
| `dependencies[].locator` | onde procurar uma dependência | **não** |

`content_identity` é o SHA-256 do manifest sem `written_at` e sem os `locator`, com todas as listas em ordem canônica. Como o manifest carrega o inventário com o hash de cada arquivo, a identidade é uma função pura do conteúdo contratual: escrever o mesmo mapa duas vezes dá a mesma identidade, mover o artifact (ou suas dependências) não a muda, e qualquer alteração de arquivo, descritor, contagem ou dependência a muda. O carimbo de tempo e o caminho descrevem *quando* e *onde*, não *o quê*.

`format_version` e `schema_version` andam separados: mudar como o mapa é gravado não muda o que ele significa, e vice-versa. O leitor aceita um conjunto **explícito** de `format_version` (`layout.SUPPORTED_FORMAT_VERSIONS`); qualquer outro é recusado com uma mensagem que lista as versões aceitas, nunca lido parcialmente. As regras de compatibilidade do `schema_version` pertencem ao schema.

## Registros

Cada linha de `entities.jsonl` é `{"key", "record"}` (a chave é o id da entidade e `record` é o registro canônico do schema); cada linha de `relations.jsonl` é `{"key", "subject", "object", "record"}` (o sujeito e o objeto são ids de entidade). Esse envelope existe para que os índices e a integridade entre registros sejam verificáveis sem interpretar o registro. Os documentos `map-metadata.json`, `geometry/geometry-reference.json` e `lineage/lineage.json` são partes do `context_map_to_record` do schema; juntos, os documentos e as duas tabelas reconstroem exatamente o registro do mapa.

## Dependências

`dependencies` lista os artifacts a montante que o mapa **referencia em vez de copiar**, e é derivada da linhagem do schema (`ContextMap.lineage`): um por artifact citado, com `artifact_type` (o `kind` da linhagem), `artifact_id`, `content_identity` (a da linhagem), `requirement` e um `locator` opcional.

- `requirement = required` para os tipos **estruturais** do schema (`ArtifactKind.is_structural`: o mapa geométrico, o run de Entity Resolution e o de Spatial Relations): necessários para resolver o mapa. Ausente ou divergente, o artifact é inutilizável.
- `requirement = optional` para os demais (sequência, percepção, fusão…): necessários só para inspecionar evidência em profundidade. Ausente, é um aviso: o núcleo do mapa continua totalmente legível.
- `content_identity` de um artifact a montante é, por convenção, o `inventory_digest` do inventário do seu manifest (SHA-256 dos `(caminho, tamanho, hash)` ordenados). Ele torna a referência **exata**: um artifact achado por qualquer caminho só é a dependência se o digest bater. O writer confere essa igualdade para todo artifact localizado, e nenhum caminho absoluto é gravado.
- `locator` é uma **dica**, relativa ao diretório do artifact (por exemplo `../geometric_mapping`). Nunca é confiada nem entra na identidade, e é a única coisa que a exportação reescreve. Quem abre o artifact pode informar o caminho de cada dependência explicitamente; o `locator` só serve quando o workspace foi movido inteiro. Uma dependência sem localização na escrita não tem dica.

A geometria é resolvida pelo `GeometricMapArtifactReader` existente (o `GeometrySource` que ele devolve continua sendo o único dono das coordenadas); o `ContextMapArtifact` guarda só a referência e a verificação.

## Determinismo

O writer produz o mesmo conteúdo para as mesmas entradas:

- registros ordenados por identidade, uma linha canônica cada (`sort_keys`, separadores fixos);
- floats no `repr` mais curto que faz o ciclo completo (o padrão de `json`);
- listas do manifest (payloads, dependências, inventário) em ordem canônica;
- nenhum UUID, hostname, usuário ou caminho absoluto no conteúdo contratual;
- o único conteúdo que varia entre duas escritas equivalentes é `written_at`, fora da identidade.

"Equivalente" é comparado por **estado semântico** (o mapa decodificado e a identidade), não por bytes do diretório inteiro: o contrato é o mapa, não a serialização.

## Portabilidade

Um diretório de artifact é copiável entre máquinas como qualquer árvore de arquivos, sem rosbag, sem workspace de origem e sem runtime de modelo. Nenhum caminho absoluto é gravado; os únicos caminhos relativos que saem do diretório são os `locator`, que são dicas. Depois de mover o artifact sem as dependências, ele ainda abre e a leitura de tudo que não precisa da geometria funciona; resolver geometria exige informar onde a dependência está, e a divergência de identidade é um erro explícito.

## Fora do contrato

Debug, checkpoints, tensores nativos de framework, pickles, rosbags e qualquer objeto de runtime nunca entram no artifact. O bundle de exportação (#159) também não os inclui por padrão.

## Compatibilidade e fixtures

O `format_version` (layout) e o `schema_version` (semântica) são independentes, e o leitor aceita um conjunto **explícito** de cada um:

- **Formato**: só as versões de `layout.SUPPORTED_FORMAT_VERSIONS` (hoje `0.1.0`); qualquer outra, inclusive um prefixo, um pré-lançamento ou vazio, é `UnsupportedFormatVersionError` com a lista das aceitas.
- **Schema**: o que `require_supported_schema_version` aceita (o mesmo `MAJOR.MINOR` enquanto o major é 0; um patch novo do mesmo minor é legível). Uma versão ilegível é rejeitada **antes** de qualquer registro ser interpretado (`UnsupportedArtifactSchemaError`, que também é a `UnsupportedSchemaVersionError` do schema).

Para cada versão prometida como legível pela v0.1.0 há uma **fixture** versionada em `tests/fixtures/context_map_artifact/<versão>/`, gravada uma vez pelo writer e nunca regenerada: `v0.1.0` é um artifact completo (formato `0.1.0`, schema `0.1.0`, três entidades, duas relações, um estado ambíguo e um não resolvido). Um teste fixa a `content_identity` da fixture, então regenerá-la por engano falha, e outros testes garantem que ela continua abrindo com hash verificado, que os registros são iguais aos do schema e que o validador completo só aponta o que a fixture não carrega (as dependências a montante, que são grandes e não versionadas). Quando uma versão de formato ou de schema nova for prometida como legível, acrescenta-se a sua fixture ao lado, sem tirar a anterior; quando uma deixar de ser legível, a mensagem de rejeição é a que a fixture antiga passa a exercitar.

## Estado desta decisão

O v0 fixa o layout, o manifest e a descrição de payloads. Escrita, leitura, validação e exportação são detalhadas em documentos próprios, acrescentados junto de cada issue (#156 a #159), e a validação de ida e volta, corrupção, portabilidade e versões está em `tests/artifact/test_context_map_serialization_roundtrip.py` (#160).
