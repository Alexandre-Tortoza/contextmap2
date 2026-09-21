# Contratos do ContextMap

Este documento descreve `src/contextmap/artifact/models.py`, `metadata.py`, `frame.py`, `versioning.py` e `records.py`.

## `ContextMap`

```text
ContextMap
├── context_map_id
├── schema_version
├── metadata
└── geometry_ref
```

| Campo | Significado |
| --- | --- |
| `context_map_id` | identidade do artifact imutável; única por mapa, **nunca um caminho** |
| `schema_version` | versão da **semântica dos dados** sob a qual o mapa foi escrito (`MAJOR.MINOR.PATCH`) |
| `metadata` | o que o mapa é, de onde veio e como foi criado |
| `geometry_ref` | o `GeometricMapArtifact` dono da geometria autoritativa |

Um mapa é imutável (`frozen`), comparável por valor e não carrega objeto de backend, tensor, tipo de ROS nem caminho de arquivo.

### Identidade

`context_map_id` identifica um artifact imutável. Reexecutar o pipeline produz **outro** mapa com **outra** identidade; um mapa finalizado nunca é reescrito. O valor precisa estar presente e não pode conter separador de caminho (`/` ou `\`): a identidade semântica não depende de onde o artifact foi gravado, e o layout em disco é decisão do serializador.

## `ContextMapMetadata`

| Campo | Significado |
| --- | --- |
| `creation` | `MapCreation`: como o mapa foi montado |
| `source_sequences` | as sequências e seleções (`SourceSequence`) usadas, ordenadas e únicas; pelo menos uma |
| `frame` | `MapFrame`: frame, unidade, lateralidade, direção "para cima" e âncora da origem |
| `bounds` | `Bounds3D` da extensão espacial, no frame do mapa |
| `time_bounds` | `ObservationWindow` das observações usadas, em um único relógio |
| `capabilities` | `DeclaredCapabilities`: o conteúdo opcional que o mapa declara |

`MapCreation` registra a `assembly_policy` (`PolicyRef`, regra versionada que compôs o mapa), a `code_version` e o `configuration_fingerprint`. Os dois últimos são `None` **explícito** quando desconhecidos, nunca omitidos e nunca vazios. Não há data de criação: ela tornaria equivalentes não idênticos e pertence ao diário do run, não ao schema.

`SourceSequence` é `(sequence_artifact_id, selection_id)`: a sequência canônica imutável e a identidade determinística da parte dela que foi usada. A mesma sequência pode aparecer com seleções distintas.

Frame, unidades, âncora, extensão e capacidades estão em [`metadata.md`](metadata.md).

## `GeometricMapLink`

A geometria é **referenciada**, nunca embutida. O contrato decide isso por três razões: as coordenadas autoritativas já vivem no `GeometricMapArtifact` imutável; copiá-las criaria uma segunda fonte de verdade para milhões de pontos; e um empacotamento que inclua a geometria (bundle) é decisão de transporte do serializador, sem mudar a semântica do schema.

| Campo | Significado |
| --- | --- |
| `map_id` | identidade do mapa geométrico imutável (`contextmap.geometric_mapping.MapId`) |
| `point_count` | número de elementos de geometria nele; permite validar o intervalo de uma referência sem abrir a geometria |

## Versão do schema

`schema_version` descreve a **semântica dos dados**, não a versão do pacote Python nem a de um serializador; as três mudam de forma independente. Construir um `ContextMap` com uma versão malformada ou ilegível levanta `UnsupportedSchemaVersionError` antes de qualquer outra validação, e nada é lido parcialmente.

A regra de leitura, `SchemaVersion.is_readable_by(reader)`, é deliberadamente estreita: durante a fase de validação (`MAJOR == 0`, ver [`docs/versioning.md`](../../../../docs/versioning.md)) só a mesma `MAJOR.MINOR` é legível, porque uma mudança de `MINOR` pode ser incompatível; a partir de `1.0.0`, qualquer versão do mesmo `MAJOR` é legível.

## Visão canônica em registros

`context_map_to_record()` escreve o mapa como valores compatíveis com JSON (mapeamentos, listas, textos, números, booleanos e `None`), com uma entrada por campo do contrato e nomeada exatamente como ele; enums viram seus valores. `context_map_from_record()` faz o caminho inverso e é **estrita**:

- a `schema_version` é verificada antes de qualquer outro campo;
- um campo ausente, um campo desconhecido ou um valor de tipo errado é `ContextMapRecordError` com o caminho do problema;
- um valor que viola uma invariante do contrato levanta o erro do próprio contrato;
- nada é preenchido por padrão, ignorado ou reparado.

Os registros **não definem arquivo, formato nem layout**. O serializador escolhe como registros chegam ao disco; os testes de schema usam registros para provar o round-trip sem depender de nenhum serializador.
