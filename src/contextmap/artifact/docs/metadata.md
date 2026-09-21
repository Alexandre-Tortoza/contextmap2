# Metadados: frame, unidades, âncora, extensão e capacidades

Este documento descreve `src/contextmap/artifact/frame.py` e `metadata.py`. Um mapa só é portável se um consumidor consegue interpretar suas coordenadas, sua origem, suas unidades, seu escopo temporal e quais conteúdos opcionais ele realmente traz, **sem abrir os internos de State Estimation e sem deduzir nada do nome de um dataset**.

```text
ContextMapMetadata
├── creation             # MapCreation
├── source_sequences[]   # SourceSequence
├── frame                # MapFrame: frame_id, unit, handedness, up_direction, anchor
├── bounds               # Bounds3D, no frame do mapa
├── time_bounds          # ObservationWindow
└── capabilities         # DeclaredCapabilities
```

Nenhum metadado ausente é omitido: o que não se sabe é `None` **explícito**, e um campo desconhecido no registro é um erro.

## `MapFrame`

| Campo | Significado |
| --- | --- |
| `frame_id` | frame do mapa; é o frame da geometria referenciada, e o `ContextMap` não aplica transformação adicional |
| `unit` | unidade de toda coordenada (`LengthUnit.METER`) |
| `handedness` | `RIGHT_HANDED` ou `LEFT_HANDED`, nunca inferida |
| `up_direction` | vetor unitário, no frame do mapa, que aponta para longe da gravidade; `None` quando desconhecido |
| `anchor` | como a origem é definida (`MapAnchor`) |

`up_direction` é um **vetor**, não um eixo nomeado: o frame local de um estimador não garante que `z` aponte para cima (ele nasce da pose inicial do run). Um vetor não unitário, não finito ou nulo é rejeitado.

## Origem: local de estimador versus ancorada externamente

| `AnchorKind` | Significado | `reference_frame_id` |
| --- | --- | --- |
| `ESTIMATOR_LOCAL` | a origem é definida por onde um run de estimador começou; as coordenadas só têm significado dentro deste artifact | deve ser `None` |
| `EXTERNALLY_ANCHORED` | a origem está amarrada a um frame externo nomeado, por um alinhamento explícito | obrigatório |

`origin_definition` descreve, em texto, como a origem é definida e, para uma âncora externa, como o alinhamento foi estabelecido.

### Comparabilidade entre mapas

`MapFrame.is_comparable_with(other)` implementa a regra "nunca implicar comparabilidade quando não existe alinhamento": só é `True` quando **os dois** frames são ancorados externamente à **mesma** referência e concordam em unidade e lateralidade. Dois frames locais de estimador não são comparáveis nem quando ambos se chamam `map`, e um frame local nunca é comparável a um ancorado.

## Extensão espacial e temporal

- `bounds` é um `Bounds3D` (de `geometric_mapping`) **no frame do mapa**; bounds em outro frame são rejeitados, porque o frame nunca é inferido;
- `time_bounds` é um `ObservationWindow`: intervalo fechado de aquisição das observações usadas, em **um** domínio de relógio (`SourceTimestamp.clock_id`). Dois relógios distintos nunca são comparados, e o fim não pode preceder o início;
- `source_sequences` lista as sequências e seleções de origem, ordenadas e únicas.

`ObservationWindow` repete a regra de `state_estimation.TimeBounds` em vez de importá-la: `artifact` só pode depender de `shared`, `geometric_mapping`, `semantic_mapping`, `entity_resolution` e `spatial_relations` (`tests/architecture/test_boundaries.py`). Se um terceiro consumidor precisar do mesmo intervalo, o caminho é promovê-lo a `contextmap.shared`, não ampliar essa fronteira.

## Capacidades declaradas

`DeclaredCapabilities` torna o conteúdo opcional **descobrível pelos metadados**, sem implicá-lo pelo nome de arquivo ou pelo layout.

| `MapCapability` | Declarada quando |
| --- | --- |
| `GEOMETRY` | sempre: um mapa sem geometria referenciada não é um `ContextMap` |
| `ENTITIES` | Entity Resolution contribuiu para o mapa |
| `RELATIONS` | Spatial Relations contribuiu para o mapa; exige `ENTITIES` |
| `POINT_REPRESENTATION_EVIDENCE` | há referências a evidência de Point Representation |

Duas distinções importam:

- **presente e vazia é diferente de ausente.** `RELATIONS` declarada com nenhuma relação significa que o estágio rodou e nada encontrou; `RELATIONS` não declarada significa que relações nunca foram calculadas. Um consumidor não pode tratar os dois casos como iguais;
- `relation_predicates` lista os tipos de relação presentes (ordenados e únicos) e só existe quando `RELATIONS` é declarada.

A declaração é canônica (ordenada por valor e única). A checagem da declaração contra o conteúdo real do mapa (entidades, relações e proveniência) depende dos contratos de composição e de linhagem e é introduzida com eles.

## Rejeição por validação de schema

| Situação | Erro |
| --- | --- |
| `frame_id` vazio | `ValueError` (`frame_id`) |
| `up_direction` nulo, não unitário ou não finito | `ValueError` (`up_direction`) |
| âncora local com referência externa, ou externa sem referência | `ValueError` (`reference_frame_id`) |
| origem sem definição | `ValueError` (`origin_definition`) |
| bounds em outro frame | `ValueError` |
| janela temporal com relógios distintos ou fim antes do início | `ValueError` |
| capacidades desordenadas, repetidas, sem geometria, com relações sem entidades ou predicados sem relações | `ValueError` |
| unidade ou lateralidade desconhecida em um registro | `ContextMapRecordError` |
