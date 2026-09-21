# Anotações de referência

O pipeline é avaliado em níveis independentes, então o reference set carrega **uma família de anotação por nível**, cada uma com schema versionado. A implementação está em `contextmap.evaluation.annotations`; o manifesto que as referencia está em [`reference-set.md`](reference-set.md).

## Famílias

| Família | Schema | Conteúdo | Consumida por |
|---|---|---|---|
| `regions` | `contextmap.reference.regions/v1` | máscaras/caixas 2D, áreas válidas e de exclusão, por frame | avaliação de Region Discovery |
| `semantics` | `contextmap.reference.semantics/v1` | conceitos open-vocabulary, ambiguidade, abstenção, rejeições explícitas, atributos | Semantic Interpretation e scoring; Semantic Fusion |
| `geometry` | `contextmap.reference.geometry/v1` | correspondências confiáveis 3D ↔ pixel (unidades e frames explícitos) | Sensor Association, Geometric Mapping |
| `identity` | `contextmap.reference.identity/v1` | identidade física entre observações, pares explicitamente distintos, escopo | Entity Resolution |
| `relations` | `contextmap.reference.relations/v1` | relações entre identidades anotadas, vocabulário de predicados e suas propriedades | Spatial Relations |
| `visibility` | `contextmap.reference.visibility/v1` | nível de visibilidade/oclusão por região, identidade ou observação | estratificação de todos os relatórios |
| `scene_context` | `contextmap.reference.scene_context/v1` | atributos de cena/contexto por amostra ou observação | estratificação de todos os relatórios |

`AnnotationFamily.from_schema()` resolve o identificador de schema declarado em `AnnotationFileEntry.schema`; uma versão desconhecida é recusada, não interpretada.

## Ausência não é verdade negativa

Anotações podem ser **parciais**. Um registro ausente significa "não anotado", nunca "não existe". Toda verdade negativa é explícita:

| Família | Como se expressa |
|---|---|
| `regions` | `Coverage.COMPLETE` no frame: toda região no escopo foi anotada, então uma predição sem par é falso positivo. `PARTIAL`: predições sem par não são penalizadas. `exclusion_areas` e `valid_areas` delimitam onde a anotação vale. |
| `semantics` | `rejected_concepts` (isto **não** é X). `UNKNOWN` significa que o anotador não soube dizer: a abstenção é a resposta certa e a correção é *não aplicável*. |
| `identity` | `distinct_pairs` (dois objetos diferentes) e `IdentityScope` por observação (`COMPLETE`/`PARTIAL`). Um par nem agrupado nem declarado distinto não está anotado. |
| `relations` | `RelationStatus.DOES_NOT_HOLD`; `AMBIGUOUS` e `UNKNOWN` são valores, não ausência. |
| `visibility` | `VisibilityLevel.UNKNOWN` é um julgamento; alvo sem registro não foi anotado. |

`SemanticStatus` fixa a contagem de conceitos: `LABELED` exige ao menos um conceito aceitável, `AMBIGUOUS` ao menos dois igualmente plausíveis e `UNKNOWN` nenhum.

## Open vocabulary e normalização

Os rótulos literais do anotador são sempre preservados. Dois rótulos só são comparados como iguais por uma **política de normalização explícita e versionada**, declarada no próprio documento:

- `casefold-exact/1` — case e espaços; **não** afirma sinonímia. É a mesma política do avaliador de Semantic Interpretation (`MATCHING_POLICY`).
- `casefold-alias/1` — a anterior mais uma tabela de aliases declarada no documento (`LabelAlias`); dois grupos não podem reivindicar o mesmo alias.

Nenhum conceito é forçado a uma taxonomia fechada. `SemanticAnnotationSet` valida os conceitos sob a política declarada: não há conceitos duplicados nem um conceito aceito e rejeitado ao mesmo tempo.

`SemanticAnnotationSet.to_semantic_annotation()` entrega um registro ao avaliador atual (`SemanticAnnotation`, que casa por `casefold-exact/1`); os aliases declarados são expandidos para as hipóteses aceitáveis. Um registro `UNKNOWN` não tem hipótese aceitável e é recusado explicitamente. `ground_truth_regions()` entrega as máscaras de um frame como `GroundTruthRegion`.

## Identidade e relações não vêm de rótulos iguais

- Uma `PhysicalIdentity` é o que o anotador declara: um agrupamento de ocorrências (`IdentityOccurrence`, uma por observação e região). Dois objetos com o mesmo conceito continuam duas identidades; não existe rótulo na identidade que permita inferir fusão.
- Uma `RelationAnnotation` liga duas identidades declaradas, com predicado do vocabulário do documento (`PredicateRule`) e **âncoras** em observações físicas. `PredicateRule` declara `symmetric` ou `inverse`; o inverso precisa estar declarado com o predicado original como seu inverso.

Os contratos de `Entity`/`Relation` das milestones de Semantic Mapping e Spatial Relations ainda não existem; estas anotações têm identidade própria (`identity_id`, `relation_id`) e não importam nem antecipam esses contratos. Quando eles existirem, adaptadores os ligarão às anotações.

## Ligação com observações físicas

Todo registro carrega `sample_id` e `observation_id` (`SourceObservationId`); `observation_references()` devolve as observações distintas de um conjunto, que é o que a validação de integridade usa para conferir o vínculo com o manifesto. `observation_id = None` (contexto de cena) significa "a amostra inteira". Nenhuma identidade de run de percepção participa.

## Persistência

- `write_annotation_set(path, set)` publica o arquivo de forma atômica e imutável (recusa sobrescrever) e devolve o `sha256:<hex>` que o `AnnotationFileEntry` do manifesto registra.
- `read_annotation_set(path)` despacha pelo `schema` do documento para a família certa; campos ausentes ou inválidos viram `AnnotationError`.

## O que este módulo não faz

- não decide o trust de uma anotação: isso é declarado no manifesto (`ReferenceTrust`) e nunca vem do conteúdo;
- não valida consistência entre registros e o manifesto (tamanho de máscara vs imagem, calibração, identidades referenciadas por relações, inverso/simetria entre relações): é a verificação de qualidade das anotações;
- máscaras usam `InlineMask` (uma lista binária por pixel), adequado para fixtures e amostras pequenas; um formato compacto para máscaras grandes fica para quando houver essa necessidade.
