# Avaliação de Spatial Relations

Relações erradas podem tornar enganoso um mapa contextual que está certo. Elas podem estar erradas por **quatro razões diferentes**: a entidade foi resolvida errado, o estágio de candidatos nunca olhou para o par, o predicado geométrico mediu errado, ou a reconciliação deixou a relação sem decisão. O avaliador `evaluate_spatial_relations()` **mantém essas causas separadas** em vez de dobrá-las num número só, e **nunca conserta uma entidade errada** para salvar uma relação.

```python
report = evaluate_spatial_relations(
    SpatialRelationsRunReader(run_dir),
    reference=relation_annotation_set,  # RelationAnnotationSet do reference set
    identity=entity_resolution_report.identity,  # IdentityEvaluation da avaliação de Entity Resolution
    identity_reproducibility=entity_resolution_report.reproducibility,  # mesma avaliação, EvaluationReproducibility
    code_version="...",
)
envelope = spatial_relations_evaluation_report(report, registry=..., reference_set=...)
```

## Entradas

- O **run persistido** de Spatial Relations (`SpatialRelationsRunReader`): relações, candidatos, exclusões e a linhagem.
- Um **`RelationAnnotationSet`** do reference set: relações entre identidades físicas anotadas, com predicados no vocabulário do próprio documento e estados `holds`, `does_not_hold`, `ambiguous` e `unknown`.
- O **`IdentityEvaluation` da avaliação de Entity Resolution** (`identity`), do **mesmo run de resolução** em que as relações foram construídas: seu `identity_of_resolved_entity` diz qual identidade anotada cada entidade resolvida tem, e `resolved_entities_spanning_identities` lista as entidades resolvidas que misturam identidades (fusões falsas), que ele deixa de fora do mapeamento e o avaliador **nunca adivinha** (`entities_spanning_identities`).
- A **`EvaluationReproducibility` dessa mesma avaliação** (`identity_reproducibility`), vinda do `EntityResolutionEvaluationReport.reproducibility` que a produziu: é a única prova de qual run de resolução e qual digest de artifact a avaliação de identidade é sobre, porque `IdentityEvaluation` sozinha não carrega essa proveniência e um mapeamento vazio (nenhuma identidade anotada/elegível) é um resultado válido que não prova nada por si. `evaluate_spatial_relations` confere `run_id` **e** `resolution_artifact_digest` contra a linhagem do próprio run de relações **antes** de pontuar qualquer coisa, e nunca infere isso das referências presentes no mapeamento; uma avaliação de identidade de outro run, ou do mesmo run mas de outro artifact, é recusada com `SpatialRelationsEvaluationError`.
- **A metadata correta não dispensa a checagem estrutural.** Depois de conferir `identity_reproducibility`, o avaliador também exige que todo `ResolvedEntityReference` que a própria `IdentityEvaluation` nomeia (em `identity_of_resolved_entity` e em `resolved_entities_spanning_identities`) tenha o mesmo `resolution_run_id` esperado. As duas checagens são independentes: uma `identity_reproducibility` correta acompanhada de referências de outro run é recusada, porque `Relation` usa a referência completa como chave e uma referência estrangeira não erraria alto sozinha — ela só deixaria de casar com qualquer entidade do run e viraria ruído (`reference_without_entity`/`supported_on_unmatched_entities`), mascarando a violação. É a única ponte entre as duas coisas: uma referência cuja identidade não tem entidade correspondente é contada à parte (`reference_without_entity`, atribuída a Entity Resolution) e **não** vira erro de relação; uma identidade casada com **mais de uma** entidade é pulada e listada (`identities_with_several_entities`), porque uma entidade duplicada torna a comparação injusta.

## Predicados e a referência

O texto do predicado da referência só é mapeado para um predicado canônico pela normalização versionada do próprio conjunto de anotações e pelos **nomes canônicos exatos** (`canonical_predicate`); qualquer outra palavra vai para `unmapped_predicates` e não é pontuada (sem sinônimos escondidos). A referência é **expandida pela taxonomia**: uma relação que vale implica o inverso (`above` → `below`) e a gêmea simétrica (`next to` nos dois sentidos), com o mesmo estado. Um vocabulário de referência que **contradiz** a taxonomia para um predicado canônico (por exemplo `above` declarado simétrico, ou o inverso declarado como outro predicado canônico) é recusado com `SpatialRelationsEvaluationError`; omitir uma propriedade é aceito porque a taxonomia a fornece, e um inverso que não é canônico (como o `supports` da referência de CI) fica sem mapear.

Estados que não pontuam: `ambiguous` e `unknown` são só contados (`ambiguous_reference`, `unknown_reference`), e a mesma tripla anotada com estados diferentes é um conflito (`conflicting_reference`) e também não é pontuada.

## Métricas por predicado

Tudo é reportado **por predicado canônico** (`RelationPredicateEvaluation`, os 11 sempre presentes, em ordem fixa); **não existe score agregado**. Os contadores são a fonte, e as taxas derivam deles:

| Contador | Significado |
| --- | --- |
| `annotated_holds` / `annotated_does_not_hold` | Relações da referência (expandidas) sobre entidades casadas. |
| `true_positives` | Vale e foi previsto `SUPPORTED`. |
| `missed_unresolved` / `missed_rejected` / `missed_not_retrieved` | Vale e **não** foi `SUPPORTED`, separado por causa: ficou `UNRESOLVED`, foi `REJECTED`, ou o estágio de candidatos nunca produziu o par. |
| `false_positives` | Não vale e foi previsto `SUPPORTED` (violação de negativo). |
| `true_negatives` | Não vale e foi `REJECTED` ou nunca foi candidato. |
| `unresolved_on_negatives` | Não vale e ficou `UNRESOLVED`. |
| `unannotated_supported` | `SUPPORTED` entre entidades casadas sem nenhuma anotação. |

`unannotated_supported` é reportado e **não** pontuado como falso positivo: a anotação é de mundo aberto, então uma previsão sem anotação é desconhecida, não errada.

| Taxa | Definição |
| --- | --- |
| `precision` | `TP / (TP + FP)` sobre relações anotadas. |
| `recall` | `TP / annotated_holds`. |
| `f1` | Média harmônica de precisão e revocação (`relations.f1` do registro). |
| `false_relation_rate` | `FP / annotated_does_not_hold` (`relations.negative_violation.rate`). |
| `missed_relation_rate` | `FN / annotated_holds`. |
| `unresolved_rate` | Relações anotadas (vale ou não) que terminaram `UNRESOLVED`, sobre todas as anotadas. |
| `candidate_retrieval_recall` | `(annotated_holds - missed_not_retrieved) / annotated_holds`. |

Uma taxa sem população é `None` (não aplicável), **nunca zero**.

## Falhas de recuperação

`retrieval_misses` conta, com a razão, os pares que valem e o estágio de candidatos nunca produziu: `excluded:<razão>` (a exclusão registrada, incluindo as de predicados derivados lidas no sentido avaliado do inverso), `skipped_predicate:frame_conventions` ou `pair_not_enumerated_or_predicate_not_selected`. Assim, uma falha de recuperação de candidatos é distinguível de uma falha do predicado geométrico ou da reconciliação.

## Consistência estrutural

Independente de qualquer anotação, o avaliador confere nas relações persistidas: **inverso** (`ABOVE`/`BELOW`, `IN_FRONT_OF`/`BEHIND`, `INSIDE`/`CONTAINS` com estados diferentes), **simétrico** (a gêmea com estado diferente) e **suporte mútuo** de um predicado dirigido nos dois sentidos. Cada violação é reportada uma vez, com as relações envolvidas (`RelationConsistencyViolation`). Um run decidido pela política baseline não as tem; uma violação persistida indica um defeito.

## Reprodutibilidade

O relatório (`SpatialRelationsEvaluationReport`, `to_dict()` em JSON) traz: o run avaliado e a versão do schema do artifact, o run de resolução e o seu digest, o mapa geométrico, a versão da taxonomia, cada política efetiva com id e fingerprint (a partir do manifest), a política de normalização e o número de relações da referência, a versão e o id do avaliador e o `code_version`. `spatial_relations_evaluation_report()` o coloca no envelope comum com as métricas do registro **por predicado** (como estratos, `relations.f1` e `relations.negative_violation.rate`), o digest da configuração e as duas linhagens de entrada; um predicado sem população anotada é `NOT_APPLICABLE`, nunca zero, e sem população nenhuma cada métrica aparece uma vez, sem estrato. O avaliador **não altera** o artifact avaliado.

## Limites conhecidos

- **Só fixtures sintéticos.** Não há relatório sobre dado real nem reference set real de relações; a referência de CI só cobre três predicados livres.
- O mapeamento entidade → identidade **vem da avaliação de Entity Resolution** (`IdentityEvaluation`), e não é produzido nem corrigido aqui: uma identidade duplicada ou uma entidade que mistura identidades é uma falha daquela avaliação, reportada e não pontuada como erro de relação.
- A anotação é de mundo aberto e só cobre alguns pares: a precisão é medida contra os negativos **explícitos**, e as previsões sem anotação ficam contadas à parte.
- Uma relação cujo par de identidades não foi anotado em nenhuma direção não entra em nenhuma taxa.
