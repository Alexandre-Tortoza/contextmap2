# Avaliação de Entity Resolution

Este documento descreve `src/contextmap/evaluation/entity_resolution.py`: a validação determinística e a avaliação de identidade de um `EntityResolutionRunArtifact` persistido.

Entity Resolution decide quais entidades de origem são o mesmo objeto físico. Uma **fusão falsa** junta objetos distintos; uma **fusão perdida** deixa um objeto duplicado. As duas falhas têm custos diferentes para quem consome o mapa, então o harness as mantém separadas, com a causa de cada uma, e **não há score composto**. O harness só lê: lê o run pelo leitor público (`EntityResolutionRunReader`), nunca lê `debug/`, não repara nada e não ajusta limiar.

## Uso

```python
report = evaluate_entity_resolution(
    EntityResolutionRunReader(resolution_run_dir),
    entities=source_entities,  # as Entity de Semantic Mapping que o run resolveu
    reference=identity_annotations,  # IdentityAnnotationSet, nunca inferida de rótulos
    links=occurrence_links,  # OccurrenceLink: ocorrência anotada -> entidade de origem
    retrieval_policy=CandidateRetrievalPolicy(...),  # a do run, para reproduzir a recuperação
    reference_set_id="ci-subset-1.0.1",
)
report.passed  # contrato, reprodutibilidade e compatibilidade
report.identity  # métricas de identidade, cada classe de falha em separado
encode_entity_resolution_report(report)  # JSON com todas as camadas separadas
```

`evaluate_identity()`, `evaluate_splits()` e `evaluate_channel_ablation()` são as partes do relatório, chamáveis sozinhas sobre objetos em memória.

## Relatório

| Bloco | O que responde |
| --- | --- |
| `checks` | O run é consistente e reprodutível? Três camadas (`ResolutionValidationLayer`): `contract`, `reproducibility`, `compatibility`. Cada `ResolutionValidationCheck` traz o que examinou e as falhas exatas. |
| `identity` | Contra a referência explícita, o run fundiu errado ou deixou duplicatas? |
| `retrieval` | O par verdadeiro chegou a ser recuperado, **antes** de qualquer decisão da política? |
| `split` | Só com `split_reference`: a detecção de divisão, julgada à parte da qualidade de fusão. |
| `reproducibility` | O que o relatório foi calculado a partir de (`EvaluationReproducibility`). |

### Camadas de checagem

- **contract**: cada decisão aponta para evidência que existe para o seu par; só pares candidatos foram comparados; toda entidade de origem está em exatamente uma entidade resolvida; o artifact está íntegro (inventário e hashes).
- **reproducibility**: materializar as mesmas decisões de novo reproduz as entidades resolvidas e seus ids; as contradições de transitividade saem como uma materialização nova as encontra; recuperar candidatos de novo reproduz os conjuntos de candidatos.
- **compatibility**: nenhum canal medido mistura espaços de embedding ou de representação.

### Reprodutibilidade

`EvaluationReproducibility` registra: avaliador e versão, run avaliado com a sua **versão de schema** (`resolution_schema_version`) e o seu **digest** (`resolution_artifact_digest`, calculado por `contextmap.entity_resolution.resolution_artifact_digest`), o run de Semantic Mapping de origem (identidade e digest), o `reference_set_id`, o digest das anotações, as políticas (`papel:policy_id:fingerprint`) e a versão do código. Quem consome o run (por exemplo Spatial Relations) fixa o digest e a versão do schema do run exato que usou.

## Métricas do registro

Definidas em `default_metric_registry()` e calculadas exatamente como lá:

- `entity.false_merge.rate` = entidades resolvidas que juntam identidades **declaradas distintas** / entidades resolvidas com ao menos um membro anotado.
- `entity.duplicate.rate` = identidades representadas por mais de uma entidade resolvida / identidades anotadas com ao menos uma entidade avaliável.

Uma taxa é `None` quando o denominador é zero: o dado é excluído, nunca vira zero. `entity.semantic_accuracy.rate` **não** é reportada: acertar o rótulo não é acertar a identidade.

## Referência explícita e ocorrências

A referência (`IdentityAnnotationSet`) é indexada por **ocorrência** (amostra, observação, região), não por entidade. O vínculo entre uma ocorrência e a entidade de origem que nasceu dela é uma entrada explícita, `OccurrenceLink`; ele nunca é adivinhado por rótulo, posição ou nome de arquivo. Regras:

- só entram ocorrências de observações com escopo `COMPLETE`; o resto é contado em `ignored_links`;
- entidade sem ocorrência anotada fica fora, não conta como acerto nem como erro;
- um vínculo para uma entidade que o run não tem é recusado (`EntityResolutionEvaluationError`);
- uma entidade de origem ligada a mais de uma identidade já estava fundida antes da resolução: aparece em `source_entities_spanning_identities` e não é erro da resolução.

## Classes de falha, cada uma à parte

Todo par anotado termina numa **causa** exclusiva, por precedência:

| Par | Causa | Significado |
| --- | --- | --- |
| mesmo objeto | `merged` | acabou na mesma entidade resolvida |
| mesmo objeto | `retrieval_miss` | nunca foi candidato: erro de recuperação, não da política |
| mesmo objeto | `not_compared` | foi candidato, mas nenhuma decisão o cobre |
| mesmo objeto | `match_withheld_by_contradiction` | `MATCH`, mas a componente foi retida por contradição de transitividade |
| mesmo objeto | `match_not_merged` | `MATCH` sem fusão e sem contradição (inconsistência a investigar) |
| mesmo objeto | `false_distinct` | decidido `DISTINCT` |
| mesmo objeto | `unresolved` | a política se absteve |
| objetos distintos | `false_merge` | acabou na mesma entidade resolvida |
| objetos distintos | `not_candidate`, `not_compared`, `decided_distinct`, `unresolved`, `match_withheld_by_contradiction`, `match_not_merged` | como acima, para o lado dos distintos |

Além disso: precisão e revocação **por par** (`pairwise_precision`, `pairwise_recall`) com as contagens visíveis; contradições de transitividade; e a **recuperação** (`retrieval_recall`, candidatos por entidade e pares comparados), julgada antes da política. Abster-se (`unresolved`) é preferível a um `MATCH` sem explicação, e por isso é contado à parte, não como erro.

## Identidade de cada entidade resolvida

`IdentityEvaluation.identity_of_resolved_entity` dá a identidade anotada de cada entidade resolvida cujos membros avaliáveis pertencem a **uma** identidade (`dict(...)` dele é o mapa `ResolvedEntityReference -> identity_id`). Várias entidades resolvidas podem mapear para a mesma identidade: isso é uma duplicata e continua visível em `duplicated_identities`. Entidades cujos membros pertencem a mais de uma identidade (fusão falsa, ou origem já misturada) ficam em `resolved_entities_spanning_identities` e **fora** do mapa; a identidade não é escolhida. É a entrada que a avaliação de Spatial Relations consome.

## Ablação de canais

`evaluate_channel_ablation()` roda um `ResolutionArm` por conjunto de canais habilitados sobre as **mesmas** entidades de origem, os mesmos candidatos, a mesma referência e os mesmos vínculos: só os canais mudam (uma variável por vez). Cada `ArmEvaluation` traz as decisões por resultado, a `IdentityEvaluation` completa do braço e o tempo de resolução e de materialização, medido e reportado à parte da qualidade. Nomes de braço repetidos são recusados. Modelo de braços (o da issue), sobre o mesmo run de entidades; o harness aceita qualquer conjunto de canais, então os braços E e F só diferem no `RepresentationComparator` e no run de Point Representation que o alimenta:

| Braço | Canais |
| --- | --- |
| A | geometria |
| B | geometria + compatibilidade semântica |
| C | geometria + aparência visual |
| D | geometria + semântica + aparência |
| E | a linha de base escolhida + Point Representation determinística (opcional) |
| F | a linha de base escolhida + Point Representation aprendida (opcional) |

Os testes exercitam A e B sobre cenas sintéticas (uma paleta e uma pessoa que se sobrepõem em geometria: só geometria as funde, com a semântica não); C, D, E e F não têm run de referência ainda. Um canal indisponível não vota nem é zero: o braço que o habilita apenas o registra como não medido. Um braço mais rápido não é automaticamente melhor para o mapa.

## Divisão

`evaluate_splits()` julga `SplitCandidate`s sobre entidades rotuladas (`SplitReference`) como de um ou de vários objetos, à parte da qualidade de fusão: `suggested_recall` nas de vários objetos e `false_suggestion_rate` nas de um. A divisão é apenas diagnóstico; nada é dividido.

## Desempenho

O relatório traz, em separado e sem virar score: candidatos por entidade (média e máximo), pares comparados, tamanho do artifact em bytes e, por braço, o tempo de resolução e o de materialização. **Não mede** a memória de pico nem o custo de carregar features ou representações à parte; isso fica como lacuna.

Linhas de base medidas localmente (CPU, sem GPU, dados sintéticos): recuperação de candidatos sobre 20 mil entidades, 2,88 s; distância de suporte entre representações 3D, de 58 ms a 1,2 s por par, para 500 a 5000 pontos por lado. Servem de referência, não de meta.

## Limites

- **Só há evidência sintética.** Todos os testes usam entidades e runs construídos; o formato das anotações é exercitado contra o de `tests/fixtures/ci_subset/1.0.1/annotations/identity.json`, mas não há run real de Entity Resolution sobre dado real. Nada aqui é evidência de qualidade em dado real, e por isso este harness ainda **não** basta para escolher limiares em dado real: só demonstra que a escolha pode ser feita, com cada falha visível.
- O relatório não mede memória de pico nem o custo de carregar features ou representações em separado.
- As políticas não podem ser calibradas na mesma referência usada no relatório final sem dizê-lo; o relatório registra qual referência e quais políticas foram usadas.
- A referência de identidade cobre só o que a anotação declara `COMPLETE`.
