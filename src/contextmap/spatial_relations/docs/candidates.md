# Geração de candidatos

Avaliar todo predicado para todo par de entidades é quadrático no tamanho do mapa e quase sempre inútil: um objeto do outro lado do prédio não está ao lado, acima nem dentro de nada daqui. `generate_relation_candidates()` reduz o trabalho com pré-condições geométricas explícitas e baratas e registra por que cada par foi mantido ou descartado.

**Um candidato é uma razão para medir, não evidência.** `RelationCandidate` não tem status, score nem estado; decidir se a relação vale é papel dos avaliadores e da política de decisão. A função só lê referências de entidades e `EntityGeometry`: não há por onde um label semântico criar ou excluir um candidato.

## Entrada e saída

```python
generate_relation_candidates(
    entities, policy=CandidatePolicy(...), conventions=FrameConventions(...)
)
```

- `entities`: `Mapping[ResolvedEntityReference, EntityGeometry]` de **um** artifact de resolução (misturar artifacts levanta `ValueError`). A ordem de inserção é irrelevante.
- `CandidatePolicy` (sem valores padrão): os predicados a gerar (únicos, cada um avaliado diretamente; um derivado como `BELOW` é recusado, seleciona-se o inverso), `proximity_radius_m` e `directional_radius_m`. O alcance direcional é separado porque "acima" e "à frente de" alcançam muito mais que "ao lado de". Os dois alcances devem cobrir as tolerâncias de distância dos avaliadores; se não cobrirem, uma relação verdadeira pode se perder antes de ser medida. A política sozinha não conhece os avaliadores, então essa premissa é verificada onde as políticas do run se encontram (ver [coerência com os avaliadores](#coerência-entre-alcance-e-tolerâncias-dos-avaliadores)); uma perda que ainda reste é uma falha de recuperação de candidatos, medida à parte da qualidade dos predicados.
- `FrameConventions`: um predicado cujo eixo não foi declarado é **pulado e reportado** (`SkippedPredicate`, uma vez por execução, não por par); geometria em outro frame ou de outro mapa levanta `IncompatibleFrameError`.
- `RelationCandidateSet`: `candidates`, `exclusions` (com a razão), `skipped_predicates`, `entity_count`, `pairs_not_enumerated` e a proveniência (`CANDIDATE_POLICY_ID = "bounds-neighborhood-candidates-v1"`, fingerprint da política, versão da taxonomia, frame, mapa geométrico e fingerprint das convenções).

## Determinismo, simetria e duplicatas

- Mesmas entidades, política e convenções dão o mesmo conjunto, ordenado por sujeito, predicado e objeto (referência = artifact de resolução, depois id).
- **Simétrico** (`NEXT_TO`, `INTERSECTS`, `TOUCHING`): um candidato por par **não ordenado**, com o sujeito de menor referência.
- **Dirigido** (`ABOVE`, `IN_FRONT_OF`, `INSIDE`, `ON_TOP_OF`, `LEANING_AGAINST`): cada sentido é verificado por conta própria, então o par `(a, b)` e o par `(b, a)` são candidatos distintos quando ambos passam.
- **Derivado** (`BELOW`, `BEHIND`, `CONTAINS`): nunca é candidato; a relação é gerada do sentido avaliado do inverso.
- Um mesmo par dirigido nunca é candidato e exclusão ao mesmo tempo.

## Pré-condições

Cada uma é uma condição **necessária** para que o avaliador correspondente **sustente** a relação (um caso que só seria ambíguo pode ser descartado antes de ser medido). São verificadas em ordem fixa, e a primeira que falha é a razão registrada:

| # | Pré-condição | Predicados | Razão de exclusão |
| --- | --- | --- | --- |
| 1 | a distância entre os limites está dentro do alcance | todos (direcional para `ABOVE`/`IN_FRONT_OF`, de proximidade para os demais) | `BEYOND_PROXIMITY_RADIUS` / `BEYOND_DIRECTIONAL_RADIUS` |
| 2 | as seções transversais ao eixo se sobrepõem (o sujeito só está "acima" do que está sob ele) | `ABOVE`, `IN_FRONT_OF`, `ON_TOP_OF` | `NO_FOOTPRINT_OVERLAP` |
| 3 | o centro do sujeito está mais adiante no eixo que o do objeto (`a ABOVE b` e `b ABOVE a` nunca são ambos plausíveis) | os mesmos | `SUBJECT_NOT_ON_DIRECTED_SIDE` |
| 4 | o sujeito cabe no objeto em todo eixo, com a folga do alcance de proximidade | `INSIDE` | `CONTAINMENT_IMPOSSIBLE` |

O eixo de 2 e 3 é o `up_axis` (`ABOVE`, `ON_TOP_OF`) ou o `forward_axis` (`IN_FRONT_OF`) declarados em `FrameConventions`. `LEANING_AGAINST` só exige proximidade: uma tábua encostada numa parede tem seção transversal adjacente, não sobreposta, e a compatibilidade de orientação é evidência dos avaliadores, não filtro de candidatos.

`bounds_gap_m` é a distância euclidiana entre as faces mais próximas (`0` se as caixas se sobrepõem ou se tocam) e fica registrada no candidato e na exclusão.

## Coerência entre alcance e tolerâncias dos avaliadores

Uma pré-condição que falha descarta o par antes de qualquer avaliador medi-lo. Se o alcance de proximidade for menor que uma tolerância de um avaliador, pares que ele aceitaria são excluídos com uma razão tecnicamente correta, e o run parece bem-sucedido com relações faltando. Por isso `RelationsRunPolicies`, o único ponto em que a política de candidatos e as dos avaliadores coexistem, recusa na construção (`ValueError`) um `proximity_radius_m` que não cubra, **só para os avaliadores presentes**:

| Avaliador | Condição | Por quê |
| --- | --- | --- |
| geométrico | `next_to_max_gap_m <= proximity_radius_m` | `NEXT_TO` aceita vão até `next_to_max_gap_m`; acima do alcance o par vira `BEYOND_PROXIMITY_RADIUS` |
| geométrico | `2 * containment_slack_m <= proximity_radius_m` | `INSIDE` tolera `containment_slack_m` além de **cada** face, então o sujeito pode exceder a extensão do objeto em `2 * containment_slack_m` num eixo; acima do alcance o par vira `CONTAINMENT_IMPOSSIBLE` |
| contato | `contact_distance_m + contact_tolerance_m <= proximity_radius_m` | é o raio de busca do canal de contato (`search_radius_m`), e o vão entre os limites nunca passa da distância entre dois pontos das entidades |

A mensagem nomeia cada parâmetro envolvido, seu valor e as relações que seriam perdidas, e lista todas as violações de uma vez. Os limites são inclusivos, e um canal ausente (`geometric`/`contact` `None`) não tem tolerância a cobrir. A soma do contato é comparada com o mesmo arredondamento de ponto flutuante que o avaliador usa: um alcance decimalmente igual à soma pode ser recusado quando a soma em ponto flutuante passa dele (a mensagem mostra o valor exato). O alcance direcional e as pré-condições de pegada e de lado não são verificados aqui. No runtime, a recusa é reportada no componente `spatial_relations.candidate`.

## Escala

Sem retorno a todos-os-pares. Os limites são ordenados ao longo do eixo em que os centros mais se espalham e uma varredura (sweep and prune) só pareia caixas cujo intervalo nesse eixo está dentro do maior alcance da política, então um corredor longo custa trabalho proporcional aos pares realmente próximos. Os pares que a varredura prova mais distantes que todo alcance **não são enumerados** e apenas contados (`pairs_not_enumerated`): só esses ficam fora do registro de exclusões. Uma varredura com 300 entidades enfileiradas gera zero candidatos e zero exclusões e conta os 44 850 pares não enumerados.

O registro de exclusões **não é limitado**: todo par enumerado gera, para cada predicado avaliado e cada sentido, um candidato ou uma `CandidateExclusion`, e todas ficam em memória e são serializadas no conjunto de candidatos. Elas crescem com predicados × sentidos × pares próximos, e também com a razão entre o alcance direcional e o de proximidade, porque um par enumerado pelo alcance direcional gera uma exclusão `BEYOND_PROXIMITY_RADIUS` para cada predicado de proximidade. Numa pilha densa de 101 entidades (#601), 26 236 exclusões para 7 764 candidatos ocupavam 77% dos bytes de `relation-candidates.jsonl`. Limitar esse registro muda o schema do artifact e está em aberto na #601.

## Limites conhecidos

- Só limites alinhados aos eixos (`EntityGeometry.bounds`): entidades muito alongadas e diagonais têm caixas folgadas, o que aumenta candidatos, nunca os perde.
- Os alcances são configuração da execução; nenhum é específico de um dataset.
