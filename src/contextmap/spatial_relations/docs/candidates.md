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
- `RelationCandidateSet`: `candidates`, `exclusions` (com a razão; todas as de cada grupo `(predicado, razão)` dentro do teto), `exclusion_summaries` (um `CandidateExclusionSummary` por grupo acima do teto, ver [Teto de exclusões](#teto-de-exclusões)), `skipped_predicates`, `entity_count`, `pairs_not_enumerated` e a proveniência (`CANDIDATE_POLICY_ID = "bounds-neighborhood-candidates-v1"`, fingerprint da política, versão da taxonomia, frame, mapa geométrico e fingerprint das convenções).

## Determinismo, simetria e duplicatas

- Mesmas entidades, política e convenções dão o mesmo conjunto, ordenado por sujeito, predicado e objeto (referência = artifact de resolução, depois id).
- **Simétrico** (`NEXT_TO`, `INTERSECTS`, `TOUCHING`): um candidato por par **não ordenado**, com o sujeito de menor referência.
- **Dirigido** (`ABOVE`, `IN_FRONT_OF`, `INSIDE`, `ON_TOP_OF`, `LEANING_AGAINST`): cada sentido é verificado por conta própria, então o par `(a, b)` e o par `(b, a)` são candidatos distintos quando ambos passam.
- **Derivado** (`BELOW`, `BEHIND`, `CONTAINS`): nunca é candidato; a relação é gerada do sentido avaliado do inverso.
- Um mesmo par dirigido nunca é candidato e exclusão ao mesmo tempo, e nunca aparece em duas exclusões (listada no conjunto e entre as mais próximas de um resumo).

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

Todo par enumerado ainda gera, para cada predicado avaliado e cada sentido, um candidato ou uma exclusão. As exclusões crescem com predicados × sentidos × pares próximos, e também com a razão entre o alcance direcional e o de proximidade, porque um par enumerado pelo alcance direcional gera uma exclusão `BEYOND_PROXIMITY_RADIUS` para cada predicado de proximidade. Numa pilha densa de 101 entidades, 26 236 exclusões para 7 764 candidatos ocupavam 77% dos bytes de `relation-candidates.jsonl`. O [teto de exclusões](#teto-de-exclusões) limita esse registro.

## Teto de exclusões

As exclusões são agrupadas por `(predicado, razão)`. Um grupo com até `_MAX_LISTED_EXCLUSIONS = 32` exclusões é listado inteiro em `exclusions`, exatamente como antes do teto (#601). Um grupo maior lista só as 32 **mais próximas** e ganha um `CandidateExclusionSummary`:

| Campo | Significado |
| --- | --- |
| `predicate`, `reason` | O grupo. |
| `count` | Todas as exclusões do grupo, listadas ou não (sempre mais que as listadas). |
| `min_gap_m`, `max_gap_m` | O menor e o maior `bounds_gap_m` do grupo inteiro; o menor é o da primeira listada. |
| `digest` | `sha256-multiset:` seguido de 64 dígitos hexadecimais, sobre **todas** as exclusões do grupo (definição abaixo). |
| `nearest` | As exclusões listadas, em ordem de `bounds_gap_m` crescente e, no empate, pela referência do sujeito (artifact de resolução, depois id), pelo predicado e pela referência do objeto. |

As mais próximas são as de menor `bounds_gap_m`, com o mesmo desempate: numa falha de alcance, são os pares que quase passaram, os que um alcance um pouco maior teria mantido. A seleção não depende da ordem em que a varredura encontra os pares, e cada grupo guarda na memória só o teto, a contagem, as distâncias extremas e o digest parcial **enquanto os pares são enumerados**, e não no fim: a memória da geração não cresce com o número de exclusões. O teto é uma constante nomeada, um limite do registro e não um limiar científico: ele não muda quais pares são candidatos, então não é parâmetro de `CandidatePolicy` nem entra no fingerprint.

**Digest de um grupo.** Cada exclusão é codificada como JSON compacto (`separators=(",", ":")`) com chaves ordenadas, exatamente sobre `subject_entity_ref`, `predicate`, `object_entity_ref`, `reason` e `bounds_gap_m` (o registro `exclusion` persistido sem a chave `record`; as referências como Entity Resolution as codifica). O digest é `sha256-multiset:` seguido de `format(sum(int(SHA-256(codificação), big-endian)) mod 2**256, "064x")`. Uma soma não depende da ordem das exclusões: a varredura as encontra fora da ordem canônica, e o digest é calculado com memória constante enquanto elas aparecem; quem tiver a lista completa (por exemplo, gerando os candidatos de novo sem teto) o recalcula em qualquer ordem. É mais fraco que o hash da lista ordenada contra colisões construídas de propósito; ou-exclusivo seria ainda mais fraco, porque exclusões iguais se cancelariam, e não é usado.

**Como o teto foi escolhido.** Critério primário: sobre o fixture anotado dos testes, nenhuma falha de recuperação que o run sem teto explica com a exclusão listada pode virar `excluded_unlisted` no avaliador. O fixture (`tests/spatial_relations/relation_storeroom_fixture.py`) é um depósito com piso, sofá, um quadro na parede, uma estante de duas tábuas com seis caixas cada e cinco caixotes enfileirados no chão, com as políticas dos testes (alcance de proximidade 0,6 m, direcional 2,0 m). A anotação é a de uma pessoa: caixotes a 0,7 m e a 0,8 m estão "next to" (além do alcance, `BEYOND_PROXIMITY_RADIUS`) e o quadro está "above" do sofá encostado na parede, embora as pegadas não se sobreponham por alguns centímetros (`NO_FOOTPRINT_OVERLAP`). Os grupos dessas exclusões têm 129 (`NEXT_TO`/`BEYOND_PROXIMITY_RADIUS`) e 240 (`ABOVE`/`NO_FOOTPRINT_OVERLAP`) exclusões, e as das falhas anotadas ficam nas posições 10 (0,7 m) e 26 (0,8 m) do primeiro e 25 e 26 (o quadro e o sofá, nos dois sentidos, a 0,404 m) do segundo, porque as caixas da estante formam vários pares mais próximos. Regra: o menor teto de {8, 16, 32, 64} que cumpre o critério.

| Teto | Falhas anotadas explicadas pela exclusão listada (das 6) | Exclusões não listadas na pilha densa | Bytes de `relation-candidates.jsonl` na pilha densa | Pico de `tracemalloc` da geração na pilha densa |
| --- | --- | --- | --- | --- |
| sem teto (antes) | 6 | 0 de 26 236 | 10 726 210 | 9,7 MiB |
| 8 | 0 | 26 164 (99,7%) | 2 514 818 | 2,5 MiB |
| 16 | 2 | 26 092 (99,5%) | 2 536 058 | 2,6 MiB |
| **32** | **6** | 25 948 (98,9%) | 2 578 522 | 2,6 MiB |
| 64 | 6 | 25 660 (97,8%) | 2 663 450 | 2,7 MiB |

A pilha densa é a de 10 × 5 × 2 caixotes de 0,4 m sobre um piso (101 entidades, identidades embaralhadas, as mesmas políticas), medida numa máquina de desenvolvimento. O teto 8 não cumpre o critério: as três falhas anotadas saem da lista. O 32 é o menor que cumpre; ele custa só ~64 KB a mais que o 8 na pilha densa, porque quase todo o registro restante são os candidatos. A geração da pilha densa passou de 0,30 s para ~0,44 s, pelo digest calculado sobre cada exclusão. `tests/spatial_relations/test_relation_evaluation.py` mantém o fixture, a matriz e o critério no CI.

## Limites conhecidos

- Só limites alinhados aos eixos (`EntityGeometry.bounds`): entidades muito alongadas e diagonais têm caixas folgadas, o que aumenta candidatos, nunca os perde.
- Os alcances são configuração da execução; nenhum é específico de um dataset.
- Acima do teto, as exclusões não listadas só são auditáveis pela contagem, pelas distâncias extremas e pelo digest: para saber se um par específico está entre elas é preciso gerar os candidatos de novo. O teto foi escolhido num fixture sintético; num mapa real com mais pares mais próximos que a falha anotada, ela pode sair da lista mesmo com 32, e o avaliador a reporta como `excluded_unlisted` em vez de dar uma razão exata.
- A seleção pelas mais próximas favorece as falhas de alcance. Para `NO_FOOTPRINT_OVERLAP`, `SUBJECT_NOT_ON_DIRECTED_SIDE` e `CONTAINMENT_IMPOSSIBLE`, o `bounds_gap_m` não mede quão perto o par esteve de passar, então as listadas são as mais próximas no espaço, não necessariamente as mais informativas.
