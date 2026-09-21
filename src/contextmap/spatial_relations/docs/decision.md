# Política de decisão

`decide_relations(candidates, evidence)` transforma a evidência em `Relation` canônicas, sem esconder a incerteza. Só os canais **medidos** (geometria e contato) decidem; a evidência de observação é **corroborante** (ver [Evidência de observação](observation-evidence.md)). A política baseline é **conservadora e determinística**: prefere `UNRESOLVED` quando a evidência é insuficiente ou materialmente contraditória. São regras explícitas, não uma soma ponderada, e ela **não tem limiar próprio**: todo número mora na evidência que ela lê. Nenhum label, nenhum completamento por conhecimento de mundo e nenhum texto entram nela.

```python
result = decide_relations(candidates, evidence)
result.relations  # Relation, avaliadas e geradas, ordenadas por sujeito, predicado e objeto
result.decisions  # uma RelationDecision por relação, na mesma ordem
```

`CONSERVATIVE_DECISION_POLICY_ID = "conservative-relation-decision-v1"`; `decision_policy_fingerprint()` cobre o id, as regras e a versão da taxonomia, e vai na proveniência de toda relação.

## Estado a partir da evidência

Para cada candidato, a evidência de cada canal medido (no máximo um registro por canal) decide assim:

| Evidência | Estado | Regra (`DecisionRule`) |
| --- | --- | --- |
| algum canal `SUPPORTS`, nenhum `CONFLICTS` | `SUPPORTED` | `CHANNELS_SUPPORT` |
| algum canal `CONFLICTS`, nenhum `SUPPORTS` | `REJECTED` | `CHANNELS_CONFLICT` |
| um canal `SUPPORTS` e outro `CONFLICTS` | `UNRESOLVED` (`CONFLICTING_EVIDENCE`) | `CHANNELS_DISAGREE` |
| só `AMBIGUOUS`/`UNAVAILABLE`, ou nenhuma evidência | `UNRESOLVED` (`INSUFFICIENT_EVIDENCE`) | `NO_DECISIVE_EVIDENCE` |

- **A evidência de observação nunca decide**: ela não estabelece, não rejeita e não sobrepõe um veredito medido, mesmo quando o contradiz; fica registrada em `ignored` com `contradicts_relation` marcado. Só observação, sem evidência medida que decidiu, deixa a relação `UNRESOLVED`.
- **Evidência que não decidiu não é voto negativo.** Um canal ambíguo ou indisponível não impede `SUPPORTED` quando outro canal sustenta; ele fica registrado em `RelationDecision.ignored` com o status que teve.
- Um candidato **sem nenhuma evidência** vira `UNRESOLVED` (não é descartado nem rejeitado): "não medi" é diferente de "medi e não vale".
- O conflito entre canais **permanece visível**: a relação lista toda a evidência (`relation_evidence_refs`), a incerteza cita os registros que sustentam e os que contradizem, e a decisão os guarda em `deciding_evidence_refs`.

## Consistência estrutural

Depois de decidir cada candidato, o conjunto é verificado. Uma violação **rebaixa** as relações envolvidas para `UNRESOLVED` (`INCONSISTENT_STRUCTURE`, regra `STRUCTURAL_INCONSISTENCY`), com a relação contrária nomeada, em vez de manter as duas em silêncio:

- um predicado **dirigido** não pode ser `SUPPORTED` nos dois sentidos (`a ABOVE b` e `b ABOVE a`; para caixas idênticas, `INSIDE` também);
- um predicado **simétrico** avaliado nos dois sentidos deve chegar ao mesmo estado.

Só relações já decididas são rebaixadas: uma `UNRESOLVED` já diz que não sabe. As relações rebaixadas mantêm os registros que as decidiram antes, e o suporte num sentido com rejeição no outro é consistente e não é tocado.

## Inverso e simetria

Só depois da consistência as relações de sentido oposto são **geradas** a partir do estado final da relação avaliada, com a mesma evidência, a mesma incerteza e `derived_from` apontando para a origem:

- inverso (`ABOVE` → `BELOW`, `IN_FRONT_OF` → `BEHIND`, `INSIDE` → `CONTAINS`): `DERIVED_FROM_INVERSE`;
- gêmea simétrica (`NEXT_TO`, `INTERSECTS`, `TOUCHING`): `DERIVED_BY_SYMMETRY`, exceto quando o candidato já foi avaliado nos dois sentidos;
- `ON_TOP_OF` e `LEANING_AGAINST` não geram nada (sem inverso no vocabulário).

Assim `BELOW` nunca discorda de `ABOVE` e uma gêmea nunca discorda da origem: não há duas medições sobre a mesma geometria.

## `RelationDecision`

| Campo | Significado |
| --- | --- |
| `relation_id` | A relação a que a decisão se refere. |
| `rule` | A regra que fixou o estado. |
| `deciding_evidence_refs` | Registros que sustentaram ou contradisseram o predicado (ordenados, únicos). |
| `ignored` | `EvidenceUse` (`evidence_id`, `channel`, `status`, `contradicts_relation`): registros que não decidiram: medidos ambíguos ou indisponíveis e toda a evidência de observação, com a contradição ao estado final marcada. |
| `detail` | Explicação determinística. |

Um registro não pode estar em `deciding_evidence_refs` e em `ignored` ao mesmo tempo, e evidência decisiva de um canal medido nunca é ignorada. Toda relação `SUPPORTED` tem, portanto, um rastro auditável até os registros que a decidiram. `encode_relation_decision`/`decode_relation_decision` serializam a decisão em JSON e revalidam na leitura.

## Validação da entrada

Levantam `ValueError`: evidência que não é sobre um candidato do conjunto; o mesmo registro duas vezes; evidência com **linhagem incompatível** (versão da taxonomia, frame do mapa, mapa geométrico ou, quando o registro traz, fingerprint das convenções de frame diferentes dos usados para gerar os candidatos; a evidência de observação não tem coordenadas e só é comparada pela taxonomia). O resultado não depende da ordem de entrada.

## O que a política não faz

- não introduz soma ponderada nem score, e não decide por um limiar próprio;
- não completa relações por conhecimento externo nem gera linguagem natural;
- não produz um objeto de decisão específico de modelo: só `Relation` e `RelationDecision`;
- não corrige entidades resolvidas erradas.
