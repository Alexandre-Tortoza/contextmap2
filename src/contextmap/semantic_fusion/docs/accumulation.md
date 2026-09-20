# Política baseline de acumulação de evidência

`accumulate_baseline_evidence` transforma as observações espaciais de um `FusionSupport` em um `FusedEvidence`. É a política **canônica de controle**: transparente, determinística e deliberadamente conservadora, o ponto de comparação das políticas posteriores (por exemplo, a ciente de qualidade).

Identidade da política: `baseline-evidence-accumulation-v1`. A regra de chave de label faz parte dessa versão: mudá-la exige nova versão.

## O que faz

1. **Uma contribuição por observação espacial.** Cada `SpatialObservation` do suporte vira uma `EvidenceContribution` com referências às suas claims, aos scores de scorer, às features visuais e, quando fornecida, à qualidade da observação. A identidade da contribuição é `evidence_contribution_id_for(suporte, observação)`.
2. **Grupos por observação física.** Os grupos vêm do agrupamento (`PhysicalObservationGrouping`), que fornece o timestamp de aquisição de cada frame.
3. **Hipóteses por chave de label.** As claims são agrupadas pela *chave de label*: o texto sob normalização Unicode NFKC, em minúsculas (`casefold`) e com espaços colapsados. É uma regra tipográfica: `"Pallet"` e `"pallet "` são uma hipótese; `"pallet"` e `"wooden pallet"` são duas. Nenhum sinônimo, plural ou taxonomia é assumido. O label exibido é a grafia (com espaços colapsados) mais comum, com empate resolvido pela menor.
4. **Stance de cada claim para cada hipótese.**
   - `ABSTAINING`: a claim é uma abstenção (ver abaixo): nem suporte nem evidência contra;
   - `SUPPORTING`: a claim propõe a hipótese;
   - `AMBIGUOUS`: qualquer outra claim que vem da mesma contribuição de uma claim que sustenta a hipótese (alternativas de uma só interpretação) ou que tem papel `ALTERNATIVE`;
   - `CONFLICTING`: qualquer outra claim.
5. **Sinais tipados por claim.** A confiança da própria claim (`CLAIM_CONFIDENCE`, com `None` quando não pontuada, nunca zero) e um `SCORER_SUPPORT` por saída de scorer. Nada é somado, ponderado ou combinado.
6. **Contagem de evidência independente.** Hipóteses são numeradas pela chave de label (`hypothesis-0001`, …), então a ordem não sugere ranking. A contagem de suporte independente é o número de **observações físicas distintas** (`FusedEvidence.supporting_physical_observations`): inferência repetida sobre um frame e uma claim sobre muita geometria contam uma vez.

## Canais de evidência

Cada canal mantém a sua semântica, a sua escala e a sua proveniência; nenhum é renomeado, convertido nem misturado com outro, e **não há concatenação de vetores** de DINO, CLIP, PTv3 e scores. Um canal só entra quando a política o **declara**: ter dados disponíveis nunca o ativa.

| Canal | O que entra | Entrada exigida quando declarado |
| --- | --- | --- |
| `semantic_claims` | hipóteses, alternativas e abstenções do interpretador (obrigatório) | — |
| `semantic_scores` | saídas de scorer (`SCORER_SUPPORT`), específicas de cada scorer | `semantic_scores` |
| `visual_features` | referências às features visuais, com o `EmbeddingSpace` | — (vem das observações) |
| `observation_quality` | referências à qualidade mensurável da vista | `observation_quality_refs` |
| `geometry_support` | a geometria que a observação enxerga; **intrínseco**, sempre ativo | — |
| `point_representation` | estrutura 3D estática, anexada ao suporte | `point_representation_refs` |

- `BaselineAccumulationPolicy(channels=...)` declara o subconjunto; o padrão é só `semantic_claims`. Declarar `geometry_support` é aceito e não muda nada.
- **A mesma entrada serve a todas as configurações.** Dados oferecidos a um canal não declarado são ignorados, então uma execução só semântica e uma rica rodam dos mesmos artifacts a montante e diferem só pelos canais.
- **Falha clara, sem coerção.** Um canal declarado sem nenhuma entrada (`None`) é erro; um canal declarado que não encontra dados para um suporte fica ativo e vazio. Uma política sem `semantic_claims` é recusada.
- **A proveniência sobrevive.** `FusedEvidence.channels` lista cada canal ativo com as identidades que o alimentaram: interpretadores e scorers (`backend/modelo/versão`), espaços de embedding, versão das definições de qualidade, espaços de representação, e o mapa e a política de suporte. O contrato **recusa** dados de um canal que não foi declarado, então um efeito pode ser atribuído a um canal.
- A agregação numérica de features só seria permitida dentro de espaços compatíveis e sob política explícita; o baseline não agrega, e por isso este canal carrega apenas referências. Nenhuma similaridade entre espaços diferentes é calculada.
- O `fingerprint()` da política cobre os canais ativos.

## Abstenção

`unknown` **não é evidência negativa**. Uma claim cujo label (pela chave de label) está entre os `abstention_labels` **configurados** nunca vira hipótese, aparece como `ABSTAINING` sob cada hipótese e não conta como suporte nem como conflito: `pallet, unknown, pallet` é uma hipótese com dois frames de suporte e uma abstenção. **Nenhum label é abstenção a menos que a política diga**; por padrão o conjunto é vazio, então `unknown` seria só mais um label. Uma claim sem score continua sendo "não pontuada", nunca zero, e um score baixo continua distinto de um score ausente.

## Incerteza: reportada, nunca resolvida

Cada registro nomeia as contribuições e claims exatas que o produziram (`EvidenceReference`), a regra versionada (`rule_id`) e as hipóteses envolvidas:

| Tipo | Quando |
| --- | --- |
| `CONTRADICTION` | Pelo menos duas hipóteses têm claims **primárias** de suporte e essas claims vêm de **pelo menos dois frames físicos distintos**. Uma maioria não esconde a contradição: todas as claims primárias de suporte são listadas. |
| `AMBIGUITY` | Há duas ou mais hipóteses sem essa contradição: alternativas de uma só interpretação, ou runs que discordam sobre **um único** frame (inferência correlacionada, não frames contraditórios). |
| `NEAR_TIE` | Duas ou mais hipóteses com suporte primário têm contagens de **frames físicos de suporte** a menos de `near_tie_margin` uma da outra (`0` = exatamente iguais). Compara só contagens, nunca scores. |
| `INSUFFICIENT_EVIDENCE` | Nenhuma hipótese existe: só abstenções e vistas sem claim. A evidência lista essas claims e essas contribuições. |

Uma contradição e um empate podem coexistir. Uma hipótese única, ou vistas que concordam, não geram registro. Hipóteses que só aparecem como alternativa continuam visíveis em `hypotheses` com o papel preservado, mas não entram na contradição.

## Configuração

`BaselineAccumulationPolicy(abstention_labels, near_tie_margin)`: os labels de abstenção (comparados pela chave de label) e a margem de empate. O `fingerprint()` cobre a identidade da política e a configuração e entra em `FusedEvidenceProvenance.configuration_fingerprint`; grafias que diferem só tipograficamente dão o mesmo fingerprint.

## Entrada

```python
fused = accumulate_baseline_evidence(
    support,
    observations=observations,  # SpatialObservation por identidade
    grouping=grouping,  # PhysicalObservationGrouping das mesmas observações
    perception_results=results,  # PerceptionResult por identidade (dono das claims)
    semantic_scores=scores,  # opcional: SemanticSupport por resultado
    observation_quality_refs=qualities,  # opcional: ObservationQualityRef por observação
    point_representation_refs=structure,  # opcional: PointRepresentationRef
    policy=policy,  # opcional: BaselineAccumulationPolicy (canais, abstenção, empate)
)
```

Os canais são **opt-in** pela política (ver acima): scores, qualidade e estrutura só participam quando declarados. Um score de uma claim que não está anexada a uma observação do suporte é ignorado. Só as referências de estrutura ancoradas dentro do suporte são mantidas, cada uma uma vez, em `FusedEvidence.point_representation_refs`.

O resultado não depende da ordem de nenhuma entrada: observações, resultados, scores e referências são ordenados por identidade.

## Fronteira de qualidade

O baseline **ignora** a qualidade de observação como peso, mas **preserva** a referência em cada contribuição. Nada aqui pondera por distância, visibilidade ou reprojeção só porque esses campos existem; isso é a política ciente de qualidade, opcional e comparada contra este baseline.

## Erros

Falha cedo, com mensagem acionável, quando: uma observação do suporte não está nas entradas, pertence a outro mapa ou tem geometria fora do suporte; um `PerceptionResult` falta ou não contém uma claim referenciada; o agrupamento não cobre um frame ou não lista uma observação sob o seu frame; ou um scorer pontuou a mesma claim duas vezes.

## O que não faz

- não pondera por qualidade de observação;
- não reconhece refinamentos compatíveis como `pallet` e `wooden pallet`: isso exige uma regra explícita e versionada que o baseline não tem, então eles competem como hipóteses separadas (o que pode gerar contradição espúria; documentado como limitação);
- não escolhe vencedor, não calcula probabilidade nem combina confiança, similaridade e qualidade;
- não cria identidade de entidade nem aplica conhecimento prévio.
