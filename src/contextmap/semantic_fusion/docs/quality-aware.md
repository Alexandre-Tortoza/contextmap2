# Política ciente de qualidade (opcional)

`accumulate_quality_aware_evidence` roda **a mesma acumulação do baseline** e acrescenta um **peso de contribuição de fusão**, inspecionável, derivado da qualidade mensurável das observações. O baseline continua inalterado e disponível como braço de controle: esta política é uma escolha explícita e **não** é o padrão; existir não a torna canônica.

Identidade: `quality-aware-evidence-accumulation-v1`.

## O que o peso é e não é

É um **fator de contribuição de fusão** em `[0, 1]`, derivado só de componentes de qualidade declarados. **Não** é confiança semântica (VLM), similaridade ou suporte de scorer (CLIP/AlphaCLIP), nem probabilidade calibrada, e nada disso é lido para calculá-lo: um teste mostra que mudar a confiança da claim e os scores não altera nenhum fator. A qualidade bruta e a evidência semântica continuam intactas e inspecionáveis depois da fusão.

## Regras versionadas

| Etapa | Regra |
| --- | --- |
| Rampa por componente | `QualityRamp(quality_input, good, bad)` mapeia uma grandeza medida a um fator: `1` em `good` ou além, `0` em `bad` ou além, linear no meio. A ordem de `good` e `bad` diz se maior ou menor é melhor. |
| Fator da contribuição | O **mínimo** dos fatores dos componentes declarados (`minimum-of-component-factors`): a condição declarada mais fraca limita. |
| Componente ausente | Um componente que não pôde ser medido, ou uma observação sem qualidade, recebe o **fator neutro** declarado (`neutral_factor`) e a razão fica registrada (`NEUTRAL_FALLBACK`). Nunca vira zero em silêncio. |
| Fator da observação física | A **média** dos fatores das suas contribuições que sustentam a hipótese (`mean-of-supporting-contribution-factors`). |
| Suporte ponderado | A **soma** dos fatores das observações físicas que sustentam a hipótese, ao lado da contagem **sem** peso de observações físicas distintas (antes e depois da política). |

Grandezas disponíveis (`QualityInput`): mediana da profundidade do suporte (m), do ângulo fora do eixo (rad) e da distância à borda (px); parcela visível; fração ocluída; fração fora do suporte válido; mediana do resíduo de reprojeção (px, só com referência confiável); densidade de suporte por pixel de máscara; contagem de associados; deslocamento temporal da pose (ns). Uma grandeza opcional que o contrato marca como indisponível vira fallback neutro com a razão do próprio `ObservationQuality`.

## Correlação preservada

- N execuções de inferência sobre um frame são **uma** observação física, com fator médio que nunca passa de 1; o resultado não é N vistas independentes.
- Uma claim sobre muitos pontos de geometria não é multiplicada: o peso não depende da quantidade de geometria.
- A estrutura 3D estática (`PointRepresentationRef`) é listada uma vez no suporte e nunca é ponderada por vista.

## Peso não descarta evidência

Hipóteses, contribuições e incerteza são **exatamente** as do baseline; um fator `0` só significa que a observação não soma suporte ponderado. Nada é removido só por ter peso baixo.

## Configuração

`QualityAwareAccumulationPolicy(definitions_version, ramps, neutral_factor, baseline)` **não tem valores padrão** para as rampas, o fator neutro nem a versão das definições: são escolhas científicas que um perfil declara.

- `definitions_version`: a versão das definições de qualidade que a política entende; uma qualidade lida sob outra versão é recusada com erro claro.
- `baseline`: as configurações do baseline (rótulos de abstenção, margem de empate e canais). O canal `observation_quality` é sempre acrescentado, porque escolher esta política é declará-lo.
- `fingerprint()` cobre a identidade da política, as regras, as rampas, o fator neutro e o fingerprint do baseline, e entra em `FusedEvidenceProvenance.configuration_fingerprint`.

## Saída

O mesmo `FusedEvidence` do baseline, mais `weighting: QualityWeighting`:

- `contributions`: para cada contribuição, os componentes (`ComponentFactor` com o valor medido, o tratamento e o fator, ou a razão do fallback) e o fator combinado;
- `hypotheses`: para cada hipótese, `supporting_physical_observations` (antes), o fator de cada observação física e `weighted_support` (depois);
- `policy_id`, `combination_rule`, `observation_rule` e `definitions_version`, para reproduzir.

O contrato exige que os pesos cubram exatamente as contribuições e as hipóteses, que as observações listadas sejam as que sustentam a hipótese e que o suporte ponderado seja a soma dos fatores.

```python
fused = accumulate_quality_aware_evidence(
    support,
    observations=observations,
    grouping=grouping,
    perception_results=results,
    observation_qualities=qualities,  # ObservationQuality por observação
    policy=policy,
)
```

As mesmas entradas rodam no baseline e na política ciente de qualidade sem regenerar nada a montante.

## Estado da evidência

A política continua **opcional**. A primeira execução real (`corridor-02`, 20 frames; ver [avaliação](../../evaluation/docs/semantic_fusion.md)) não tem claims e não tem anotações, então não existe medida de que o peso ajude ou piore a semântica: o que ela mostra é o peso, inspecionável, variando com a distância e a borda. Duas lições de dado real para quem declarar rampas: (1) `visible_share` é muito baixa numa parte grande das regiões porque a pegada inclui os pontos ocluídos atrás da superfície de um mapa acumulado, então uma rampa sobre ela zera a maioria das vistas; (2) as rampas não têm valor padrão e devem ser declaradas antes de olhar qualquer saída de fusão.

## O que não faz

- não usa modelo aprendido de ponderação: toda regra e todo número estão versionados na política;
- não promove esta política a padrão sem avaliação controlada (issue #121);
- não altera a evidência semântica, os scores nem a estrutura 3D;
- não combina fator, confiança, similaridade e qualidade em um escalar.
