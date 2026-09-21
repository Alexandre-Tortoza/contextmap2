# Estado semântico da entidade

Este documento descreve `src/contextmap/semantic_mapping/semantic_state.py` (os contratos) e `state_mapping.py` (as regras que mapeiam `FusedEvidence` para eles).

Uma entidade preserva a evidência semântica acumulada a montante em vez de colapsá-la em um par label/confiança. O estado é inspecionável **sem carregar nenhum backend de percepção ou de modelo**: só identidades, textos e números.

## Distinções preservadas

| Distinção | Como o contrato a mantém |
| --- | --- |
| hipótese × atributo | `EntityHypothesis` é o que a entidade pode ser; `EntityAttribute` é uma propriedade dela, com evidência e regra próprias. |
| alternativa × contradição | Alternativas competem entre si; uma contradição (`UncertaintyKind.CONTRADICTION`) é observações físicas distintas apoiando hipóteses incompatíveis. |
| desconhecido/abstenção × evidência negativa | `EvidenceStance.ABSTAINING` não é apoio nem evidência contra. |
| sem score × score baixo | `SupportSignal.value` `None` (não pontuado) é diferente de `0.0`. |
| propriedade observada × conhecimento externo | `AttributeOrigin.OBSERVED`/`DERIVED` exigem evidência; `EXTERNAL_KNOWLEDGE` é rotulado e exige uma derivação documentada. |

## Contratos

### `EntitySemanticState`

| Campo | Significado |
| --- | --- |
| `hypotheses` | Todos os candidatos (`EntityHypothesis`), ordenados por evidência fundida e hipótese; vazio quando nenhuma vista produziu claim. |
| `ambiguity_state` | `AmbiguityState`, **sempre** o que `derive_ambiguity_state` calcula dos registros: o contrato recusa outro valor. |
| `provenance` | `SemanticStateProvenance`: a regra de mapeamento e a política do primário. |
| `primary_hypothesis` | `EntityHypothesisRef` ou `None`: conveniência, só quando o estado é não ambíguo. |
| `attributes` | `EntityAttribute` ordenados por nome, valor, origem e derivação. |
| `uncertainty` | `EntityUncertainty`: cada registro de incerteza da fusão, com a evidência fundida que o reportou. |

Propriedades: `primary`, `alternative_hypotheses` (todas menos a primária) e `conflicts` (só as contradições).

Os campos `evidence_refs[]` e `fused_evidence_refs[]` do desenho conceitual não se repetem aqui: a evidência de cada claim vive em `EntityHypothesis.evidence` e as referências à evidência fundida em `Entity.evidence` (`EntityEvidenceLinks`), uma única fonte de verdade.

### `EntityHypothesis`

`fused_evidence_id` + `hypothesis_id` (identidade dentro da entidade), o `label` verbatim e a `evidence` (`HypothesisEvidence` de Semantic Fusion: claim, stance, papel e sinais tipados). Uma hipótese exige ao menos uma evidência que a suporte.

### `AmbiguityState`

| Valor | Quando |
| --- | --- |
| `conflicting` | Há uma contradição entre observações físicas. |
| `ambiguous` | Há ambiguidade ou quase empate, ou várias hipóteses de uma mesma evidência fundida sem nada que as decida. |
| `insufficient_evidence` | Nenhuma hipótese, ou um registro de evidência insuficiente. |
| `unambiguous` | Existe uma hipótese e nada compete com ela. |

A precedência é a da tabela, de cima para baixo.

### `EntityAttribute`

`name`, `value`, `origin`, `derivation_id` (regra ou estágio versionado que produziu o valor), `evidence` (`EvidenceReference` de contribuição e claim) e `support` (sinais tipados próprios, quando o produtor os emitiu; vazio significa "sem sinal", não suporte zero). Um atributo observado ou derivado sem evidência é recusado.

## Regras de mapeamento (`semantic_state_from_fused_evidence`)

Versionadas: `fused-evidence-semantic-state-v1`.

1. **Nada é descartado.** Toda hipótese, com toda a sua evidência (incluindo alternativas, claims ambíguas ou conflitantes, abstenções e sinais sem score), e todo registro de incerteza voltam exatamente como foram fundidos.
2. **Primária só quando justificada** (`unambiguous-single-hypothesis-v1`): exatamente uma hipótese e nenhuma competição. Caso contrário não há primária, e a entidade não recebe um label que a evidência não sustenta. O contrato recusa uma primária exposta enquanto o estado não é `unambiguous`, porque as alternativas ficariam escondidas.
3. **Um único atributo derivado.** `class`, derivado da hipótese primária (`primary-hypothesis-label-v1`, origem `derived`), citando exatamente as claims que a suportam. Nenhuma propriedade de senso comum, enriquecimento de ontologia ou conhecimento externo é adicionado: se `material` ou `condition` existirem, vêm de um estágio posterior que declare a regra.

O mapeamento é determinístico e não reexecuta percepção nem fusão.

## Validação

`tests/semantic_mapping/test_entity_semantic_state.py` usa **evidência fundida real**, construída pela acumulação baseline de Semantic Fusion (não registros feitos à mão), e cobre alternativas, contradição, quase empate, abstenção, apenas abstenções, vista sem claims, score ausente × zero, preservação integral, determinismo, persistência e as invariantes do contrato.
