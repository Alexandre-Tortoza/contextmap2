# Calibração de scores nativos (guarda de split)

Este documento descreve `src/contextmap/evaluation/score_calibration.py` (issue #573).

## Regra

Um score nativo de modelo — estatísticas de decoder do LocateAnything, o `predicted_iou` do
SAM2, scores de discovery — **não** é a probabilidade de uma saída estar correta. Enquanto
não houver uma calibração demonstrada, esses valores ficam **só como diagnóstico**: nenhum
contrato de grounding ou de refinamento tem campo `confidence`, probabilidade ou valor
calibrado (`tests/visual_perception/test_grounding_confidence_guard.py`), e nenhuma
`SemanticClaim` é criada a partir de um rótulo de grounding.

Promover um score a probabilidade exige, nesta ordem:

1. um reference set anotado e congelado, validado (`require_valid_reference_set`);
2. uma política de casamento positivo/negativo definida **antes** do ajuste;
3. um par de splits disjunto (`CalibrationSplitPair`): ajuste num split que não seja `TEST`,
   avaliação num split `TEST` retido, ambos do mesmo esquema e da mesma versão do reference
   set;
4. um calibrador versionado, com a identidade do reference set e do par de splits em que foi
   ajustado;
5. um relatório de confiabilidade (curva, ECE ou equivalente, Brier/log loss onde fizer
   sentido, precisão/recall por faixa de score, estratificação por classe/família de query e
   por fallback híbrido).

Só os itens 1 e 3 existem em código. O contrato de score calibrado (itens 4–5) **não é
criado** até que uma calibração seja aceita com dados reais: sem ele, é mecanicamente
impossível produzir um score calibrado sem identidade de calibrador.

## `CalibrationSplitPair`

`CalibrationSplitPair(calibration=SelectionBinding, evaluation=SelectionBinding)` recusa
(`CalibrationSplitError`):

- splits de reference sets (ou versões) diferentes, ou de esquemas diferentes;
- o mesmo split dos dois lados;
- ajuste num split `TEST`;
- avaliação num split que não seja `TEST`;
- qualquer amostra compartilhada entre os dois.

`CalibrationSplitPair.from_reference_set(validated, scheme_id=..., calibration_split=...,
evaluation_split=...)` só aceita um `ValidatedReferenceSet`, que já recusa esquemas com
amostra, observação física, unidade de agrupamento (sequência, cena, ...) ou janela temporal
adjacente dos dois lados (ver [`reference-integrity.md`](reference-integrity.md)). O par é
serializável (`to_record`/`from_record`) e revalida todas as regras ao ser relido.

## O que precisa de execução real

O relatório de confiabilidade e qualquer calibração do LocateAnything precisam do slice de
referência anotado e de execuções reais do modelo (#528/#576). Os campos diagnósticos que
essa avaliação vai usar estão documentados em
[`locateanything.md`](../../visual_perception/docs/locateanything.md#diagnósticos-nativos-e-calibração-573)
e em [`region-refinement.md`](../../visual_perception/docs/region-refinement.md).
